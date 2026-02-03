import math
from typing import Optional

import numpy as np
import onnxruntime
import torch
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from torch import nn

from core.constants import FEAT_DIM, SEQ_LEN, ModelSize
from core.mediapipe_utils import FACE_NUM, HAND_NUM, POSE_NUM
from core.utils import extract_metadata_from_checkpoint


class STTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_prob=0.3):
        super().__init__()

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_prob
        )
        self.spatial_norm = nn.LayerNorm(embed_dim)

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_prob
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_prob),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, G, D = x.shape

        x_s = x.contiguous().view(B * T, G, D)
        x_s = self.spatial_norm(x_s)
        attn_out, _ = self.spatial_attention(x_s, x_s, x_s)
        x_s = x_s + attn_out
        x = x_s.view(B, T, G, D)

        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * G, T, D)
        x_t = self.temporal_norm(x_t)
        attn_out, _ = self.temporal_attention(x_t, x_t, x_t)
        x_t = x_t + attn_out
        x = x_t.view(B, G, T, D).permute(0, 2, 1, 3)

        return x + self.mlp(self.mlp_norm(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(div_term * (-math.log(1e5) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :, :]  # type: ignore[non-subscriptable]


class GroupTokenEmbedding(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()

        self.pose_num = POSE_NUM * FEAT_DIM  # type: ignore
        self.face_num = FACE_NUM * FEAT_DIM  # type: ignore
        self.rh_num = self.lh_num = HAND_NUM * FEAT_DIM  # type: ignore

        self.pose_proj = nn.Linear(self.pose_num, token_dim)
        self.face_proj = nn.Linear(self.face_num, token_dim)
        self.rh_proj = nn.Linear(self.rh_num, token_dim)
        self.lh_proj = nn.Linear(self.lh_num, token_dim)

        self.layer_norm = nn.LayerNorm(token_dim)
        self.activation = nn.GELU()

        self.part_embed = nn.Parameter(torch.zeros(1, 1, 4, token_dim))
        nn.init.trunc_normal_(self.part_embed, std=0.02)

        total_features = self.pose_num + self.face_num + self.rh_num + self.lh_num
        self.input_bn = nn.BatchNorm1d(total_features)

    def forward(self, x):
        offset = self.pose_num + self.face_num

        x = x.permute(0, 2, 1)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)

        pose = self.pose_proj(x[:, :, : self.pose_num])
        face = self.face_proj(x[:, :, self.pose_num : offset])
        rh = self.rh_proj(x[:, :, offset : offset + self.rh_num])
        lh = self.lh_proj(x[:, :, offset + self.rh_num :])

        tokens = torch.stack((pose, face, rh, lh), dim=2)
        x = self.layer_norm(tokens + self.part_embed)
        return self.activation(x)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return torch.sum(x * self.attention_weights(x), dim=1)


class STTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_size: ModelSize,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes  # ty:ignore[unresolved-attribute]

        self.model_size = model_size  # ty:ignore[unresolved-attribute]
        head_size, num_heads, num_layers = self.model_size.params
        embed_dim = head_size * num_heads

        self.embedding = GroupTokenEmbedding(embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                STTransformerBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attention_pool = AttentionPooling(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=2)

        x = self.attention_pool(x)
        logits = self.fc(x)
        return logits


def get_model_instance(
    num_signs: int, model_size: ModelSize, device="cpu"
) -> STTransformer:
    model = STTransformer(num_signs, model_size, 0.5)
    return model.to(device)


def save_model(checkpoint_path, model, optimizer, scheduler):
    try:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_path,
        )
        return True
    except Exception as e:
        print(f"[Save model checkpoint - error]: {e = }")


def load_model(
    checkpoint_path,
    model: Optional[STTransformer] = None,
    num_signs: Optional[int] = None,
    model_size: Optional[ModelSize] = None,
    device: str = "cpu",
) -> STTransformer:
    metadata = extract_metadata_from_checkpoint(checkpoint_path)
    if metadata:
        num_signs = num_signs or metadata[0]
        model_size = model_size or metadata[1]

    if not model:
        assert num_signs and model_size, (
            "Model metadata, num signs and model metadata(head_size, num_heads, num_layers) must be provided"
        )
        model = get_model_instance(num_signs, model_size, device=device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device(device))["model"]
    )
    return model


def load_onnx_model(onnx_model_path, device="cpu") -> InferenceSession:
    providers = [
        ["CPUExecutionProvider"],
        [("CUDAExecutionProvider", {"device_id": 0})],
    ][int(device == "cuda")]
    return onnxruntime.InferenceSession(onnx_model_path, providers=providers)


def onnx_inference(
    ort_session: InferenceSession, input_data: list[np.ndarray]
) -> Optional[np.ndarray]:
    inputs = {}
    for ort_input, input_tensor in zip(ort_session.get_inputs(), input_data):
        if (num_elements := (SEQ_LEN - input_tensor.shape[1])) > 0:
            elements_to_add = np.repeat(input_tensor[:, -1:, :], num_elements, axis=1)
            input_tensor = np.concatenate((input_tensor, elements_to_add), axis=1)

        inputs[ort_input.name] = input_tensor

    output_name = ort_session.get_outputs()[0].name
    outputs = ort_session.run([output_name], inputs)[0]
    if isinstance(outputs, np.ndarray):
        return outputs
