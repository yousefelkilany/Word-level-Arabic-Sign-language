from typing import Optional

import numpy as np
import onnxruntime
import torch
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from torch import nn

from core.constants import SEQ_LEN
from core.mediapipe_utils import FACE_NUM, HAND_NUM, POSE_NUM
from core.utils import extract_num_words_from_checkpoint


class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2, batch_first=True, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.layer_norm(x + self.dropout(self.lstm(x)[0]))


class SpatialGroupEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.pose_num = POSE_NUM * 3  # type: ignore
        self.face_num = FACE_NUM * 3  # type: ignore
        self.rh_num = self.lh_num = HAND_NUM * 3  # type: ignore

        # hs = 384, hs/48 = 8
        self.pose_proj = nn.Linear(self.pose_num, hidden_size * 5 // 48)  # 40
        self.face_proj = nn.Linear(self.face_num, hidden_size * 11 // 48)  # 88
        self.rh_proj = nn.Linear(self.rh_num, int(hidden_size * 16 // 48))  # 128
        self.lh_proj = nn.Linear(self.lh_num, int(hidden_size * 16 // 48))  # 128

        self.activation = nn.GELU()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        offset = self.pose_num + self.face_num

        pose = self.pose_proj(x[:, :, : self.pose_num])
        face = self.face_proj(x[:, :, self.pose_num : offset])
        rh = self.rh_proj(x[:, :, offset : (offset + self.rh_num)])
        lh = self.lh_proj(x[:, :, offset + self.rh_num :])

        x = self.activation(torch.cat((pose, face, rh, lh), dim=2))
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return torch.sum(x * self.attention_weights(x), dim=1)


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_lstm_blocks,
        num_classes,
        lstm_dropout_prob=0.3,
        attn_dropout_prob=0.3,
        network_dropout_prob=0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.embedding = SpatialGroupEmbedding(hidden_size)

        self.lstms = nn.Sequential(
            *[
                ResidualBiLSTMBlock(hidden_size, lstm_dropout_prob)
                for _ in range(num_lstm_blocks)
            ]
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=attn_dropout_prob,
            batch_first=True,
        )

        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        self.pool = AttentionPooling(hidden_size)

        self.dropout = nn.Dropout(network_dropout_prob)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        for lstm_block in self.lstms:
            x = lstm_block(x)

        x = self.attn_layer_norm(x + self.attention(x, x, x)[0])

        x = self.pool(x)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def get_model_instance(num_words, device="cpu"):
    hidden_size = 384
    num_lstm_blocks = 4
    model = AttentionBiLSTM(
        hidden_size,
        num_lstm_blocks,
        num_words,
        lstm_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        network_dropout_prob=0.5,
    )
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


def load_model(checkpoint_path, model=None, num_words=None, device="cpu") -> nn.Module:
    num_words = num_words or extract_num_words_from_checkpoint(checkpoint_path)
    assert model or num_words, "Either a model instance or `num_words` must be provided"

    model = model or get_model_instance(num_words, device)
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
