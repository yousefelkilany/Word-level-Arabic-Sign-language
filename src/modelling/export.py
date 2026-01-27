import os
import argparse

import onnx
import torch

from modelling.model import load_model, load_onnx_model, onnx_inference
from core.constants import DEVICE, FEAT_NUM, SEQ_LEN, FEAT_DIM


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--onnx_model_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    checkpoint_path = args.checkpoint_path
    assert checkpoint_path is not None, "--checkpoint_path is required"

    model = load_model(checkpoint_path, device=DEVICE)
    model.eval()

    torch_input = (torch.rand(1, SEQ_LEN, FEAT_NUM * FEAT_DIM, device=DEVICE),)
    torch_output = model(*torch_input)

    onnx_model_path = args.onnx_model_path or f"{checkpoint_path}.onnx"
    # batch_dim = torch.export.Dim("batch_size")
    onnx_model = torch.onnx.export(
        model,
        torch_input,
        onnx_model_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        # dynamic_shapes={"x": {0: batch_dim}},
        # dynamo=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    # if not onnx_model:
    if os.path.exists(onnx_model_path):
        print("✅ ONNX model exported successfully!")
    else:
        raise ValueError("❌ Failed to export ONNX model ❌")

    try:
        onnx.checker.check_model(onnx_model_path)
    except onnx.checker.ValidationError as e:
        print("The model is invalid:", e)
    else:
        print("✅ ONNX model checked! ✅")

    onnx_input = [tensor.numpy(force=True) for tensor in torch_input]
    ort_session = load_onnx_model(onnx_model_path)
    onnxruntime_output = onnx_inference(ort_session, onnx_input)

    if onnxruntime_output is None:
        raise Exception("❌ ONNX Runtime inference failed! ❌")

    assert len(torch_output) == len(onnxruntime_output), (
        "❌ Mismatch in number of outputs between PyTorch and ONNX Runtime ❌"
    )
    for torch_output, onnxruntime_output in zip(torch_output, onnxruntime_output):
        torch.testing.assert_close(
            torch_output, torch.tensor(onnxruntime_output), atol=1e-1, rtol=1e-1
        )

    print("✅ PyTorch🔥 and ONNX Runtime output matched! ✅")
