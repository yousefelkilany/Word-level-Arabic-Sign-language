import argparse
import torch
import onnx

from utils import DEVICE, FEAT_NUM, SEQ_LEN
from model import load_model, load_onnx_model, onnx_inference


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

    torch_input = (torch.rand(2, SEQ_LEN, FEAT_NUM * 3, device=DEVICE),)
    torch_output = model(*torch_input)

    onnx_model_path = args.onnx_model_path or f"{checkpoint_path}.onnx"
    batch_dim = torch.export.Dim("batch_size")
    onnx_model = torch.onnx.export(
        model,
        torch_input,
        onnx_model_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {0: batch_dim}},
        dynamo=True,
    )
    if not onnx_model:
        raise ValueError("Failed to export ONNX model")

    try:
        onnx.checker.check_model(onnx_model.model_proto)
    except onnx.checker.ValidationError as e:
        print("The model is invalid:", e)
    else:
        print("✅ ONNX model checked! ✅")

    onnx_input = [tensor.numpy(force=True) for tensor in torch_input]
    ort_session = load_onnx_model(onnx_model_path)
    onnxruntime_output = onnx_inference(ort_session, onnx_input)

    # print(f"{torch_output.shape = }")
    # print(f"{onnxruntime_output.shape = }")
    # print("Torch output:", torch_output[0])
    # print("ONNX Runtime output:", onnxruntime_output[0])

    assert len(torch_output) == len(onnxruntime_output), (
        "Mismatch in number of outputs between PyTorch and ONNX Runtime"
    )
    for torch_output, onnxruntime_output in zip(torch_output, onnxruntime_output):
        torch.testing.assert_close(
            torch_output, torch.tensor(onnxruntime_output), atol=1e-1, rtol=1e-1
        )

    print("✅ PyTorch🔥 and ONNX Runtime output matched! ✅")
