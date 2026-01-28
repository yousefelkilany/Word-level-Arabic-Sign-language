import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from core.constants import DEVICE, DatasetType, SplitType
from core.utils import extract_num_signs_from_checkpoint
from data.dataloader import prepare_dataloader
from modelling.export import checkpoint_cli
from modelling.model import load_onnx_model, onnx_inference

if __name__ == "__main__":
    cli_args = checkpoint_cli()
    checkpoint_path = cli_args.checkpoint_path
    onnx_model_path = cli_args.onnx_model_path
    assert checkpoint_path is not None or onnx_model_path is not None, (
        "at least one of --checkpoint_path and --onnx_model_path must be passed"
    )
    onnx_model_path = onnx_model_path or f"{checkpoint_path}.onnx"

    num_signs = extract_num_signs_from_checkpoint(checkpoint_path)
    if not num_signs:
        raise ValueError("Couldn't determine `num_signs` for loading dataset")

    test_dl = prepare_dataloader(
        DatasetType.lazy, SplitType.test, ["01", "02", "03"], range(1, num_signs + 1)
    )

    ort_session = load_onnx_model(onnx_model_path, device=DEVICE)

    # test_acc = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for kps, labels in tqdm(test_dl, desc="Testing"):
            onnx_input = [tensor.numpy(force=True) for tensor in (kps,)]
            predicted = onnx_inference(ort_session, onnx_input)
            if not predicted:
                continue

            predicted = np.argmax(predicted, 1)
            # test_acc += (predicted == labels).sum().item()
            all_labels.extend(labels)
            all_predicted.extend(predicted)

        # test_acc /= len(test_dl.dataset)
        test_acc = accuracy_score(all_labels, all_predicted)
        test_f1 = f1_score(all_labels, all_predicted, average="weighted")
        print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print(f"Test F1 score: {test_f1:.4f} ({test_f1 * 100:.2f}%)")
