import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from core.constants import DEVICE, DatasetType, ModelSize, SplitType
from core.utils import extract_metadata_from_checkpoint
from data.dataloader import prepare_dataloader
from modelling.model import load_onnx_model, onnx_inference


def onnx_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("--num_signs", type=int, default=502)
    parser.add_argument(
        "--model_metadata", type=str, default=ModelSize.get_default().to_str()
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = onnx_cli()
    metadata = extract_metadata_from_checkpoint(cli_args.onnx_model_path)
    if metadata:
        num_signs, model_size = metadata

    num_signs = cli_args.num_signs or num_signs
    if not num_signs:
        raise ValueError("Couldn't determine `num_signs` for loading dataset")

    test_dl = prepare_dataloader(
        DatasetType.lazy, SplitType.test, ["01", "02", "03"], range(1, num_signs + 1)
    )
    ort_session = load_onnx_model(cli_args.onnx_model_path, device=DEVICE)

    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for kps, labels in tqdm(test_dl, desc="Testing"):
            onnx_input = [tensor.numpy(force=True) for tensor in (kps,)]
            predicted = onnx_inference(ort_session, onnx_input)
            if predicted is None:
                continue

            predicted = np.argmax(predicted, 1)
            all_labels.extend(labels)
            all_predicted.extend(predicted)

        test_acc = accuracy_score(all_labels, all_predicted)
        test_f1 = f1_score(all_labels, all_predicted, average="weighted")
        print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print(f"Test F1 score: {test_f1:.4f} ({test_f1 * 100:.2f}%)")
