import torch
import numpy as np
from tqdm import tqdm

from utils import DEVICE, extract_num_words_from_checkpoint
from dataloader import prepare_dataloader
from model import load_onnx_model, onnx_inference
from export import cli

from sklearn.metrics import f1_score, accuracy_score

if __name__ == "__main__":
    args = cli()
    checkpoint_path = args.checkpoint_path
    onnx_model_path = args.onnx_model_path
    assert checkpoint_path is not None or onnx_model_path is not None, (
        "at least one of --checkpoint_path and --onnx_model_path must be passed"
    )
    onnx_model_path = onnx_model_path or f"{checkpoint_path}.onnx"

    num_words = extract_num_words_from_checkpoint(checkpoint_path)
    if not num_words:
        raise ValueError("Couldn't determine `num_words` for loading dataset")

    test_dl = prepare_dataloader("test", range(1, num_words + 1))

    ort_session = load_onnx_model(onnx_model_path, device=DEVICE)

    # test_acc = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for kps, labels in tqdm(test_dl, desc="Testing"):
            onnx_input = [tensor.numpy(force=True) for tensor in (kps,)]
            predicted = onnx_inference(ort_session, onnx_input)
            predicted = np.argmax(predicted, 1)
            # test_acc += (predicted == labels).sum().item()
            all_labels.extend(labels)
            all_predicted.extend(predicted)

        # test_acc /= len(test_dl.dataset)
        test_acc = accuracy_score(all_labels, all_predicted)
        test_f1 = f1_score(all_labels, all_predicted, average="weighted")
        print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print(f"Test F1 score: {test_f1:.4f} ({test_f1 * 100:.2f}%)")
