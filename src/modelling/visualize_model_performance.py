import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from core.constants import DEVICE, DatasetType, SplitType
from core.utils import extract_num_words_from_checkpoint
from data.dataloader import prepare_dataloader
from modelling.model import load_model


def visualize_metrics(checkpoint_path, test_dl, device=DEVICE, top_k_errors=20):
    """
    Generates a 3-part dashboard:
    1. A clean Heatmap (no text) to see overall diagonal sharpness.
    2. A Scatter plot identifying the worst performing classes.
    3. A Bar chart showing the most frequent specific confusions (e.g. 'Class A' mistaken for 'Class B').
    """

    print("Generating predictions for metrics...")

    model = load_model(checkpoint_path, device=device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for kps, labels in tqdm(test_dl, desc="Inference"):
            kps, labels = kps.to(device), labels.to(device)
            outputs = model(kps)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)

    ax_cm = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, ax=ax_cm, cmap="Blues", xticklabels=False, yticklabels=False, cbar=True
    )
    ax_cm.set_title(f"Global Confusion Matrix ({len(cm)} classes)")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")

    ax_scat = fig.add_subplot(gs[0, 1])
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_metrics = {int(k): v["f1-score"] for k, v in report.items() if k.isdigit()}
    df_scores = pd.DataFrame(
        list(class_metrics.items()),
        columns=["Class_ID", "F1_Score"],  # ty:ignore[invalid-argument-type]
    )
    sns.scatterplot(
        data=df_scores, x="Class_ID", y="F1_Score", ax=ax_scat, alpha=0.6, s=30
    )
    bad_classes = df_scores[df_scores["F1_Score"] < 0.5]
    if not bad_classes.empty:
        sns.scatterplot(
            data=bad_classes,
            x="Class_ID",
            y="F1_Score",
            ax=ax_scat,
            color="red",
            s=50,
            label="F1 < 0.5",
        )
    ax_scat.set_title("Per-Class Performance (F1 Score)")
    ax_scat.set_ylim(-0.05, 1.05)
    ax_scat.legend(loc="lower right")
    ax_scat.grid(True, alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, :])
    np.fill_diagonal(cm, 0)
    pairs = np.argwhere(cm > 0)
    counts = cm[pairs[:, 0], pairs[:, 1]]
    error_df = pd.DataFrame(
        {"True_Class": pairs[:, 0], "Pred_Class": pairs[:, 1], "Count": counts}
    )
    error_df = error_df.sort_values("Count", ascending=False).head(top_k_errors)
    error_df["Label"] = error_df.apply(
        lambda x: f"{int(x.True_Class)} -> {int(x.Pred_Class)}", axis=1
    )
    sns.barplot(
        data=error_df,
        x="Count",
        y="Label",
        ax=ax_bar,
        hue="Label",
        legend=False,
        palette="viridis",
    )
    ax_bar.set_title(
        f"Top {top_k_errors} Most Frequent Misclassifications (True -> Pred)"
    )
    ax_bar.set_xlabel("Number of Error Occurrences")

    plt.tight_layout()
    checkpoint_dir = os.path.dirname(checkpoint_path)
    save_path = os.path.join(
        checkpoint_dir, f"{os.path.basename(checkpoint_path)} - Model_Diagnostics.jpg"
    )
    plt.savefig(save_path, dpi=150)
    print(f"✅ Saved visual diagnostics to {save_path}")

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(
            checkpoint_dir,
            f"{os.path.basename(checkpoint_path)} - classification_report.csv",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    num_words = extract_num_words_from_checkpoint(args.checkpoint_path)
    if not num_words:
        raise ValueError("Number of words not found in checkpoint path")

    test_dl = prepare_dataloader(
        DatasetType.lazy, SplitType.test, signs=range(1, 1 + num_words)
    )
    visualize_metrics(args.checkpoint_path, test_dl)
