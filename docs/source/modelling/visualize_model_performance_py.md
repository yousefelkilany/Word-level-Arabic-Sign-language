---
title: visualize_model_performance.py
date: 2026-01-28
lastmod: 2026-02-01
src_hash: deb020781b68923418593d774ad40cb6c3ace7019d74cb873f5e209118e79305
aliases: ["Training Metrics Visualization", "Performance Analysis Plots"]
---

# visualize_model_performance.py

#source #modelling #visualization #analytics

**File Path**: `src/modelling/visualize_model_performance.py`

**Purpose**: Generates advanced visual diagnostics and classification reports for trained models.

## Overview

The `visualize_metrics` function evaluates a model's performance on a test dataset and produces a comprehensive 3-part visual dashboard for error analysis.

## Visual Dashboard Components

1. **Global Confusion Matrix (Heatmap)**:
   - Visualizes overall diagonal sharpness across all classes.
   - Axes are unlabeled (indices only) to keep the view clean for high class counts (e.g., 502).

2. **Per-Class Performance (Scatter Plot)**:
   - Plots the F1-Score for every class ID.
   - Highlights underperforming classes (F1 < 0.5) in **red** for quick identification.

3. **Top Misclassifications (Bar Chart)**:
   - Specifically identifies the top 20 most frequent "True -> Predicted" error pairs.
   - Helps identify semantic similarities causing model confusion.

## Outputs

- **Visual Dashboard**: Saved as `{checkpoint_name}-Model_Diagnostics.jpg` in the checkpoint directory.
- **Classification Report**: Saved as `{checkpoint_name} - classification_report.csv` containing precision, recall, and f1-score for all classes.

## Usage

```bash
# Can be called via the shared CLI (compatible with onnx_benchmark)
python src/modelling/visualize_model_performance.py --checkpoint_path checkpoints/best.pth
```

## Related Documentation

- [[train_py|train.py]] - Automated trigger after training completion.
- [[../core/utils_py|utils.py]] - Metadata extraction from checkpoint paths.
