# Makefile Commands

#development #automation #makefile

The `Makefile` provides shorthand commands for common tasks, abstracting away complex arguments and module paths.

## Primary Commands

| Command               | Description                                               | Code                                 |
| :-------------------- | :-------------------------------------------------------- | :----------------------------------- |
| `make train`          | Runs the single-GPU training loop.                        | `uv run -m modelling.train`          |
| `make parallel_train` | Runs distributed training on multiple GPUs.               | `uv run -m modelling.parallel_train` |
| `make export_model`   | Exports a checkpoint to ONNX. Requires `checkpoint_path`. | `uv run -m modelling.export ...`     |

## Data Processing

| Command                     | Description                                   |
| :-------------------------- | :-------------------------------------------- |
| `make preprocess_mmap_data` | Converts raw dataset to memory-mapped format. |
| `make prepare_npz_kps`      | Extracts keypoints from raw videos to .npz.   |
| `make generate_face_map`    | Generates symmetry map for face landmarks.    |

## Visualization

| Command                        | Description                                            |
| :----------------------------- | :----------------------------------------------------- |
| `make visualize_metrics`       | Generates plots for training/validation metrics.       |
| `make visualization_dashboard` | Launches the Streamlit dashboard for data exploration. |

## Parameter Overrides
You can pass arguments to make commands:

```bash
make export_model checkpoint_path=checkpoints/best.pth
```

## CPU Mode
Prefix any command with `cpu_` to force CPU usage:

```bash
make cpu_train
```

## Related Documentation

- [[source/modelling/train_py|train.py]]
- [[deployment/docker_setup|Docker Setup]]
