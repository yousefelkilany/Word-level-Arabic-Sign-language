# Configuration Options (Constants)

#reference #configuration #constants

Beyond environment variables, the system logic is governed by internal constants defined in `src/core/constants.py`.

## Data Dimensions

- **`SEQ_LEN`**: `30`
    - The number of frames in a sequence used for inference.
- **`FEAT_DIM`**: `1024` (or similar, dependent on feature vector size)
    - Dimensionality of the input feature vector per frame.

## Directories

- **`KARSL_DATA_DIR`**: `data/`
    - Root directory for raw and processed data.
- **`LANDMARKERS_DIR`**: `landmarkers/`
    - Storage for MediaPipe `.task` model files.
- **`MODELS_DIR`**: `models/`
    - Storage for trained PyTorch/ONNX checkpoints.

## Feature Indices

Specific indices are used to slice the flattened keypoint vector:
- **Pose**: First block (33 points).
- **Face**: Second block (Selected subset).
- **Hands**: Final blocks (Left/Right).

## Related Documentation

- [[../source/core/constants_py|constants.py Source]]
- [[../deployment/environment_configuration|Environment Variables]]
