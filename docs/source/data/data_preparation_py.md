---
title: data_preparation.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Preprocessing Utilities", "Temporal and Spatial Augmentation"]
---

# data_preparation.py

#source #data #pytorch #preprocessing

**File Path**: `src/data/data_preparation.py`

**Purpose**: Essential preprocessing utilities for sampling video frames (Time) and augmenting keypoints (Space).

## Classes

### `TSNSampler`

**Purpose**: Implements Temporal Segment Network (TSN) sampling strategy to extract a fixed-length sequence (`SEQ_LEN`) from variable-length videos.

#### `__init__(target_len, mode, jitter_scale)`
- `target_len`: 50 (default)
- `mode`: `train` (random jitter) or `val/test` (center crop equivalents)

#### `__call__(kps)`
**Process**:
1. Divides video into `target_len` equal segments.
2. **Train**: Randomly selects one frame within each segment (jitter).
3. **Test**: Selects the center frame of each segment.
4. **Interpolation**: Uses simple linear interpolation (mixing previous and next frame) to handle sub-integer indices.

### `DataAugmentor`

**Purpose**: Spatial augmentation for skeleton data.

#### `__init__(...)`
**Hyperparameters**:
- `p_flip`: 0.5 (Horizontal Flip probability)
- `rotate_range`: (-15, 15) degrees
- `scale_range`: (0.85, 1.15)
- `shift_range`: (-0.1, 0.1)

#### `_apply_hflip(sequence)`
**Logic**:
1. Multiplies X-coordinate by -1.
2. **Permutation**: Swaps Left/Right body parts using index maps.
   - Example: Left Hand indices <-> Right Hand indices.
   - Uses `FACE_SYMMETRY_MAP_PATH` for the 468 face landmarks.

#### `_apply_affine(kps)`
**Logic**:
1. Generates random Rotation matrix (2x2).
2. Generates random Scale factor (scalar).
3. Generates random Shift vector (2D).
4. `New_KPs = (Old_KPs @ Rotation) * Scale + Shift`.

## Usage Example

```python
# Initialize
sampler = TSNSampler(mode=SplitType.train)
augmentor = DataAugmentor()

# Process
sampled_seq = sampler(raw_numpy_sequence)
final_seq = augmentor(sampled_seq)
```

## Related Documentation

**Depends On**:
- [[../../source/core/constants_py|constants.py]] - `FACE_SYMMETRY_MAP_PATH`
- [[../../source/core/mediapipe_utils_py|mediapipe_utils.py]] - `KP2SLICE`

**Used By**:
- [[../../source/data/lazy_dataset_py|lazy_dataset.py]]
- [[../../source/data/mmap_dataset_py|mmap_dataset.py]]
