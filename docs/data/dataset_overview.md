# Dataset Overview

#data #dataset #karsl

The project utilizes the **KArSL-502** (King Saud University Arabic Sign Language) dataset, a large-scale word-level sign language dataset.

## Dataset Characteristics

- **Size**: 502 classes (Arabic sign words).
- **Subjects**: 3 distinct signers (01, 02, 03).
- **Structure**:
  - **Raw Video**: Folder structure `signer/split/word/video.mp4`.
  - **Keypoints**: Extracted MediaPipe landmarks stored as `.npz` files.

## Directory Structure

The processed dataset is expected to follow this hierarchy:

```text
data/
├── 01/             # Signer 01
│   ├── train/
│   │   ├── 0001/   # Word ID 1
│   │   └── ...
│   └── test/
├── 02/             # Signer 02
└── 03/             # Signer 03
```

## Keypoint Data Format

Each video is converted into a sequence of keypoints with the following shape:
`(SEQ_LEN, TOTAL_FEATURES)`

Where `TOTAL_FEATURES` is the flattened vector of all landmarks:

- **Pose**: 33 points × 3 coords (x, y, z)
- **Face**: Selected subset points × 3 coords
- **Hands**: 21 points × 3 coords × 2 hands

## Related Documentation

- [[../source/data/mmap_dataset_py|mmap_dataset.py Source Code]]
- [[../source/core/constants_py|constants.py Source Code]]
- [[../reference/dataset_citation|Dataset Citation]]
