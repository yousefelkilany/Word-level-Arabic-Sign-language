from os.path import join as os_join
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from core.constants import INPUT_PREPROCESSED_DIR, SplitType
from data.data_augmentation import AlbumentationsWrapper


class MmapKArSLDataset(Dataset):
    def __init__(
        self,
        split: SplitType,
        signers: list[str],
        signs: range,
        train_transforms: Optional[AlbumentationsWrapper] = None,
        val_transforms: Optional[AlbumentationsWrapper] = None,
        test_transforms: Optional[AlbumentationsWrapper] = None,
    ):
        super().__init__()

        self.split = split
        self.transform = AlbumentationsWrapper()
        match split:
            case SplitType.train:
                self.transform = train_transforms or self.transform
            case SplitType.val:
                self.transform = val_transforms or self.transform
            case SplitType.test:
                self.transform = test_transforms or self.transform

        data_path = os_join(INPUT_PREPROCESSED_DIR, f"{split}_X.mmap")
        label_path = os_join(INPUT_PREPROCESSED_DIR, f"{split}_y.npy")
        data_shape_path = os_join(INPUT_PREPROCESSED_DIR, f"{split}_X_shape.npy")
        data_seq_lens_path = os_join(INPUT_PREPROCESSED_DIR, f"{split}_X_lens.npy")

        self.y: np.ndarray = np.load(label_path)
        X_shape: np.ndarray = np.load(data_shape_path)
        self.X = np.memmap(data_path, dtype="float32", mode="r", shape=X_shape)
        self.X_lens: np.ndarray = np.load(data_seq_lens_path)
        self.X_offsets = np.concatenate(([0], self.X_lens.cumsum()[:-1]))

        assert len(self.X_lens) == len(self.y), (
            "mismatched length of X samples and y labels"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        chunk_idx, chunks_len = self.X_offsets[index], self.X_lens[index]
        if chunks_len > 1:
            # TODO: find a good way to selet which seq from fragemented sign sequence
            chunk_idx = np.random.randint(
                chunk_idx, chunk_idx + chunks_len
            )  # uniformly select random seq
        return self.transform(self.X[chunk_idx]), np.longlong(self.y[index])
