from itertools import product
from os.path import join as os_join
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from core.constants import FEAT_DIM, MMAP_PREPROCESSED_DIR, SplitType
from data.data_preparation import DataAugmentor, TSNSampler


class MmapKArSLDataset(Dataset):
    def __init__(
        self,
        split: SplitType,
        signers: list[str],
        signs: range,
        transforms: Optional[DataAugmentor] = None,
    ):
        super().__init__()

        self.split = split
        self.tsn_sampler = TSNSampler(mode=split)
        self.transform = transforms or DataAugmentor()

        data_path = os_join(MMAP_PREPROCESSED_DIR, f"{split}_X.mmap")
        label_path = os_join(MMAP_PREPROCESSED_DIR, f"{split}_y.npz")
        data_shape_path = os_join(MMAP_PREPROCESSED_DIR, f"{split}_X_shape.npy")
        data_seq_lens_path = os_join(
            MMAP_PREPROCESSED_DIR, f"{split}_X_map_samples_lens.npy"
        )

        self.y: np.ndarray = np.load(label_path)["arr_0"]
        X_shape = tuple(int(dim) for dim in np.load(data_shape_path))
        self.X = np.memmap(data_path, dtype="float32", mode="r", shape=X_shape)
        X_map_samples_lens = np.load(data_seq_lens_path, allow_pickle=True).item()
        self.X_lens: np.ndarray = np.concatenate(
            [
                X_map_samples_lens[int(sign)][int(signer)]
                for sign, signer in product(signs, signers)
            ]
        )
        self.X_offsets = np.concatenate(([0], self.X_lens.cumsum()[:-1]))

        assert len(self.X_lens) == len(self.y), (
            f"[ERROR] Mismatched length of X samples and y labels\n{self.X_lens.shape = }\n{self.y.shape = }"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        chunk_idx, chunks_len = self.X_offsets[index], self.X_lens[index]
        sample = self.tsn_sampler(self.X[chunk_idx : chunk_idx + chunks_len])
        return self.transform(sample.reshape(-1, FEAT_DIM)), np.longlong(self.y[index])
