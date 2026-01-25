from functools import lru_cache
from os.path import join as os_join
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from core.constants import NPZ_KPS_DIR, SplitType
from data.data_preparation import DataAugmentor, TSNSampler


class LazyKArSLDataset(Dataset):
    def __init__(
        self,
        split: SplitType,
        signers: list[str],
        signs: range,
        train_transforms: Optional[DataAugmentor] = None,
        val_transforms: Optional[DataAugmentor] = None,
        test_transforms: Optional[DataAugmentor] = None,
    ):
        super().__init__()

        self.split = split
        self.tsn_sampler = TSNSampler(mode=split)
        self.transform = DataAugmentor()
        match split:
            case SplitType.train:
                self.transform = train_transforms or self.transform
            case SplitType.val:
                self.transform = val_transforms or self.transform
            case SplitType.test:
                self.transform = test_transforms or self.transform

        self.samples = []
        print(f"Building index map for {split} split...")
        for word in tqdm(signs, desc=f"Words - {split}"):
            for signer in signers:
                word_kps_path = os_join(
                    NPZ_KPS_DIR, "all_kps", f"{signer}-{split}", f"{word:04}.npz"
                )

                try:
                    word_kps_data = np.load(word_kps_path, allow_pickle=True)
                except FileNotFoundError:
                    continue

                for vid, kps in word_kps_data.items():
                    num_chunks = 1  # calculate_num_chunks(kps.shape[0])
                    self.samples.extend(
                        ((word_kps_path, vid, i, word - 1) for i in range(num_chunks))
                    )

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=1024)
    def _load_file(self, path):
        """Loads a single sequence from a file and processes it."""
        return np.load(path, allow_pickle=True)

    def __getitem__(self, index):
        path, vid, chunk_idx, label = self.samples[index]
        # processed_chunks = prepare_raw_kps([self._load_file(path)[vid]])

        # return self.transform(processed_chunks[0][chunk_idx]), np.longlong(label)
        return None, -1
