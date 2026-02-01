from functools import lru_cache
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from core.constants import FEAT_DIM, NPZ_KPS_DIR, SplitType, os_join
from data.data_preparation import DataAugmentor, TSNSampler


class LazyKArSLDataset(Dataset):
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

        self.samples = []
        print(f"Building index map for {split} split...")
        for sign in tqdm(signs, desc=f"Signs - {split}"):
            for signer in signers:
                sign_kps_path = os_join(
                    NPZ_KPS_DIR, f"{signer}-{split}", f"{sign:04}.npz"
                )

                try:
                    sign_kps_data = np.load(sign_kps_path, allow_pickle=True)
                except FileNotFoundError:
                    continue

                self.samples.extend(
                    [(signer, vid, sign - 1) for vid in sign_kps_data.keys()]
                )

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=1024)
    def _load_file(self, path):
        """Loads a single sequence from a file and processes it."""
        return np.load(path, allow_pickle=True)

    def __getitem__(self, index):
        signer, vid, label = self.samples[index]
        path = os_join(NPZ_KPS_DIR, f"{signer}-{self.split}", f"{label + 1:04}.npz")
        sample = self.tsn_sampler(self._load_file(path)[vid])
        return self.transform(sample.reshape(-1, FEAT_DIM)), np.longlong(label)
