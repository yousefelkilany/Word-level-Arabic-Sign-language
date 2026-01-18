from functools import lru_cache
from os.path import join as os_join
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from core.constants import NPZ_KPS_DIR
from data.data_augmentation import AlbumentationsWrapper
from data.dataset_preprocessing import calculate_num_chunks, prepare_raw_kps


class LazyKArSLDataset(Dataset):
    def __init__(
        self,
        split,
        signers,
        selected_words,
        train_transforms: Optional[AlbumentationsWrapper] = None,
        val_transforms: Optional[AlbumentationsWrapper] = None,
        test_transforms: Optional[AlbumentationsWrapper] = None,
    ):
        super().__init__()
        self.samples = []
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.split = split

        print(f"Building index map for {split} split...")
        for word in tqdm(selected_words, desc=f"Words - {split}"):
            for signer in signers:
                word_kps_path = os_join(
                    NPZ_KPS_DIR, "all_kps", f"{signer}-{split}", f"{word:04}.npz"
                )

                try:
                    word_kps_data = np.load(word_kps_path, allow_pickle=True)
                except FileNotFoundError:
                    continue

                for vid, kps in word_kps_data.items():
                    num_chunks = calculate_num_chunks(kps.shape[0])
                    self.samples.extend(
                        ((word_kps_path, vid, i, word - 1) for i in range(num_chunks))
                    )

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=1024)
    def _load_file(self, path):
        """Loads a single sequence from a file and processes it."""
        return np.load(path, allow_pickle=True)

    def _load_and_process_file(self, path, vid):
        """Loads a single sequence from a file and processes it."""
        outputs = prepare_raw_kps([self._load_file(path)[vid]])
        return outputs[0]

    def __getitem__(self, index):
        path, vid, chunk_idx, label = self.samples[index]
        processed_chunks = self._load_and_process_file(path, vid)
        kps, label = processed_chunks[chunk_idx], np.longlong(label)
        match self.split:
            case "train":
                kps = self.train_transforms(kps) if self.train_transforms else kps
            case "val":
                kps = self.val_transforms(kps) if self.val_transforms else kps
            case "test":
                kps = self.test_transforms(kps) if self.test_transforms else kps
        return kps, label
