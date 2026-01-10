from functools import lru_cache
from os.path import join as os_join

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_preprocessing import calculate_num_chunks, prepare_raw_kps
from utils import KPS_DIR


class KArSLDataset(Dataset):
    def __init__(self, split, signers, selected_words):
        super().__init__()
        self.samples = []

        print(f"Building index map for {split} split...")
        for word in tqdm(selected_words, desc=f"Words - {split}"):
            for signer in signers:
                word_kps_path = os_join(
                    KPS_DIR, "all_kps", f"{signer}-{split}", f"{word:04}.npz"
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
        return prepare_raw_kps([self._load_file(path)[vid]])[0]

    def __getitem__(self, index):
        path, vid, chunk_idx, label = self.samples[index]
        processed_chunks = self._load_and_process_file(path, vid)
        return processed_chunks[chunk_idx], np.longlong(label)
