from os.path import join as os_join

import numpy as np
from torch.utils.data import Dataset

from utils import PREPROCESSED_DIR


class KArSLDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        data_path = os_join(PREPROCESSED_DIR, f"{split}_X.mmap")
        label_path = os_join(PREPROCESSED_DIR, f"{split}_y.npy")
        data_shape_path = os_join(PREPROCESSED_DIR, f"{split}_X_shape.npy")

        self.y = np.load(label_path)
        X_shape = np.load(data_shape_path)
        self.X = np.memmap(data_path, dtype="float32", mode="r", shape=X_shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
