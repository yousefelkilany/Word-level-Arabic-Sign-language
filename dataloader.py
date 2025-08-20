import numpy as np
from os.path import join as os_join

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils import tqdm, KPS_DIR, SEQ_LEN, FEAT_NUM


class KArSLDataset(Dataset):
    def __init__(self, split, signers, selected_words, device="cpu"):
        super().__init__()
        self.X, self.y = self._load_kps(split, signers, selected_words)
        self.X = self.prepare_kps(self.X)
        self.y = self.prepare_labels(self.y, self.X) - 1
        self.X = np.concatenate(self.X, dtype=np.float32)

        self.X = torch.from_numpy(self.X).to(device)
        self.y = torch.from_numpy(self.y).to(device)

    def prepare_kps(self, X):
        def pad_split_seq(kps):
            # Pad sequences (with length < SEQ_LEN) to SEQ_LEN, no matter what is its length.
            kps_len = kps.shape[0]
            if SEQ_LEN > kps_len:
                kps = np.concatenate([kps, np.tile(kps[-1], (SEQ_LEN - kps_len, 1, 1))])

            # If sequence is longer, slice it into x sequences with the last slice filled if
            # it's 2/3 of SEQ_LEN, otherwise it's too short to be padded and is dropped.
            elif kps_len % SEQ_LEN >= (SEQ_LEN * 2 // 3):
                tile_cnt = (kps_len // SEQ_LEN + 1) * SEQ_LEN - kps_len
                kps = np.concatenate([kps, np.tile(kps[-1], (tile_cnt, 1, 1))])

            else:
                kps = kps[: (kps_len // SEQ_LEN) * SEQ_LEN]

            # Collapse last two dimensions, 184x3 to 552
            kps = kps.reshape(-1, SEQ_LEN, FEAT_NUM * 3)
            return np.nan_to_num(kps, nan=0.0, posinf=0.0, neginf=0.0)

        return [pad_split_seq(kps) for kps in X]

    def prepare_labels(self, y, X):
        y = [y_ for y_, x in zip(y, X) for _ in range(x.shape[0])]
        return np.array(y, dtype=np.longlong)

    def _load_kps(self, split, signers, selected_words):
        X, y = [], []
        for word in tqdm(selected_words, desc=f"Words - {split}"):
            vids_cnt = 0
            for signer in signers:
                word_kps_path = os_join(
                    KPS_DIR, "all_kps", f"{signer}-{split}", f"{word:04}.npz"
                )
                word_kps = np.load(word_kps_path, allow_pickle=True)
                X.extend([kp for kp in word_kps.values()])
                vids_cnt += len(word_kps)
            y.extend([word] * vids_cnt)

        return X, np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloader(
    split, selected_words, signers=None, batch_size=64, shuffle=False, device="cpu"
):
    signers = signers or ["01", "02", "03"]
    ds = KArSLDataset(split, signers, selected_words, device=device)
    if split == "test":
        print(f"{split.capitalize()} dataset size: {len(ds)}")
        return DataLoader(ds, batch_size=batch_size)

    train_size = int(len(ds) * 0.8)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    print(
        f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}"
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    return train_dl, val_dl


def prepare_dataloaders(
    selected_words, signers=None, batch_size=64, shuffle_train=True, device="cpu"
):
    train_dl, val_dl = prepare_dataloader(
        "train", selected_words, signers, batch_size, shuffle_train, device
    )
    test_dl = prepare_dataloader(
        "test", selected_words, signers, batch_size, device=device
    )
    return train_dl, val_dl, test_dl
