import numpy as np
from tqdm.notebook import tqdm
from os.path import join as os_join

from torch.utils.data import Dataset, DataLoader, random_split

KPS_DIR = "/kaggle/working/karsl-kps"
SEQ_LEN = 60
FEAT_NUM = 184


class KArSLDataset(Dataset):
    def __init__(self, split, signers, selected_words):
        super().__init__()
        self.X, self.y = self._load_kps(split, signers, selected_words)
        self.X = self.prepare_kps(self.X)
        self.y = self.prepare_labels(self.y, self.X) - 1
        self.X = np.concatenate(self.X, dtype=np.float32)

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
            return kps.reshape(-1, SEQ_LEN, FEAT_NUM * 3)

        return [pad_split_seq(kps) for kps in X]

    def prepare_labels(self, y, X):
        y = [y_ for y_, x in zip(y, X) for _ in range(x.shape[0])]
        return np.array(y)

    def _load_kps(self, split, signers, selected_words):
        X, y = [], []
        for word in tqdm(selected_words, desc="Words"):
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


def prepare_dataloaders(signers, selected_words, shuffle_train=True):
    train_val = KArSLDataset("train", signers, selected_words)
    train_size = int(len(train_val) * 0.8)
    val_size = len(train_val) - train_size
    train_ds, val_ds = random_split(train_val, [train_size, val_size])

    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    test_ds = KArSLDataset("test", signers, selected_words)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, val_dl, test_dl
