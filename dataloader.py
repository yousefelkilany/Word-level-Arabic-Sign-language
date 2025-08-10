import numpy as np
from tqdm.notebook import tqdm
from os.path import join as os_join

from torch.utils.data import Dataset

KPS_DIR = "/kaggle/working/karsl-kps"
SEQ_LEN = 60
FEAT_NUM = 184


class KArSLDataset(Dataset):
    def __init__(self, split, signers, selected_words):
        super().__init__()

        self.X, self.y = self._load_kps(split, signers, selected_words)
        self.X = self.prepare_kps(self.X)
        self.y = self.prepare_labels(self.y, self.X)
        self.X = np.concatenate(self.X, dtype=np.float32)
        self.num_classes = self.y.shape[1]
        print(self.X.shape, self.y.shape)

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
        gt_labels = np.zeros((len(y), np.max(self.y) + 1))
        gt_labels[np.arange(len(y)), y] = 1
        return gt_labels

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
        # print(self.X.shape, self.y.shape)
        # print(self.X[idx].shape, self.y[idx].shape)
        return self.X[idx], self.y[idx]
