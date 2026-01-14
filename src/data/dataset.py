from torch.utils.data import Dataset

from data.dataset_preprocessing import load_raw_kps, prepare_labels, prepare_raw_kps


class KArSLDataset(Dataset):
    def __init__(self, split, signers, selected_words):
        super().__init__()
        self.X, self.y = load_raw_kps(split, signers, selected_words)
        self.y = prepare_labels(self.y, self.X)
        self.X = prepare_raw_kps(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
