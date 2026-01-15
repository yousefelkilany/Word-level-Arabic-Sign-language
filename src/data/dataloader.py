from torch.utils.data import DataLoader, random_split

from data.lazy_dataset import LazyKArSLDataset
from data.mmap_dataset import MmapKArSLDataset


def prepare_lazy_dataloader(
    split, selected_words, signers=None, batch_size=64, shuffle=False, num_workers=0
):
    signers = signers or ["01", "02", "03"]
    ds = LazyKArSLDataset(split, signers, selected_words)
    if split == "test":
        print(f"{split.capitalize()} dataset size: {len(ds)}")
        return DataLoader(ds, batch_size=batch_size)

    train_size = int(len(ds) * 0.8)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    print(
        f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}"
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl


def prepare_lazy_dataloaders(
    selected_words, signers=None, batch_size=64, shuffle_train=True, num_workers=0
):
    train_dl, val_dl = prepare_lazy_dataloader(
        "train", selected_words, signers, batch_size, shuffle_train, num_workers
    )
    test_dl = prepare_lazy_dataloader(
        "test", selected_words, signers, batch_size, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl


def prepare_mmap_dataloader(split, batch_size=64, shuffle=False, num_workers=0):
    ds = MmapKArSLDataset(split)
    if split == "test":
        print(f"{split.capitalize()} dataset size: {len(ds)}")
        return DataLoader(ds, batch_size=batch_size)

    train_size = int(len(ds) * 0.8)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    print(
        f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}"
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl


def prepare_mmap_dataloaders(batch_size=64, shuffle_train=True, num_workers=0):
    train_dl, val_dl = prepare_mmap_dataloader(
        "train", batch_size, shuffle_train, num_workers
    )
    test_dl = prepare_mmap_dataloader("test", batch_size, num_workers=num_workers)
    return train_dl, val_dl, test_dl
