from typing import Optional

from torch.utils.data import DataLoader, random_split

from core.constants import DatasetType, SplitType
from data import DataAugmentor, LazyKArSLDataset, MmapKArSLDataset


def prepare_dataloader(
    dataset_type: DatasetType,
    split: SplitType,
    signers: Optional[list[str]] = None,
    signs: Optional[range] = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    train_transforms: Optional[DataAugmentor] = None,
):
    signers = signers or ["01", "02", "03"]
    signs = signs or range(1, 503)

    match dataset_type:
        case DatasetType.lazy:
            dataset = LazyKArSLDataset(split, signers, signs, train_transforms)
        case DatasetType.mmap:
            dataset = MmapKArSLDataset(split, signers, signs, train_transforms)

    if split == SplitType.test:
        print(f"Test dataset size: {len(dataset)}")
        return DataLoader(dataset, batch_size=batch_size)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(
        f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}"
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl


def prepare_dataloaders(
    dataset_type: DatasetType,
    signers: Optional[list[str]] = None,
    signs: Optional[range] = None,
    train_transforms: Optional[DataAugmentor] = None,
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
):
    train_dl, val_dl = prepare_dataloader(
        dataset_type,
        SplitType.train,
        signers,
        signs,
        batch_size,
        shuffle_train,
        num_workers,
        train_transforms,
    )
    test_dl = prepare_dataloader(
        dataset_type,
        SplitType.test,
        signers,
        signs,
        batch_size,
        num_workers=num_workers,
    )
    return train_dl, val_dl, test_dl
