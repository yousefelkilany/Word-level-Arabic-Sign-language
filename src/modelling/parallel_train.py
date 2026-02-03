import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from core.constants import DatasetType, ModelSize, SplitType, get_model_size
from data import DataAugmentor, LazyKArSLDataset
from data.dataloader import prepare_dataloader
from modelling.model import get_model_instance
from modelling.train import train, train_cli, visualize_metrics


def setup(rank, world_size):
    """Sets up the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Standard port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_training(
    rank: int,
    world_size: int,
    signers: list[str],
    signs: range,
    num_epochs: int,
    model_size: ModelSize,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)

    batch_size = 64
    dataset = LazyKArSLDataset(
        SplitType.train, signers=signers, signs=signs, transforms=DataAugmentor()
    )
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
    )

    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=4,
    )

    model = get_model_instance(len(signs), model_size, device=device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    lr = 1e-3 * torch.sqrt(torch.tensor(world_size)).item()
    weight_decay = 1e-4
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=3)

    best_checkpoint = train(
        model,
        loss,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
        num_epochs,
        str(device),
        rank=rank,
        train_sampler=train_sampler,
    )

    print(f"Best model checkpoint: {best_checkpoint}")

    if rank == 0:
        test_dl = prepare_dataloader(
            DatasetType.lazy,
            SplitType.test,
            signers,
            signs,
            batch_size,
            transforms=DataAugmentor(0, 0),
        )
        num_signs = len(signs)
        visualize_metrics(best_checkpoint, num_signs, model_size, test_dl)


def training_wrapper(
    rank: int,
    world_size: int,
    signers: list[str],
    signs: range,
    num_epochs: int,
    model_size: ModelSize,
):
    try:
        setup(rank, world_size)
        run_training(rank, world_size, signers, signs, num_epochs, model_size)
    except Exception as e:
        print(f"[Parallel training error]: { e = }")
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    cli_args = train_cli()
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs found!")
        exit(1)
    print(f"Using {world_size} GPUs!")

    signers: list[str] = cli_args.signers
    signs = range(cli_args.selected_signs_from, cli_args.selected_signs_to + 1)
    num_epochs = int(cli_args.num_epochs)
    model_size = get_model_size(cli_args.model_metadata)
    print(
        f"Training signs {cli_args.selected_signs_from} to {cli_args.selected_signs_to}, for {num_epochs} epochs"
    )

    args = (world_size, signers, signs, num_epochs, model_size)
    mp.spawn(training_wrapper, args=args, nprocs=world_size, join=True)
