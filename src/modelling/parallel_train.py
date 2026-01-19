from data.dataloader import prepare_lazy_dataloader
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data.lazy_dataset import LazyKArSLDataset
from modelling.model import get_model_instance
from modelling.train import train, visualize_metrics


def setup(rank, world_size):
    """Sets up the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Standard port
    # Initialize process group using NCCL backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_training(rank, world_size):
    setup(rank, world_size)

    # 1. Device Setup
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"Using {torch.cuda.device_count()} GPUs!")

    num_words = 10
    signers = ["01", "02", "03"]
    selected_words = range(1, num_words + 1)
    batch_size = 64

    model = get_model_instance(num_words, device=device)
    model = DDP(model, device_ids=[rank])

    # 3. Dataset & Sampler
    # Use DistributedSampler to partition data across GPUs
    dataset = LazyKArSLDataset(
        split="train",
        signers=signers,
        selected_words=selected_words,
        train_transforms=None,  # AlbumentationsWrapper(),
    )
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)

    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler)

    # 4. Training Loop
    num_epochs = 1
    lr = 1e-3
    weight_decay = 1e-4
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=3)

    best_checkpoint = train(
        model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs, str(device)
    )

    print(f"Best model checkpoint: {best_checkpoint}")

    if rank == 0:
        test_dl = prepare_lazy_dataloader("test", selected_words, signers, batch_size)
        visualize_metrics(best_checkpoint, test_dl)

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # for epoch in range(2):
    #     sampler.set_epoch(epoch)  # Necessary for shuffling
    #     for data in dataloader:
    #         inputs = data.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, inputs)
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Rank {rank} finished epoch {epoch}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of T4 GPUs
    # Use spawn to launch processes for each GPU
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)
