import gc
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Optional

import torch
from torch import distributed as dist
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from core.constants import DEVICE, TRAIN_CHECKPOINTS_DIR, use_gpu
from data.dataloader import prepare_lazy_dataloaders
from modelling.model import get_model_instance, save_model
from modelling.visualize_model_performance import visualize_metrics


def train(
    model,
    loss,
    optimizer,
    scheduler,
    train_dl,
    val_dl,
    num_epochs,
    device=DEVICE,
    rank=-1,
    train_sampler: Optional[DistributedSampler] = None,
):
    best_val_loss = float("inf")
    best_checkpoint = ""
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    num_words = model.module.num_classes if use_gpu else model.num_classes
    checkpoint_root = (
        f"{TRAIN_CHECKPOINTS_DIR}/checkpoint_{timestamp}-words_{num_words}"
    )

    gc.collect()
    autocast_ctx = nullcontext()
    if rank <= 0:
        os.makedirs(checkpoint_root, exist_ok=True)
    if rank >= 0:
        autocast_ctx = autocast(device_type="cuda", dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    scaler = GradScaler(device=device, enabled=use_gpu)
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training", disable=(rank > 0)):
        model.train()
        train_loss = 0.0

        if train_sampler:
            train_sampler.set_epoch(epoch)

        for kps, labels in tqdm(
            train_dl,
            desc=f"Training Epoch {epoch}",
            total=len(train_dl),
            leave=False,
            disable=(rank >= 0),
        ):
            kps, labels = kps.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast_ctx:
                predicted = model(kps)
                loss_value = loss(predicted, labels)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss_value.item()

        model.eval()
        metrics_tensor = torch.zeros(2).to(device)

        for kps, labels in tqdm(
            val_dl,
            desc=f"Validation Epoch {epoch}",
            total=len(val_dl),
            leave=False,
            disable=(rank >= 0),
        ):
            kps, labels = kps.to(device), labels.to(device)
            with torch.no_grad():
                with autocast_ctx:
                    predicted = model(kps)
                    loss_value = loss(predicted, labels).item()
                    metrics_tensor[0] += loss_value
                    metrics_tensor[1] += 1

        if rank > -1:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)  # ty:ignore[possibly-missing-attribute]

        val_loss = metrics_tensor[0] / metrics_tensor[1]
        train_loss /= len(train_dl)

        if rank <= 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = f"{checkpoint_root}/{epoch}.pth"
                save_model(
                    best_checkpoint,
                    model if rank == -1 else model.module,
                    optimizer,
                    scheduler,
                )

            print(
                f"[Epoch {epoch}/{num_epochs}]: Training Loss: {train_loss:.4f} |  Validation Loss: {val_loss:.4f}"
            )

        scheduler.step(val_loss)

    return best_checkpoint


if __name__ == "__main__":
    num_words = 10
    train_dl, val_dl, test_dl = prepare_lazy_dataloaders(range(1, num_words + 1))

    num_epochs = 1
    lr = 1e-3
    weight_decay = 1e-4
    model = get_model_instance(num_words, DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=3)

    try:
        best_checkpoint = train(
            model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs, DEVICE
        )
    except Exception as e:
        print(f"[training error]: { e = }")

    print(f"Best model checkpoint: {best_checkpoint}")

    visualize_metrics(best_checkpoint, test_dl)
