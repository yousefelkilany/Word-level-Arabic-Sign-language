import argparse
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

from core.constants import (
    DEVICE,
    TRAIN_CHECKPOINTS_DIR,
    DatasetType,
    ModelSize,
    get_model_size,
)
from data.dataloader import prepare_dataloaders
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
    gc.collect()
    autocast_ctx = nullcontext()
    if rank <= 0:
        best_val_loss = float("inf")
        best_checkpoint = ""
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        num_signs = model.module.num_classes if rank >= 0 else model.num_classes
        model_size = model.module.model_size if rank >= 0 else model.model_size
        checkpoint_root = f"{TRAIN_CHECKPOINTS_DIR}/checkpoint_{timestamp}-signs_{num_signs}_{model_size.to_str()}"
        os.makedirs(checkpoint_root, exist_ok=True)
    if rank >= 0:
        autocast_ctx = autocast(device_type="cuda", dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    scaler = GradScaler(device=device, enabled=(rank >= 0))
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        if rank >= 0:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)  # ty:ignore[possibly-missing-attribute]

        val_loss = metrics_tensor[0] / metrics_tensor[1]
        train_loss /= len(train_dl)

        if rank <= 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = f"{checkpoint_root}/{epoch}.pth"
                save_model(
                    best_checkpoint,
                    model.module if rank == 0 else model,
                    optimizer,
                    scheduler,
                )

            print(
                f"[Epoch {epoch}/{num_epochs}]: Training Loss: {train_loss:.4f} |  Validation Loss: {val_loss:.4f}"
            )

        scheduler.step(val_loss)

    return best_checkpoint if rank <= 0 else None


def train_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signers", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--selected_signs_from", type=int, default=1)
    parser.add_argument("--selected_signs_to", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--model_metadata", type=str, default=ModelSize.get_default().to_str()
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = train_cli()
    train_dl, val_dl, test_dl = prepare_dataloaders(
        DatasetType.mmap,
        signers=cli_args.signers,
        signs=range(cli_args.selected_signs_from, cli_args.selected_signs_to + 1),
    )

    num_signs = cli_args.selected_signs_to - cli_args.selected_signs_from + 1
    model_size = get_model_size(cli_args.model_metadata)
    lr = 1e-3
    weight_decay = 1e-4
    model = get_model_instance(num_signs, model_size, device=DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=3)

    print(
        f"Training signs {cli_args.selected_signs_from} to {cli_args.selected_signs_to}, for {cli_args.num_epochs} epochs"
    )

    try:
        best_checkpoint = train(
            model,
            loss,
            optimizer,
            scheduler,
            train_dl,
            val_dl,
            cli_args.num_epochs,
            DEVICE,
        )
        print(f"Best model checkpoint: {best_checkpoint}")

    except Exception as e:
        print(f"[training error]: { e = }")

    visualize_metrics(best_checkpoint, num_signs, cli_args.model_metadata, test_dl)
