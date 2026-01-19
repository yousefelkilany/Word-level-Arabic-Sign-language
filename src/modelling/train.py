import gc
import os
from datetime import datetime
from typing import Optional

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from core.constants import DEVICE, TRAIN_CHECKPOINTS_DIR, use_gpu
from data.dataloader import prepare_lazy_dataloaders
from modelling.model import get_model_instance, load_model, save_model


def train(
    model,
    loss,
    optimizer,
    scheduler,
    train_dl,
    val_dl,
    num_epochs,
    device=DEVICE,
    sampler: Optional[DistributedSampler] = None,
):
    best_val_loss = float("inf")
    best_checkpoint = ""
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    num_words = model.module.num_classes if use_gpu else model.num_classes
    checkpoint_root = (
        f"{TRAIN_CHECKPOINTS_DIR}/checkpoint_{timestamp}-words_{num_words}"
    )
    os.makedirs(checkpoint_root, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)

    scaler = GradScaler(device=device, enabled=use_gpu)
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        model.train()
        train_loss = 0.0
        if sampler:
            sampler.set_epoch(epoch)
        for kps, labels in tqdm(
            train_dl,
            desc=f"Training Epoch {epoch}",
            total=len(train_dl),
            leave=False,
        ):
            kps, labels = kps.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device, enabled=use_gpu, dtype=torch.bfloat16):
                predicted = model(kps)
                loss_ = loss(predicted, labels)
            scaler.scale(loss_).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss_.item()

        model.eval()
        val_loss = 0.0
        for kps, labels in tqdm(
            val_dl, desc=f"Eval Epoch {epoch}", total=len(val_dl), leave=False
        ):
            kps, labels = kps.to(device), labels.to(device)
            with torch.no_grad():
                with autocast(
                    device_type=device, enabled=use_gpu, dtype=torch.bfloat16
                ):
                    predicted = model(kps)
                    val_loss += loss(predicted, labels).item()

        val_loss /= len(val_dl)
        train_loss /= len(train_dl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = f"{checkpoint_root}/{epoch}.pth"
            save_model(best_checkpoint, model, optimizer, scheduler)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step(val_loss)
        new_lr = scheduler.get_last_lr()[0]
        if abs(new_lr - last_lr) > 1e-6:
            print(f"new lr = {new_lr}")

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    return best_checkpoint


def visualize_metrics(checkpoint_path, test_dl, device=DEVICE):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    def test_confusion_matrix(model, test_dl, device=DEVICE):
        model.eval()
        test_labels = []
        test_predicted = []
        with torch.no_grad():
            for kps, labels in tqdm(test_dl, desc="Testing"):
                kps, labels = kps.to(device), labels.to(device)
                test_labels.extend(labels)
                outputs = model(kps)
                _, predicted = torch.max(outputs.data, 1)
                test_predicted.extend(predicted)
        return confusion_matrix(test_labels, test_predicted)

    model = load_model(checkpoint_path, device=device)
    conf_mat = test_confusion_matrix(model, test_dl, device=device)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot(cmap=plt.cm.Blues)  # type: ignore
    checkpoint_dir = os.path.dirname(checkpoint_path)
    plt.title("KArSL Confusion Matrix")
    plt.savefig(os.path.join(checkpoint_dir, "KArSL Confusion Matrix.jpg"))


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

    best_checkpoint = train(
        model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs, DEVICE
    )

    print(f"Best model checkpoint: {best_checkpoint}")

    visualize_metrics(best_checkpoint, test_dl)
