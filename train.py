import os
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import prepare_dataloaders
from model import get_model_instance, save_model


def train(model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs):
    best_val_loss = float("inf")
    best_checkpoint = ""
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    model_checkpoint_root = f"checkpoints/checkpoint_{timestamp}"
    os.makedirs(model_checkpoint_root)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for kps, labels in tqdm(
            train_dl,
            desc=f"Training Epoch {epoch + 1}",
            total=len(train_dl),
            leave=False,
        ):
            optimizer.zero_grad()
            predicted = model(kps)
            loss_ = loss(predicted, labels)
            loss_.backward()
            optimizer.step()
            train_loss += loss_.item()

        model.eval()
        val_loss = 0.0
        for kps, labels in tqdm(
            val_dl, desc=f"Eval Epoch {epoch + 1}", total=len(val_dl), leave=False
        ):
            predicted = model(kps)
            loss_ = loss(predicted, labels)
            val_loss += loss_.item()

        val_loss /= len(val_dl)
        train_loss /= len(train_dl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_checkpoint = f"{model_checkpoint_root}/{epoch}.pth"
            save_model(model_checkpoint, model, optimizer, scheduler)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step(val_loss)
        new_lr = scheduler.get_last_lr()[0]
        if abs(new_lr - last_lr) > 1e-6:
            print(f"new lr = {new_lr}")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    return best_checkpoint


def test_confusion_matrix(model, test_dl):
    model.eval()
    test_labels = []
    test_predicted = []
    with torch.no_grad():
        for kps, labels in tqdm(test_dl, desc="Testing"):
            test_labels.extend(labels)
            outputs = model(kps)
            _, predicted = torch.max(outputs.data, 1)
            test_predicted.extend(predicted)
    return confusion_matrix(test_labels, test_predicted)


if __name__ == "__main__":
    signers = ["01", "02", "03"]
    num_words = 10
    train_dl, val_dl, test_dl = prepare_dataloaders(signers, range(1, num_words + 1))

    num_epochs = 1
    lr = 1e-3
    weight_decay = 1e-4
    model = get_model_instance(num_words)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=5)

    best_checkpoint = train(
        model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs
    )
    model = get_model_instance(num_words)
    model.load_state_dict(torch.load(best_checkpoint)["model"])
    conf_mat = test_confusion_matrix(model, test_dl)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("KArSL Confusion Matrix")
    plt.savefig("KArSL Confusion Matrix.jpg")
