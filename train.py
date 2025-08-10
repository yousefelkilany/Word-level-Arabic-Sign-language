import datetime
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from optim.lr_scheduler import ReduceLROnPlateau

from dataloader import FEAT_NUM, KArSLDataset
from model import AttentionBiLSTM


def train(model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs):
    # training loop
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
            outputs = model(kps)
            loss_ = loss(outputs, labels)
            loss_.backward()
            optimizer.step()
            train_loss += loss_.item()

        model.eval()
        val_loss = 0.0
        for kps, labels in tqdm(
            val_dl, desc=f"Eval Epoch {epoch + 1}", total=len(val_dl), leave=False
        ):
            outputs = model(kps)
            loss_ = loss(outputs, labels)
            val_loss += loss_.item()

        val_loss /= len(val_dl)
        train_loss /= len(train_dl)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dl)}")


def test(model, test_dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for kps, labels in test_dl:
            outputs = model(kps)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {correct / total}")


def prepare_dataloaders(signers, selected_words):
    train_val = KArSLDataset("train", signers, selected_words)
    train_size = int(len(train_val) * 0.8)
    val_size = len(train_val) - train_size
    train_ds, val_ds = random_split(train_val, [train_size, val_size])

    batch_size = 16
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    test_ds = KArSLDataset("test", signers, selected_words)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, val_dl, test_dl, train_val.num_classes


def save_model(model):
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")  # Dec17_14-22-35
    model_checkpoint = f"checkpoint_{timestamp}.pth"
    torch.save(
        {
            "model": trained_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        model_checkpoint,
    )

    return model_checkpoint


if __name__ == "__main__":
    signers = ["01", "02", "03"]
    num_words = 10
    train_dl, val_dl, test_dl, num_classes = prepare_dataloaders(
        signers, range(1, num_words + 1)
    )

    input_size = FEAT_NUM * 3
    hidden_size = 512
    num_lstm_blocks = 2
    num_classes = num_classes
    model = AttentionBiLSTM(input_size, hidden_size, num_lstm_blocks, num_classes)
    loss = nn.CrossEntropyLoss()

    num_epochs = 10
    weight_decay = 1e-4
    lr = 1e-3
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.2, patience=5, verbose=True
    )

    trained_model = train(
        model, loss, optimizer, scheduler, train_dl, val_dl, num_epochs
    )
    model_checkpoint = save_model(trained_model)
    model = torch.load(model_checkpoint)["model"]
    test(model, test_dl)
