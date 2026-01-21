import os

import numpy as np
import streamlit as st
import torch

from core.constants import DEVICE, DatasetType, SplitType
from core.utils import extract_num_words_from_checkpoint
from data.dataloader import prepare_dataloaders
from modelling.model import load_model


@st.cache_data
def get_checkpoints_num_words(checkpoint_path):
    return extract_num_words_from_checkpoint(checkpoint_path)


@st.cache_data
def load_cached_checkpoints(checkpoints_dir: str):
    if not os.path.exists(checkpoints_dir):
        return None

    ckpt_files = []
    for root, _, files in os.walk(checkpoints_dir):
        for f in files:
            if f.endswith(".pth"):
                ckpt_files.append(os.path.join(os.path.basename(root), f))
    return ckpt_files


@st.cache_resource
def load_cached_model(checkpoint_path: str, num_words: int):
    model = load_model(checkpoint_path, num_words=num_words, device=DEVICE)
    model.eval()
    return model


@st.cache_resource
def get_cached_dataloaders(num_words: int):
    train_dl, val_dl, test_dl = prepare_dataloaders(
        DatasetType.mmap, signs=range(1, num_words + 1)
    )
    return {"train": train_dl, "val": val_dl, "test": test_dl}


def get_split_dataloader(num_words: int, split_name: SplitType):
    return get_cached_dataloaders(num_words).get(split_name, None)


@st.cache_data
def run_inference(_model, _dataloader, device, checkpoint_path, split_name):
    y_true, y_pred, y_probs = [], [], []
    progress_bar = st.progress(0)
    total = len(_dataloader)

    with torch.no_grad():
        for i, (kps, labels) in enumerate(_dataloader):
            kps, labels = kps.to(device), labels.to(device)
            outputs = _model(kps)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            progress_bar.progress((i + 1) / total)

    progress_bar.empty()
    return np.array(y_true), np.array(y_pred), np.array(y_probs)
