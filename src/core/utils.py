import os
import json
import logging
from typing import Optional

from core.constants import (
    LABELS_JSON_PATH,
    LOGS_DIR,
    os_join,
    ModelSize,
    get_model_size,
)


AR_WORDS, EN_WORDS = [], []


def init_signs():
    global AR_WORDS, EN_WORDS
    if len(AR_WORDS) > 0 and len(EN_WORDS) > 0:
        return

    with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        signs = json.load(f)
    AR_WORDS, EN_WORDS = signs["AR_WORDS"], signs["EN_WORDS"]


init_signs()


def extract_metadata_from_checkpoint(
    checkpoint_path,
) -> Optional[tuple[int, ModelSize]]:
    import re

    matches = re.search(r".*?signs_(\d+)_(\w_\d_\d).*?", checkpoint_path)
    if not matches or len(matches.groups()) != 2:
        raise ValueError(
            f"Couldn't find number of signs or model size in checkpoint path: {checkpoint_path}"
        )

    num_signs = int(matches.groups()[0])
    model_metadata = get_model_size(matches.groups()[1])
    print(f"Number of signs in checkpoint: {num_signs}")
    print(f"Model size in checkpoint: {model_metadata}")
    return num_signs, model_metadata


default_logger = None


def get_default_logger() -> logging.Logger:
    global default_logger

    if default_logger:
        return default_logger

    logging_level = logging.DEBUG

    default_logger = logging.getLogger("wl-ar-sl")
    default_logger.setLevel(logging_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    file_handler = logging.FileHandler(os_join(LOGS_DIR, "server_producer.log"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging_level)
    default_logger.addHandler(file_handler)

    return default_logger


def is_git_lfs_pointer(filepath):
    """
    Checks if a file is a Git LFS pointer file based on its content signature.
    """
    LFS_POINTER_PREFIX = b"version https://git-lfs.github.com"

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    try:
        with open(filepath, "rb") as f:
            initial_bytes = f.read(50)
        return initial_bytes.startswith(LFS_POINTER_PREFIX)
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return False
