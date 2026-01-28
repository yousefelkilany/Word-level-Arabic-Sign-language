import os
import json
import logging
from typing import Optional

from core.constants import LABELS_JSON_PATH, LOGS_DIR, os_join


def init_signs() -> tuple[list[str], list[str]]:
    with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        signs = json.load(f)
    return signs["AR_WORDS"], signs["EN_WORDS"]


AR_WORDS, EN_WORDS = init_signs()


def extract_num_signs_from_checkpoint(checkpoint_path) -> Optional[int]:
    import re

    matches = re.search(r".*?signs_(\d+).*?", checkpoint_path)
    if not matches:
        raise ValueError(
            f"Couldn't find number of signs in checkpoint path: {checkpoint_path}"
        )

    num_signs = int(matches.groups()[0])
    print(f"Number of signs in checkpoint: {num_signs}")
    return num_signs


default_logger = None


def get_default_logger() -> logging.Logger:
    global default_logger

    if default_logger:
        return default_logger

    default_logger = logging.getLogger("wl-ar-sl")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    file_handler = logging.FileHandler(os_join(LOGS_DIR, "server_producer.log"))
    file_handler.setFormatter(formatter)
    default_logger.addHandler(file_handler)
    default_logger.setLevel(logging.DEBUG)

    return default_logger
