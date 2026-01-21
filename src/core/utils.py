import json
import logging
from typing import Optional

from core.constants import LABELS_JSON_PATH


def init_words() -> tuple[list[str], list[str]]:
    with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        signs = json.load(f)
    return signs["AR_WORDS"], signs["EN_WORDS"]


AR_WORDS, EN_WORDS = init_words()


def extract_num_words_from_checkpoint(checkpoint_path) -> Optional[int]:
    import re

    matches = re.search(r".*?words_(\d+).*?", checkpoint_path)
    if not matches:
        raise ValueError(
            f"Couldn't find number of words in checkpoint path: {checkpoint_path}"
        )

    num_words = int(matches.groups()[0])
    print(f"Number of words in checkpoint: {num_words}")
    return num_words


default_logger = None


def get_default_logger() -> logging.Logger:
    global default_logger

    if default_logger:
        return default_logger

    default_logger = logging.getLogger("wl-ar-sl")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("logs/server_producer.log")
    file_handler.setFormatter(formatter)
    default_logger.addHandler(file_handler)
    default_logger.setLevel(logging.DEBUG)

    return default_logger
