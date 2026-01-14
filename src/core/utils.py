import pandas as pd

from core.constants import LABELS_PATH


def init_words():
    global AR_WORDS, EN_WORDS
    if len(AR_WORDS) == 0 or len(EN_WORDS) == 0:
        words = pd.read_excel(LABELS_PATH, usecols=["Sign-Arabic", "Sign-English"])
        AR_WORDS, EN_WORDS = words.to_dict(orient="list").items()
        AR_WORDS, EN_WORDS = AR_WORDS[1], EN_WORDS[1]


AR_WORDS, EN_WORDS = [], []
init_words()


def extract_num_words_from_checkpoint(checkpoint_path) -> int | None:
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


def get_default_logger():
    global default_logger
    import logging

    if default_logger:
        return default_logger

    default_logger = logging.getLogger("wl-ar-sl")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("logs/server_producer.log")
    file_handler.setFormatter(formatter)
    default_logger.addHandler(file_handler)
    default_logger.setLevel(logging.DEBUG)

    return default_logger
