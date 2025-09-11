import os
import pandas as pd
from torch.cuda import is_available as cuda_is_available

os_join = os.path.join

use_gpu = os.environ.get("USE_CPU", "0") == "0" and cuda_is_available()
DEVICE = ["cpu", "cuda"][int(use_gpu)]
LOCAL_DEV = int(os.environ.get("LOCAL_DEV", 0))
DATA_DIR = os_join(["/kaggle/input", "data"][LOCAL_DEV], "karsl-502")
KAGGLE_KPS_DIR = "/kaggle/input"
LABELS_PATH = os_join([KAGGLE_KPS_DIR, "data"][LOCAL_DEV], "KARSL-502_Labels.xlsx")
KPS_DIR = os_join([KAGGLE_KPS_DIR, "data"][LOCAL_DEV], "karsl-kps")
PREPROCESSED_DIR = "/kaggle/working/preprocessed"
INPUT_PREPROCESSED_DIR = (
    "/kaggle/input/word-level-arabic-sign-language-preprcsd-keypoints"
)
MODELS_DIR = os_join(["/kaggle/working", ""][LOCAL_DEV], "models")
MS_30FPS = 1000 / 30
MS_30FPS_INT = 1000 // 30
SEQ_LEN = 60
FEAT_NUM = 184


def init_words():
    global AR_WORDS, EN_WORDS
    if AR_WORDS is None or EN_WORDS is None:
        words = pd.read_excel(LABELS_PATH, usecols=["Sign-Arabic", "Sign-English"])
        AR_WORDS, EN_WORDS = words.to_dict(orient="list").items()


AR_WORDS, EN_WORDS = (None,) * 2
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
