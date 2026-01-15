import os

from torch.cuda import is_available as cuda_is_available

os_join = os.path.join

use_gpu = os.environ.get("USE_CPU", "0") == "0" and cuda_is_available()
DEVICE = ["cpu", "cuda"][int(use_gpu)]
LOCAL_DEV = int(os.environ.get("LOCAL_DEV", 0))
DATA_DIR = os_join(["/kaggle/input", "data"][LOCAL_DEV], "karsl-502")
KAGGLE_KPS_DIR = "/kaggle/input"
LABELS_PATH = os_join([KAGGLE_KPS_DIR, "data"][LOCAL_DEV], "KARSL-502_Labels.xlsx")
KPS_DIR = os_join([KAGGLE_KPS_DIR, "data"][LOCAL_DEV], "karsl-kps")
INPUT_PREPROCESSED_DIR = (
    "/kaggle/input/word-level-arabic-sign-language-preprcsd-keypoints"
)
MODELS_DIR = os_join(["/kaggle/working", ""][LOCAL_DEV], "models")
MS_30FPS = 1000 / 30
MS_30FPS_INT = 1000 // 30
SEQ_LEN = 60
FEAT_NUM = 184

MAX_WORKERS = 4
