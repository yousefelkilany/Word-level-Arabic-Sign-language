import os

from torch.cuda import is_available as cuda_is_available

os_join = os.path.join

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOCAL_DATA_DIR = os.path.join(ROOT_DIR, "data")
KAGGLE_DATA_DIR = "/kaggle/input"
use_gpu = os.environ.get("USE_CPU", "0") == "0" and cuda_is_available()
DEVICE = ["cpu", "cuda"][int(use_gpu)]
LOCAL_DEV = int(os.environ.get("LOCAL_DEV", 0))
DATA_DIR = [KAGGLE_DATA_DIR, LOCAL_DATA_DIR][LOCAL_DEV]
KARSL_DATA_DIR = os_join(DATA_DIR, "karsl-502")
LABELS_PATH = os_join(DATA_DIR, "KARSL-502_Labels.xlsx")
LABELS_JSON_PATH = os_join(DATA_DIR, "KARSL-502_Labels.json")
KPS_DIR = os_join(DATA_DIR, "karsl-kps")

INPUT_PREPROCESSED_DIR = (
    "/kaggle/input/word-level-arabic-sign-language-preprcsd-keypoints"
)
MODELS_DIR = os_join(["/kaggle/working", ""][LOCAL_DEV], "models")
MS_30FPS = 1000 / 30
MS_30FPS_INT = 1000 // 30
SEQ_LEN = 50
FEAT_NUM = 184

MAX_WORKERS = 4
