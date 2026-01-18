import os

from torch.cuda import is_available as cuda_is_available

os_join = os.path.join

use_gpu = os.environ.get("USE_CPU", "0") == "0" and cuda_is_available()
DEVICE = ["cpu", "cuda"][int(use_gpu)]
LOCAL_DEV = int(os.environ.get("LOCAL_DEV", 0))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOCAL_INPUT_DATA_DIR = LOCAL_OUTPUT_DATA_DIR = os.path.join(ROOT_DIR, "data")
KAGGLE_INPUT_DATA_DIR = "/kaggle/input"
KAGGLE_OUTPUT_DATA_DIR = "/kaggle/working"
DATA_INPUT_DIR = [KAGGLE_INPUT_DATA_DIR, LOCAL_INPUT_DATA_DIR][LOCAL_DEV]
DATA_OUTPUT_DIR = [KAGGLE_OUTPUT_DATA_DIR, LOCAL_OUTPUT_DATA_DIR][LOCAL_DEV]
NPZ_KPS_DIR = os_join(
    DATA_INPUT_DIR, "word-level-arabic-sign-language-extrcted-keypoints/karsl-kps"
)
LABELS_PATH = os_join(DATA_INPUT_DIR, "KARSL-502_Labels.xlsx")
LABELS_JSON_PATH = os_join(ROOT_DIR, "data", "KARSL-502_Labels.json")
KARSL_DATA_DIR = os_join(DATA_INPUT_DIR, "karsl-502")

INPUT_PREPROCESSED_DIR = os_join(
    DATA_INPUT_DIR, "word-level-arabic-sign-language-preprcsd-keypoints"
)
MODELS_DIR = os_join(DATA_OUTPUT_DIR, "models")
MS_30FPS: float = 1000 / 30
MS_30FPS_INT = 1000 // 30
SEQ_LEN = 50
FEAT_NUM = 184

MAX_WORKERS = 4
