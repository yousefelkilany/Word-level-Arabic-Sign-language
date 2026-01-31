import os
from enum import StrEnum, auto
from os.path import join as os_join
from typing import TYPE_CHECKING

from torch.cuda import is_available as cuda_is_available

use_gpu = os.environ.get("USE_CPU", "0") == "0" and cuda_is_available()
DEVICE = ["cpu", "cuda"][int(use_gpu)]
LOCAL_DEV = int(os.environ.get("LOCAL_DEV", 1))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = os_join(PROJECT_ROOT_DIR, "logs")
LANDMARKERS_DIR = os_join(PROJECT_ROOT_DIR, "landmarkers")
MODELS_DIR = os_join(PROJECT_ROOT_DIR, "models")
PROJECT_DATA_DIR = os_join(PROJECT_ROOT_DIR, "data")
LABELS_PATH = os_join(PROJECT_DATA_DIR, "KARSL-502_Labels.xlsx")
LABELS_JSON_PATH = os_join(PROJECT_DATA_DIR, "KARSL-502_Labels.json")
FACE_SYMMETRY_MAP_PATH = os_join(PROJECT_DATA_DIR, "face_mesh_symmetry_map.npy")
LOCAL_INPUT_DATA_DIR = LOCAL_OUTPUT_DATA_DIR = PROJECT_DATA_DIR
KAGGLE_INPUT_DATA_DIR = "/kaggle/input"
KAGGLE_OUTPUT_DATA_DIR = "/kaggle/working"
DATA_INPUT_DIR = [KAGGLE_INPUT_DATA_DIR, LOCAL_INPUT_DATA_DIR][LOCAL_DEV]
DATA_OUTPUT_DIR = [KAGGLE_OUTPUT_DATA_DIR, LOCAL_OUTPUT_DATA_DIR][LOCAL_DEV]
KARSL_DATA_DIR = os_join(DATA_INPUT_DIR, "karsl-502")
NPZ_KPS_DIR = os_join(
    DATA_INPUT_DIR, "word-level-arabic-sign-language-extrcted-keypoints", "karsl-kps"
)
MMAP_PREPROCESSED_DIR = os_join(
    DATA_INPUT_DIR,
    "word-level-arabic-sign-language-preprcsd-keypoints",
    "word-level-arabic-sign-language-preprcsd-keypoints",
)
MMAP_OUTPUT_PREPROCESSED_DIR = os_join(
    DATA_OUTPUT_DIR, "word-level-arabic-sign-language-preprcsd-keypoints"
)
TRAIN_CHECKPOINTS_DIR = os_join(DATA_OUTPUT_DIR, "checkpoints")

MS_30FPS = 1000 / 30
MS_30FPS_INT = 1000 // 30
SEQ_LEN = 50
FEAT_NUM = 184
FEAT_DIM = 4  # x, y, z, v (visibility)

MAX_WORKERS = 4


class SplitType(StrEnum):
    train = auto()
    val = auto()
    test = auto()


class DatasetType(StrEnum):
    lazy = auto()
    mmap = auto()


if TYPE_CHECKING:
    from data import LazyKArSLDataset, MmapKArSLDataset

type KarslDatasetType = "LazyKArSLDataset" | "MmapKArSLDataset"
