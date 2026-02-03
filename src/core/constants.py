import os
from enum import StrEnum, auto
from os.path import join as os_join
from typing import TYPE_CHECKING, Optional

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


class HeadSize(StrEnum):
    tiny = "t"
    small = "s"
    medium = "m"
    large = "l"

    @classmethod
    def from_str(cls, head_size_letter: str) -> "HeadSize":
        return {h.value: h for h in HeadSize}[head_size_letter]

    @classmethod
    def get_val(cls, head_size: "HeadSize") -> int:
        match head_size:
            case HeadSize.tiny:
                return 16
            case HeadSize.small:
                return 32
            case HeadSize.medium:
                return 64
            case HeadSize.large:
                return 128
            case _:
                raise ValueError(f"Unknown head size: {head_size}")


class ModelSize:
    def __init__(
        self,
        head_size: HeadSize,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        match head_size:
            case HeadSize.tiny:
                num_heads = 4
                num_layers = 2
            case HeadSize.small:
                num_heads = 4
                num_layers = 4
            case HeadSize.medium:
                num_heads = 4
                num_layers = 6
            case HeadSize.large:
                num_heads = 6
                num_layers = 8
            case _:
                raise ValueError(f"Unknown head size: {head_size}")

        self.num_heads = self.num_heads or num_heads
        self.num_layers = self.num_layers or num_layers

    def __str__(self):
        return f"ModelSize({self.head_size}={HeadSize.get_val(self.head_size)}, {self.num_heads}, {self.num_layers})"

    @property
    def params(self):
        return (
            HeadSize.get_val(self.head_size),
            self.num_heads,
            self.num_layers,
        )

    def to_str(self) -> str:
        return f"{self.head_size.value}_{self.num_heads}_{self.num_layers}"

    @classmethod
    def from_str(cls, model_metadata: str) -> "ModelSize":
        if not model_metadata or len(model_metadata) != 5:
            raise ValueError(f"Invalid model metadata: {model_metadata}")

        return cls(
            head_size=HeadSize.from_str(model_metadata[0]),
            num_heads=int(model_metadata[2]),
            num_layers=int(model_metadata[4]),
        )

    @classmethod
    def get_default(cls) -> "ModelSize":
        return cls(head_size=HeadSize.small, num_heads=4, num_layers=2)


def get_model_size(model_metadata: str) -> ModelSize:
    return ModelSize.from_str(model_metadata)


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
