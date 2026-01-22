import random

import albumentations as A
import cv2
import numpy as np

from core.constants import FACE_SYMMETRY_MAP_PATH, SEQ_LEN
from core.mediapipe_utils import KP2SLICE
from core.utils import get_default_logger


class DataAugmentor:
    def __init__(self, p_flip=0.5, p_affine=0.5):
        self.p_flip = p_flip
        self.p_affine = p_affine
        self.pose_perm = np.array([1, 0, 3, 2, 5, 4])
        self.face_perm = np.load(FACE_SYMMETRY_MAP_PATH)
        self.logger = get_default_logger()

        self.affine_transform = A.Compose(
            [
                A.Affine(
                    scale=0.15,
                    translate_percent=0.1,
                    rotate=15,
                    interpolation=cv2.INTER_LINEAR,
                    p=self.p_affine,
                    border_mode=0,
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def _apply_hflip(self, sequence: np.ndarray) -> np.ndarray:
        """
        Expects sequence shape: (Seq_Len, Feats, 4) -> x, y, z, vis
        """
        sequence[..., 0] *= -1

        pose_len = sequence[:, KP2SLICE["pose"], :].shape[1]
        assert len(self.pose_perm) == pose_len, (
            f"Pose Permutation map length ({len(self.pose_perm)}) != Pose Slice length ({pose_len})"
        )
        sequence[:, KP2SLICE["pose"], :] = sequence[:, KP2SLICE["pose"], :][
            :, self.pose_perm, :
        ]

        face_len = sequence[:, KP2SLICE["face"], :].shape[1]
        assert len(self.face_perm) == face_len, (
            f"Face Permutation map length ({len(self.face_perm)}) != Face Slice length ({face_len})"
        )
        sequence[:, KP2SLICE["face"], :] = sequence[:, KP2SLICE["face"], :][
            :, self.face_perm, :
        ]

        tmp = sequence[:, KP2SLICE["lh"], :].copy()
        sequence[:, KP2SLICE["lh"], :] = sequence[:, KP2SLICE["rh"], :]
        sequence[:, KP2SLICE["rh"], :] = tmp

        return sequence

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        if not sequence.flags.writeable or sequence.base is not None:
            # If it's a view/mmap, copy to avoid crashing or corruption
            sequence = sequence.copy()

        sequence = sequence.reshape(SEQ_LEN, -1, 4)
        if random.random() < self.p_flip:
            sequence = self._apply_hflip(sequence)
        sequence = sequence.reshape(-1, 4)

        dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
        aug_xy = self.affine_transform(image=dummy_img, keypoints=sequence[..., :2])
        aug_xy = np.array(aug_xy["keypoints"], dtype=np.float32)

        return np.hstack((aug_xy, sequence[..., 2:])).reshape(SEQ_LEN, -1)
