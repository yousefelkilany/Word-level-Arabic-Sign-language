from typing import Any

import albumentations as A
import cv2
import numpy as np

from core.constants import FEAT_NUM
from core.mediapipe_utils import KP2SLICE
from core.utils import get_default_logger

POSE_PERM = np.array([1, 0, 3, 2, 5, 4])


class FlipHorizontalKps(A.DualTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)
        self.logger = get_default_logger()

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return super().apply(img, *args, **params)

    def apply_to_keypoints(
        self, keypoints: np.ndarray, *args: Any, **params: Any
    ) -> np.ndarray:
        # kps = np.transpose(keypoints)
        kps = keypoints.copy()
        kps_len = kps.shape[0]
        self.logger.debug(f"{kps.shape = }")

        # flip all x coords
        kps[:, 0] = -kps[:, 0]

        kps_xy = kps[:, :2].reshape(kps_len, FEAT_NUM, 2)
        kps_z = kps[:, 2].reshape(kps_len, FEAT_NUM)

        # swap POSE
        kps_xy[:, KP2SLICE["face"], :] = kps_xy[:, KP2SLICE["face"], :][:, POSE_PERM, :]
        kps_z[:, KP2SLICE["face"]] = kps_z[:, KP2SLICE["face"]][:, POSE_PERM]

        # swap HAND
        rh_xy = kps_xy[:, KP2SLICE["rh"], :].copy()
        self.logger.debug(f"{rh_xy = }")
        lh_xy = kps_xy[:, KP2SLICE["lh"], :].copy()
        self.logger.debug(f"{lh_xy = }")
        kps_xy[:, KP2SLICE["rh"], :] = lh_xy
        self.logger.debug(f"{kps_xy[:, KP2SLICE["rh"], :] = }")
        kps_xy[:, KP2SLICE["lh"], :] = rh_xy
        self.logger.debug(f"{kps_xy[:, KP2SLICE["lh"], :] = }")

        rh_z = kps_z[:, KP2SLICE["rh"]].copy()
        lh_z = kps_z[:, KP2SLICE["lh"]].copy()
        kps_z[:, KP2SLICE["rh"]] = lh_z
        kps_z[:, KP2SLICE["lh"]] = rh_z

        # TODO: swap head keypoints, follow steps in the notes

        return kps_xy.reshape(-1)


class AlbumentationsWrapper:
    def __init__(self):
        super().__init__()

        self.transform = A.Compose(
            [
                # TODO: test flip horizontal augmentation
                # FlipHorizontalKps(p=0.5),
                A.Affine(
                    scale=0.15,
                    translate_percent=0.1,
                    rotate=15,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_LINEAR,
                    p=0.5,
                    border_mode=0,
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xyz",
                label_fields=["keypoint_labels"],  # test with and without
            ),
        )

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        length_min = np.min(sequence)
        length_max = np.max(sequence)
        length_mean = np.mean(sequence)
        print(f"{length_mean = }, {length_min = }, {length_max = }")
        print(f"{(1.5 * length_mean) = }, {np.percentile(sequence, 75) = }")

        length_bracket_width = 1
        custom_bins = np.arange(
            length_min, length_max + length_bracket_width, length_bracket_width
        )
        sign_length_histogram = np.histogram(sequence, bins=custom_bins)[0]
        print(f"{sign_length_histogram = }")

        seq_len = sequence.shape[0]
        flat_seq = sequence.reshape(seq_len, -1)
        dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)

        transformed = self.transform(image=dummy_img, keypoints=flat_seq[:, :2])
        transformed_xy = np.array(transformed["keypoints"], dtype=np.float32)

        return np.column_stack((transformed_xy, flat_seq[:, :2])).reshape(seq_len, -1)
