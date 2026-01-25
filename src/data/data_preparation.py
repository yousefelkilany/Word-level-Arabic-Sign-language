import random

import cv2
import numpy as np

from core.constants import FACE_SYMMETRY_MAP_PATH, SEQ_LEN, SplitType
from core.mediapipe_utils import KP2SLICE, face_kps_idx
from core.utils import get_default_logger


class TSNSampler:
    def __init__(
        self,
        target_len: int = SEQ_LEN,
        mode: SplitType = SplitType.train,
        jitter_scale: float = 0.8,
    ):
        self.target_len = target_len
        self.mode: SplitType = mode
        self.jitter_scale = jitter_scale
        # self.jitter_scale = np.clip(jitter_scale, 0.2, 0.8)

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        seq_len = kps.shape[0]
        ticks = np.linspace(0, seq_len, self.target_len + 1)
        start_ticks, end_ticks = ticks[:-1], ticks[1:]
        bin_width = end_ticks - start_ticks

        if self.mode == SplitType.train:
            margin = (1.0 - self.jitter_scale) / 2.0

            lower_bound = start_ticks + (bin_width * margin)
            upper_bound = end_ticks - (bin_width * margin)

            indices = lower_bound + np.random.rand(self.target_len) * (
                upper_bound - lower_bound
            )
        else:
            indices = start_ticks + bin_width / 2.0

        indices = np.clip(indices, 0, seq_len - 1 - 1e-6)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.ceil(indices).astype(int)
        alpha = (indices - idx_floor)[:, np.newaxis, np.newaxis]

        return (1 - alpha) * kps[idx_floor] + alpha * kps[idx_ceil]


class DataAugmentor:
    def __init__(self, p_flip=0.5, p_affine=0.5):
        self.p_flip = p_flip
        self.p_affine = p_affine
        self.pose_perm = np.array([1, 0, 3, 2, 5, 4])
        self.face_perm = self._get_face_perm_subset(np.load(FACE_SYMMETRY_MAP_PATH))
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

    def _get_face_perm_subset(self, face_perm) -> np.ndarray:
        face_kps_idx_arr = np.array(face_kps_idx)
        global_to_subset_lookup = np.full(len(face_perm), -1)
        global_to_subset_lookup[face_kps_idx_arr] = np.arange(len(face_kps_idx_arr))
        flipped_global_indices = face_perm[face_kps_idx_arr]
        face_perm_subset = global_to_subset_lookup[flipped_global_indices]

        missing_indices = flipped_global_indices[face_perm_subset == -1]
        if missing_indices.size > 0:
            raise ValueError(
                f"Your face_kps_idx subset is not symmetric. "
                f"Flipped points {missing_indices} are missing from the subset."
            )

        return face_perm_subset

    def _apply_hflip(self, sequence: np.ndarray) -> np.ndarray:
        """
        Expects sequence shape: (Seq_Len, Feats, 4) -> x, y, z, vis
        """
        sequence[..., 0] *= -1

        pose_slice = sequence[:, KP2SLICE["pose"], :]
        pose_len = pose_slice.shape[1]
        assert len(self.pose_perm) == pose_len, (
            f"Pose Permutation map length ({len(self.pose_perm)}) != Pose Slice length ({pose_len})"
        )
        pose_slice = pose_slice[:, self.pose_perm, :]

        face_slice = sequence[:, KP2SLICE["face"], :]
        face_len = face_slice.shape[1]
        assert len(self.face_perm) == face_len, (
            f"Face Permutation map length ({len(self.face_perm)}) != Face Slice length ({face_len})"
        )
        face_slice = face_slice[:, self.face_perm, :]

        lh_slice_cpy = sequence[:, KP2SLICE["lh"], :].copy()
        sequence[:, KP2SLICE["lh"], :] = sequence[:, KP2SLICE["rh"], :]
        sequence[:, KP2SLICE["rh"], :] = lh_slice_cpy

        return sequence

    def _apply_affine(self, kps: np.ndarray) -> np.ndarray:
        """
        Expects sequence shape: (Seq_Len * Feats, 2) -> x, y
        """
        theta = np.radians(np.random.uniform(-15, 15))
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(((c, s), (-s, c)))  # For row-vector multiplication

        scale_factor = np.random.uniform(0.85, 1.15)

        tx = np.random.uniform(-0.1, 0.1)
        ty = np.random.uniform(-0.1, 0.1)

        return np.dot(kps, rotation_matrix) * scale_factor + [[tx, ty]]

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        if not seq.flags.writeable or seq.base is not None:
            # If it's a view/mmap, copy to avoid crashing or corruption
            seq = seq.copy()

        if random.random() < self.p_flip:
            seq = self._apply_hflip(seq.reshape(SEQ_LEN, -1, 4)).reshape(-1, 4)

        if random.random() < self.p_affine:
            seq[:, :2] = self._apply_affine(seq[:, :2])

        return seq.astype(np.float32).reshape(SEQ_LEN, -1)
