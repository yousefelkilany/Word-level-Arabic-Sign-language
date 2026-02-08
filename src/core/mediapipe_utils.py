import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from core.constants import (
    FEAT_DIM,
    FEAT_NUM,
    LANDMARKERS_DIR,
    SIMPLIFIED_FACE_CONNECTIONS_PATH,
    os_join,
    use_gpu,
)
from core.utils import get_default_logger

delegate = [BaseOptions.Delegate.CPU, BaseOptions.Delegate.GPU][int(use_gpu)]

# The idea of points selection is inspired from MuteMotion notebook, I then updated
# magic numbers to corresponding keypoints names from mediapipe class members definitions
# https://www.kaggle.com/code/abd0kamel/mutemotion-wlasl-translation-model?scriptVersionId=154920607&cellId=17

from mediapipe.python.solutions import face_mesh_connections as mp_facemesh
from mediapipe.python.solutions.hands import HandLandmark as mp_hand_landmark
from mediapipe.python.solutions.pose import PoseLandmark as mp_pose_landmark

POSE_KPS_CONNECTIONS = [
    (mp_pose_landmark.LEFT_SHOULDER, mp_pose_landmark.RIGHT_SHOULDER),
    (mp_pose_landmark.LEFT_SHOULDER, mp_pose_landmark.LEFT_ELBOW),
    (mp_pose_landmark.LEFT_ELBOW, mp_pose_landmark.LEFT_WRIST),
    (mp_pose_landmark.RIGHT_SHOULDER, mp_pose_landmark.RIGHT_ELBOW),
    (mp_pose_landmark.RIGHT_ELBOW, mp_pose_landmark.RIGHT_WRIST),
]
FACE_KPS_CONNECTIONS = [
    *mp_facemesh.FACEMESH_CONTOURS,
    *mp_facemesh.FACEMESH_IRISES,
]
from mediapipe.python.solutions.hands import (
    HAND_CONNECTIONS as HAND_KPS_CONNECTIONS,  # noqa: F401
)
from mediapipe.python.solutions.hands_connections import HAND_PALM_CONNECTIONS

pose_kps_mp_idx = (
    mp_pose_landmark.LEFT_SHOULDER,
    mp_pose_landmark.RIGHT_SHOULDER,
    mp_pose_landmark.LEFT_ELBOW,
    mp_pose_landmark.RIGHT_ELBOW,
    mp_pose_landmark.LEFT_WRIST,
    mp_pose_landmark.RIGHT_WRIST,
)
face_kps_mp_idx = tuple(
    sorted(set(point for edge in FACE_KPS_CONNECTIONS for point in edge))
)
hand_kps_mp_idx = tuple(range(len(mp_hand_landmark)))


simplified_face_contours, simplified_face_paths, reduced_face_kps = [], [], []


def init_reduced_face_kps():
    global simplified_face_contours, simplified_face_paths, reduced_face_kps
    if (
        len(simplified_face_contours) > 0
        and len(simplified_face_paths) > 0
        and len(reduced_face_kps) > 0
    ):
        return

    with open(SIMPLIFIED_FACE_CONNECTIONS_PATH, "r") as f:
        simplified_face = json.load(f)
    simplified_face_contours = simplified_face["face_contours"]
    simplified_face_paths = simplified_face["face_paths"]
    reduced_face_kps = tuple(
        sorted(
            set(
                point
                for contour in simplified_face_contours.values()
                for edge in contour
                for point in edge
            )
        )
    )


init_reduced_face_kps()


# reference: https://ai.google.dev/static/edge/mediapipe/images/solutions/hand-landmarks.png
simplified_hand_connections = tuple(
    set([*HAND_PALM_CONNECTIONS, (1, 4), (5, 8), (9, 12), (13, 16), (17, 20)])
)
reduced_hand_kps = tuple(
    sorted(set(point for edge in simplified_hand_connections for point in edge))
)

mp_pose_nose_idx = mp_pose_landmark.NOSE
mp_pose_shoulders_idx = pose_kps_mp_idx[:2]

mp_face_nose_idx = sorted(mp_facemesh.FACEMESH_NOSE)[0][0]
left_iris = mp_facemesh.FACEMESH_LEFT_IRIS
right_iris = mp_facemesh.FACEMESH_RIGHT_IRIS
mp_face_eyes_idx = (list(sorted(left_iris))[0][0], list(sorted(right_iris))[0][0])

mp_hand_wrist_idx = mp_hand_landmark.WRIST
mp_hands_palm_idx = (mp_hand_landmark.THUMB_MCP, mp_hand_landmark.PINKY_MCP)

POSE_NUM = len(pose_kps_mp_idx)
FACE_NUM = len(face_kps_mp_idx)
HAND_NUM = len(hand_kps_mp_idx)

KP2SLICE: dict[str, slice] = {
    "pose": slice(0, POSE_NUM),
    "face": slice(POSE_NUM, POSE_NUM + FACE_NUM),
    "rh": slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM),
    "lh": slice(POSE_NUM + FACE_NUM + HAND_NUM, POSE_NUM + FACE_NUM + HAND_NUM * 2),
}

mp_idx_to_kps_idx = {
    "pose": {mp_idx: kps_idx for kps_idx, mp_idx in enumerate(pose_kps_mp_idx)},
    "face": {
        mp_idx: kps_idx + POSE_NUM for kps_idx, mp_idx in enumerate(face_kps_mp_idx)
    },
    "rh": {
        mp_idx: kps_idx + POSE_NUM + FACE_NUM
        for kps_idx, mp_idx in enumerate(hand_kps_mp_idx)
    },
    "lh": {
        mp_idx: kps_idx + POSE_NUM + FACE_NUM + HAND_NUM
        for kps_idx, mp_idx in enumerate(hand_kps_mp_idx)
    },
}
reduced_mp_kps_idx_view = [
    *[mp_idx_to_kps_idx["pose"][mp_idx] for mp_idx in pose_kps_mp_idx],
    *[mp_idx_to_kps_idx["face"][mp_idx] for mp_idx in reduced_face_kps],
    *[mp_idx_to_kps_idx["rh"][mp_idx] for mp_idx in reduced_hand_kps],
    *[mp_idx_to_kps_idx["lh"][mp_idx] for mp_idx in reduced_hand_kps],
]

rh_offset = POSE_NUM + len(reduced_face_kps)
mp_idx_to_reduced_kps_idx = {
    "pose": {mp_idx: kps_idx for kps_idx, mp_idx in enumerate(pose_kps_mp_idx)},
    "face": {
        mp_idx: kps_idx + POSE_NUM for kps_idx, mp_idx in enumerate(reduced_face_kps)
    },
    "rh": {
        mp_idx: kps_idx + rh_offset for kps_idx, mp_idx in enumerate(reduced_hand_kps)
    },
    "lh": {
        mp_idx: kps_idx + rh_offset + len(reduced_hand_kps)
        for kps_idx, mp_idx in enumerate(reduced_hand_kps)
    },
}

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker

pose_model: PoseLandmarker
face_model: FaceLandmarker
hands_model: HandLandmarker


class LandmarkerProcessor:
    def __init__(self):
        self.video_mode: bool
        self.pose_model: PoseLandmarker
        self.face_model: FaceLandmarker
        self.hands_model: HandLandmarker
        self.landmarkers: list[str]
        self.logger = get_default_logger()

    @classmethod
    def create(
        cls,
        landmarkers: Optional[list[str]] = None,
        video_mode: bool = False,
    ):
        self = cls()
        self.landmarkers = landmarkers or ["pose", "face", "hands"]
        self.video_mode = video_mode
        self.__init_mediapipe_landmarkers()
        return self

    @classmethod
    async def create_async(
        cls, landmarkers: Optional[list[str]] = None, video_mode: bool = False
    ) -> "LandmarkerProcessor":
        return await asyncio.to_thread(cls.create, landmarkers, video_mode)

    def __init_mediapipe_landmarkers(self):
        running_mode = (
            vision.RunningMode.VIDEO if self.video_mode else vision.RunningMode.IMAGE
        )

        if "pose" in self.landmarkers:
            pose_base_options = BaseOptions(
                model_asset_path=os_join(LANDMARKERS_DIR, "pose_landmarker.task"),
                delegate=delegate,
            )
            pose_options = vision.PoseLandmarkerOptions(
                base_options=pose_base_options, running_mode=running_mode
            )
            self.pose_model = vision.PoseLandmarker.create_from_options(pose_options)

        if "face" in self.landmarkers:
            face_base_options = BaseOptions(
                model_asset_path=os_join(LANDMARKERS_DIR, "face_landmarker.task"),
                delegate=delegate,
            )
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options, running_mode=running_mode, num_faces=1
            )
            self.face_model = vision.FaceLandmarker.create_from_options(face_options)

        if "hands" in self.landmarkers:
            hand_base_options = BaseOptions(
                model_asset_path=os_join(LANDMARKERS_DIR, "hand_landmarker.task"),
                delegate=delegate,
            )
            hands_options = vision.HandLandmarkerOptions(
                base_options=hand_base_options, running_mode=running_mode, num_hands=1
            )
            self.hands_model = vision.HandLandmarker.create_from_options(hands_options)

        self.logger.info(f"Models loaded successfully (PID: {os.getpid()})")

    def close(self):
        if "pose" in self.landmarkers and self.pose_model:
            self.pose_model.close()
        if "face" in self.landmarkers and self.face_model:
            self.face_model.close()
        if "hands" in self.landmarkers and self.hands_model:
            self.hands_model.close()

    def extract_frame_keypoints(
        self, frame_rgb, timestamp_ms=-1, adjusted=False, return_both=False
    ):
        # do some preprocessing on frame if needed
        ...

        # define numpy views, pose=6 -> face=136 -> rh=21 -> lh=21
        all_kps_raw = np.zeros((FEAT_NUM, FEAT_DIM))
        pose_kps_raw = all_kps_raw[KP2SLICE["pose"]]
        face_kps_raw = all_kps_raw[KP2SLICE["face"]]
        rh_kps_raw = all_kps_raw[KP2SLICE["rh"]]
        lh_kps_raw = all_kps_raw[KP2SLICE["lh"]]

        all_kps = np.zeros((FEAT_NUM, FEAT_DIM))
        pose_kps = all_kps[KP2SLICE["pose"]]
        face_kps = all_kps[KP2SLICE["face"]]
        rh_kps = all_kps[KP2SLICE["rh"]]
        lh_kps = all_kps[KP2SLICE["lh"]]
        np_xyzv = np.dtype((float, FEAT_DIM))

        def lm_xyzv(lm, v=1.0):
            return (lm.x, lm.y, lm.z, v)

        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        def landmarks_distance(lms_list, lm_idx):
            p1, p2 = lms_list[lm_idx[0]], lms_list[lm_idx[1]]
            return max(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), 1e-6)

        def get_pose():
            nonlocal pose_kps, pose_kps_raw
            if self.video_mode:
                results = self.pose_model.detect_for_video(frame, timestamp_ms)
            else:
                results = self.pose_model.detect(frame)

            if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
                return

            lms = results.pose_landmarks[0]
            pose_kps[:] = np.fromiter(
                (lm_xyzv(lms[idx], lms[idx].visibility) for idx in pose_kps_mp_idx),
                dtype=np_xyzv,
            )
            if adjusted:
                if return_both:
                    pose_kps_raw[:] = pose_kps[:]
                pose_kps[:, :3] -= pose_kps[mp_pose_nose_idx, :3]
                pose_kps[:, :3] /= landmarks_distance(lms, mp_pose_shoulders_idx)

        def get_face():
            nonlocal face_kps, face_kps_raw
            if self.video_mode:
                results = self.face_model.detect_for_video(frame, timestamp_ms)
            else:
                results = self.face_model.detect(frame)
            if results.face_landmarks is None or len(results.face_landmarks) == 0:
                return

            lms = results.face_landmarks[0]
            face_kps[:] = np.fromiter(
                (lm_xyzv(lms[idx]) for idx in face_kps_mp_idx), dtype=np_xyzv
            )
            if adjusted:
                if return_both:
                    face_kps_raw[:] = face_kps[:]
                face_kps[:, :3] -= face_kps[mp_face_nose_idx, :3]
                face_kps[:, :3] /= landmarks_distance(lms, mp_face_eyes_idx)

        def get_hands():
            nonlocal rh_kps, lh_kps, rh_kps_raw, lh_kps_raw
            if self.video_mode:
                results = self.hands_model.detect_for_video(frame, timestamp_ms)
            else:
                results = self.hands_model.detect(frame)
            if results.hand_landmarks is None or len(results.hand_landmarks) == 0:
                return

            for handedness, hand_lms in zip(results.handedness, results.hand_landmarks):
                is_left = handedness[0].category_name == "Left"
                target_hand = lh_kps if is_left else rh_kps
                target_hand[:] = np.fromiter(
                    (lm_xyzv(hand_lms[idx]) for idx in hand_kps_mp_idx),
                    dtype=np_xyzv,
                )
                if adjusted:
                    if return_both:
                        target_hand_raw = lh_kps_raw if is_left else rh_kps_raw
                        target_hand_raw[:] = target_hand[:]
                    target_hand[:, :3] -= target_hand[mp_hand_wrist_idx, :3]
                    target_hand[:, :3] /= landmarks_distance(
                        hand_lms, mp_hands_palm_idx
                    )

        with ThreadPoolExecutor(max_workers=3) as executor:
            if "pose" in self.landmarkers and self.pose_model:
                pose_res = executor.submit(get_pose)
                pose_res.result()

            if "face" in self.landmarkers and self.face_model:
                face_res = executor.submit(get_face)
                face_res.result()

            if "hands" in self.landmarkers and self.hands_model:
                hand_res = executor.submit(get_hands)
                hand_res.result()

        # do some preprocessing on kps if needed
        ...

        if return_both:
            return all_kps, all_kps_raw

        return all_kps
