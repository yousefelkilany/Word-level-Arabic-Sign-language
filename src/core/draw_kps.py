import os
from itertools import product

import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import face_mesh, hands
from mediapipe.python.solutions.drawing_styles import (
    get_default_face_mesh_tesselation_style,
    get_default_hand_connections_style,
    get_default_hand_landmarks_style,
    get_default_pose_landmarks_style,
)
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks

from core.constants import KARSL_DATA_DIR, NPZ_KPS_DIR
from core.mediapipe_utils import (
    FACE_KPS_CONNECTIONS,
    HAND_KPS_CONNECTIONS,
    KP2SLICE,
    POSE_KPS_CONNECTIONS,
    face_kps_mp_idx,
    hand_kps_mp_idx,
    mp_pose_landmark,
    pose_kps_mp_idx,
)
from core.utils import get_default_logger

landmark_styling_fallback = DrawingSpec(
    color=(224, 224, 224), thickness=1, circle_radius=1
)
connection_styling_fallback = landmark_styling_fallback
logger = get_default_logger()


def get_lms_list(kps, kps_idx, lms_num, return_as_lm=True):
    if return_as_lm:
        lms_list = [landmark_pb2.NormalizedLandmark() for _ in range(lms_num)]  # type: ignore
        for idx, kp in zip(kps_idx, kps):
            lms_list[idx] = landmark_pb2.NormalizedLandmark(x=kp[0], y=kp[1], z=kp[2])  # type: ignore
        return lms_list

    lms_list = [(0, 0, 0)] * lms_num
    for idx, kp in zip(kps_idx, kps):
        lms_list[idx] = kp[:3]
    return np.array(lms_list)


def draw_kps_on_image(
    annotated_image,
    lms_list,
    kp_connections,
    landmark_style=None,
    connection_style=None,
):
    landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
    landmarks_proto.landmark.extend(lms_list)

    draw_landmarks(
        annotated_image,
        landmarks_proto,
        kp_connections,
        landmark_style if landmark_style else landmark_styling_fallback,
        connection_style if connection_style else connection_styling_fallback,
    )
    return annotated_image


def get_pose_lms_list(frame_kps, return_as_lm=True):
    return get_lms_list(
        frame_kps[KP2SLICE["pose"]],
        pose_kps_mp_idx,
        len(mp_pose_landmark),
        return_as_lm,
    )


def draw_pose_kps_on_image(
    rgb_image, frame_kps, landmark_style=None, connection_style=None
):
    return draw_kps_on_image(
        rgb_image,
        get_pose_lms_list(frame_kps),
        POSE_KPS_CONNECTIONS,
        landmark_style or get_default_pose_landmarks_style(),
        connection_style,
    )


def get_face_lms_list(frame_kps, return_as_lm=True):
    return get_lms_list(
        frame_kps[KP2SLICE["face"]],
        face_kps_mp_idx,
        face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES,
        return_as_lm,
    )


def draw_face_kps_on_image(
    rgb_image, frame_kps, landmark_style=None, connection_style=None
):
    return draw_kps_on_image(
        rgb_image,
        get_face_lms_list(frame_kps),
        FACE_KPS_CONNECTIONS,
        landmark_style or get_default_face_mesh_tesselation_style(),
        connection_style or get_default_face_mesh_tesselation_style(),
    )


def get_hand_lms_list(frame_kps, handedness, return_as_lm=True):
    return get_lms_list(
        frame_kps[KP2SLICE[handedness]],
        hand_kps_mp_idx,
        len(hands.HandLandmark),
        return_as_lm,
    )


def draw_hand_kps_on_image(
    rgb_image, frame_kps, handedness, landmark_style=None, connection_style=None
):
    return draw_kps_on_image(
        rgb_image,
        get_hand_lms_list(frame_kps, handedness),
        HAND_KPS_CONNECTIONS,
        landmark_style or get_default_hand_landmarks_style(),
        connection_style or get_default_hand_connections_style(),
    )


def draw_all_kps_on_image(
    rgb_image,
    frame_kps,
    return_separate_images=False,
    pose_landmark_style=None,
    pose_connection_style=None,
    face_landmark_style=None,
    face_connection_style=None,
    hand_landmark_style=None,
    hand_connection_style=None,
):
    pose_image, face_image, rh_image, lh_image = (None,) * 4
    annotated_image = np.copy(rgb_image)
    if return_separate_images:
        pose_image = draw_pose_kps_on_image(
            annotated_image,
            frame_kps,
            pose_landmark_style,
            pose_connection_style,
        )
        face_image = draw_face_kps_on_image(
            annotated_image,
            frame_kps,
            face_landmark_style,
            face_connection_style,
        )
        rh_image = draw_hand_kps_on_image(
            annotated_image,
            frame_kps,
            "rh",
            hand_landmark_style,
            hand_connection_style,
        )
        lh_image = draw_hand_kps_on_image(
            annotated_image,
            frame_kps,
            "lh",
            hand_landmark_style,
            hand_connection_style,
        )

    annotated_image = draw_pose_kps_on_image(
        annotated_image,
        frame_kps,
        pose_landmark_style,
        pose_connection_style,
    )
    annotated_image = draw_face_kps_on_image(
        annotated_image,
        frame_kps,
        face_landmark_style,
        face_connection_style,
    )
    annotated_image = draw_hand_kps_on_image(
        annotated_image,
        frame_kps,
        "rh",
        hand_landmark_style,
        hand_connection_style,
    )
    annotated_image = draw_hand_kps_on_image(
        annotated_image,
        frame_kps,
        "lh",
        hand_landmark_style,
        hand_connection_style,
    )
    return annotated_image, pose_image, face_image, rh_image, lh_image


if __name__ == "__main__":
    # Drawing keypoints requires the extracted keypoints to be neither centered
    # nor normalized, i.e., run `prepare_kps.py` without --adjusted cli arg
    # For actual training, you may train on unadjusted keypoints, but
    # it's not recommended and you should extract them with --adjusted
    splits = ["train", "test"][-1:]
    signers = ["01", "02", "03"][-1:]
    signs = [f"{502:04}"]
    for signer, split, sign in product(signers, splits, signs):
        kps_path = os.path.join(
            NPZ_KPS_DIR, "all_kps", f"{signer}-{split}", f"{sign}.npz"
        )

        video_npz = np.load(kps_path, allow_pickle=True)
        video_idx = 0
        video_name, video_kps = list(video_npz.items())[video_idx]
        video_dir = os.path.join(
            KARSL_DATA_DIR, signer, signer, split, sign, video_name
        )
        video_frames = sorted(os.listdir(video_dir))

        assert video_kps.min() != 0 and video_kps.max() != 0, (
            "Corrupted Keypoints, all keypoints are zeros"
        )

        frame_idx = 0
        frame_path = os.path.join(video_dir, video_frames[frame_idx])
        frame = cv2.imread(frame_path)

        print(f"Annotating frame {frame_path}")
        annotated_frame = draw_all_kps_on_image(frame, video_kps[frame_idx])
        annotated_frame_name = (
            f"{signer}-{split}-{sign}/{video_name}-frame_{frame_idx}-annotated.jpg"
        )
        os.makedirs(os.path.dirname(annotated_frame_name), exist_ok=True)

        annotated_frame = np.hstack((frame, annotated_frame))
        if cv2.imwrite(annotated_frame_name, annotated_frame):
            print(f"Annotated frame saved to {annotated_frame_name}")
        else:
            print("failed to save Annotated frame")
