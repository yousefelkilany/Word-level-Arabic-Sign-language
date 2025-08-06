import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from prepare_kps import (
    pose_kps_idx,
    POSE_KPS_CONNECTIONS,
    POSE_KB2SLICE,
    face_kps_idx,
    FACE_KPS_CONNECTIONS,
    FACE_KB2SLICE,
    hand_kps_idx,
    HAND_KPS_CONNECTIONS,
    RH_KB2SLICE,
    LH_KB2SLICE,
)


def draw_kps_on_image(rgb_image, kps, kps_idx, lms_num, kp_connections):
    annotated_image = np.copy(rgb_image)
    lms_list = [landmark_pb2.NormalizedLandmark(x=0, y=0, z=0) for _ in range(lms_num)]

    landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    for idx, kp in zip(kps_idx, kps):
        lms_list[idx] = landmark_pb2.NormalizedLandmark(x=kp[0], y=kp[1], z=kp[2])
    landmarks_proto.landmark.extend(lms_list)

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        landmarks_proto,
        kp_connections,
        solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    return annotated_image


def draw_pose_kps_on_image(rgb_image, pose_kps):
    lms_num = len(mp.solutions.pose.PoseLandmark)
    return draw_kps_on_image(
        rgb_image, pose_kps, pose_kps_idx, lms_num, POSE_KPS_CONNECTIONS
    )


def draw_face_kps_on_image(rgb_image, face_kps):
    lms_num = mp.solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES
    return draw_kps_on_image(
        rgb_image, face_kps, face_kps_idx, lms_num, FACE_KPS_CONNECTIONS
    )


def draw_hand_kps_on_image(rgb_image, hand_kps):
    lms_num = len(mp.solutions.hands.HandLandmark)
    return draw_kps_on_image(
        rgb_image, hand_kps, hand_kps_idx, lms_num, HAND_KPS_CONNECTIONS
    )


def draw_all_kps_on_image(rgb_image, frame_kps):
    annotated_image = rgb_image
    annotated_image = draw_pose_kps_on_image(annotated_image, frame_kps[POSE_KB2SLICE])
    annotated_image = draw_face_kps_on_image(annotated_image, frame_kps[FACE_KB2SLICE])
    annotated_image = draw_hand_kps_on_image(annotated_image, frame_kps[RH_KB2SLICE])
    annotated_image = draw_hand_kps_on_image(annotated_image, frame_kps[LH_KB2SLICE])
    return annotated_image


DATA_DIR = "/kaggle/input/karsl-502"
KPS_DIR = "/kaggle/working/karsl-kps"

signer, split, word = "03", "test", f"{502:04}"
kps_path = os.path.join(KPS_DIR, "all_kps", f"{signer}-{split}", f"{word}.npz")

video_npz = np.load(kps_path, allow_pickle=True)
video_idx = 0
video_name, video_kps = list(video_npz.items())[video_idx]
video_dir = os.path.join(DATA_DIR, signer, signer, split, word, video_name)
video_frames = sorted(os.listdir(video_dir))

frame_idx = 0
frame = cv2.imread(os.path.join(video_dir, video_frames[frame_idx]))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

annotated_frame = draw_all_kps_on_image(frame, video_kps[frame_idx])
annotated_frame_name = (
    f"{signer}-{split}-{word}-{video_name}-frame_{frame_idx}-annotated.jpg",
)
cv2.imwrite(annotated_frame_name, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

print(f"Annotated frame saved to {annotated_frame_name}")
