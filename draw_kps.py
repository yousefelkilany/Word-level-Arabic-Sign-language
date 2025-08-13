import os
import cv2
import numpy as np
from itertools import product

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_styles import (
    get_default_pose_landmarks_style,
    get_default_face_mesh_tesselation_style,
    get_default_hand_landmarks_style,
    get_default_hand_connections_style,
)

from utils import (
    DATA_DIR,
    KPS_DIR,
    KP2SLICE,
    pose_kps_idx,
    POSE_KPS_CONNECTIONS,
    face_kps_idx,
    FACE_KPS_CONNECTIONS,
    hand_kps_idx,
    HAND_KPS_CONNECTIONS,
)

landmark_styling_fallback = DrawingSpec(
    color=(224, 224, 224), thickness=1, circle_radius=1
)
connection_styling_fallback = landmark_styling_fallback


def draw_kps_on_image(
    annotated_image,
    kps,
    kps_idx,
    lms_num,
    kp_connections,
    landmark_style=None,
    connection_style=None,
):
    lms_list = [landmark_pb2.NormalizedLandmark() for _ in range(lms_num)]
    for idx, kp in zip(kps_idx, kps):
        lms_list[idx] = landmark_pb2.NormalizedLandmark(x=kp[0], y=kp[1], z=kp[2])

    landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    landmarks_proto.landmark.extend(lms_list)

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        landmarks_proto,
        kp_connections,
        landmark_style if landmark_style else landmark_styling_fallback,
        connection_style if connection_style else connection_styling_fallback,
    )
    return annotated_image


def draw_pose_kps_on_image(rgb_image, pose_kps):
    return draw_kps_on_image(
        rgb_image,
        pose_kps,
        pose_kps_idx,
        len(solutions.pose.PoseLandmark),
        POSE_KPS_CONNECTIONS,
        get_default_pose_landmarks_style(),
    )


def draw_face_kps_on_image(rgb_image, face_kps):
    return draw_kps_on_image(
        rgb_image,
        face_kps,
        face_kps_idx,
        solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES,
        FACE_KPS_CONNECTIONS,
        get_default_face_mesh_tesselation_style(),
    )


def draw_hand_kps_on_image(rgb_image, hand_kps):
    return draw_kps_on_image(
        rgb_image,
        hand_kps,
        hand_kps_idx,
        len(solutions.hands.HandLandmark),
        HAND_KPS_CONNECTIONS,
        get_default_hand_landmarks_style(),
        get_default_hand_connections_style(),
    )


def draw_all_kps_on_image(rgb_image, frame_kps):
    annotated_image = np.copy(rgb_image)
    annotated_image = draw_pose_kps_on_image(
        annotated_image, frame_kps[KP2SLICE["pose"]]
    )
    annotated_image = draw_face_kps_on_image(
        annotated_image, frame_kps[KP2SLICE["face"]]
    )
    annotated_image = draw_hand_kps_on_image(annotated_image, frame_kps[KP2SLICE["rh"]])
    annotated_image = draw_hand_kps_on_image(annotated_image, frame_kps[KP2SLICE["lh"]])
    return annotated_image


if __name__ == "__main__":
    # Drawing keypoints requires the extracted keypoints to be neither centered
    # nor normalized, i.e., run `prepare_kps.py` without --adjusted cli arg
    # For actual training, you may train on unadjusted keypoints, but
    # it's not recommended and you should extract them with --adjusted
    splits = ["train", "test"][-1:]
    signers = ["01", "02", "03"][-1:]
    words = [f"{502:04}"]
    for signer, split, word in product(signers, splits, words):
        kps_path = os.path.join(KPS_DIR, "all_kps", f"{signer}-{split}", f"{word}.npz")

        video_npz = np.load(kps_path, allow_pickle=True)
        video_idx = 0
        video_name, video_kps = list(video_npz.items())[video_idx]
        video_dir = os.path.join(DATA_DIR, signer, signer, split, word, video_name)
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
            f"{signer}-{split}-{word}/{video_name}-frame_{frame_idx}-annotated.jpg"
        )
        os.makedirs(os.path.dirname(annotated_frame_name), exist_ok=True)

        annotated_frame = np.hstack((frame, annotated_frame))
        if cv2.imwrite(annotated_frame_name, annotated_frame):
            print(f"Annotated frame saved to {annotated_frame_name}")
        else:
            print("failed to save Annotated frame")
