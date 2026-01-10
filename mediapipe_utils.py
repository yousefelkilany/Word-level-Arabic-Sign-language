import os

from mediapipe.tasks.python import BaseOptions, vision

delegate = BaseOptions.Delegate.CPU
pose_base_options = BaseOptions(
    model_asset_path="pose_landmarker.task", delegate=delegate
)
face_base_options = BaseOptions(
    model_asset_path="face_landmarker.task", delegate=delegate
)
hand_base_options = BaseOptions(
    model_asset_path="hand_landmarker.task", delegate=delegate
)

# after different experiments with VIDEO mode, it's found that IMAGE consistently doesn't miss any keypoints, or at least at acceptable level.
# exp 1: I tried VIDEO mode and initialize a new model for each video (inside `process_video_wrapper` and pass to `extract_keypoints_from_frames`), which is already very slow.
# exp 2: I tried VIDEO mode with global timestamp so it doesn't require re-initialization, but misses faces and hands alot, since it loses track of the older ones but doesn't re-detect them.
running_mode = vision.RunningMode.IMAGE

# The idea of points selection is inspired from MuteMotion notebook, I then updated
# magic numbers to corresponding keypoints names from mediapipe class members definitions
# https://www.kaggle.com/code/abd0kamel/mutemotion-wlasl-translation-model?scriptVersionId=154920607&cellId=17

from mediapipe.python.solutions.pose import PoseLandmark as mp_pose_landmark
from mediapipe.python.solutions import face_mesh_connections as mp_facemesh
from mediapipe.python.solutions.hands import HandLandmark as mp_hand_landmark

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
from mediapipe.python.solutions.hands import HAND_CONNECTIONS as HAND_KPS_CONNECTIONS

pose_kps_idx = tuple(
    (
        mp_pose_landmark.LEFT_SHOULDER,
        mp_pose_landmark.RIGHT_SHOULDER,
        mp_pose_landmark.LEFT_ELBOW,
        mp_pose_landmark.RIGHT_ELBOW,
        mp_pose_landmark.LEFT_WRIST,
        mp_pose_landmark.RIGHT_WRIST,
    )
)
face_kps_idx = tuple(
    sorted(
        set(
            point
            for edge in [*mp_facemesh.FACEMESH_CONTOURS, *mp_facemesh.FACEMESH_IRISES]
            for point in edge
        )
    )
)
hand_kps_idx = tuple(range(len(mp_hand_landmark)))

mp_pose_nose_idx = mp_pose_landmark.NOSE
mp_pose_shoulders_idx = pose_kps_idx[:2]

mp_face_nose_idx = sorted(mp_facemesh.FACEMESH_NOSE)[0][0]
left_iris = mp_facemesh.FACEMESH_LEFT_IRIS
right_iris = mp_facemesh.FACEMESH_RIGHT_IRIS
mp_face_eyes_idx = (list(sorted(left_iris))[0][0], list(sorted(right_iris))[0][0])

mp_hand_wrist_idx = mp_hand_landmark.WRIST
mp_hands_palm_idx = (mp_hand_landmark.THUMB_MCP, mp_hand_landmark.PINKY_MCP)

POSE_NUM = len(pose_kps_idx)
FACE_NUM = len(face_kps_idx)
HAND_NUM = len(hand_kps_idx)

POSE_KP2SLICE = (slice(0, POSE_NUM),)
FACE_KP2SLICE = (slice(POSE_NUM, POSE_NUM + FACE_NUM),)
RH_KP2SLICE = (slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM),)
LH_KP2SLICE = (
    slice(POSE_NUM + FACE_NUM + HAND_NUM, POSE_NUM + FACE_NUM + HAND_NUM * 2),
)
KP2SLICE = {
    "pose": POSE_KP2SLICE,
    "face": FACE_KP2SLICE,
    "rh": RH_KP2SLICE,
    "lh": LH_KP2SLICE,
}


def init_mediapipe_worker(inference_mode=False):
    global pose_model, face_model, hands_model

    running_mode = (
        vision.RunningMode.VIDEO if inference_mode else vision.RunningMode.IMAGE
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options, running_mode=running_mode
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options, running_mode=running_mode, num_faces=1
    )
    hands_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options, running_mode=running_mode, num_hands=2
    )

    pose_model = vision.PoseLandmarker.create_from_options(pose_options)
    face_model = vision.FaceLandmarker.create_from_options(face_options)
    hands_model = vision.HandLandmarker.create_from_options(hands_options)

    print(f"Worker process {os.getpid()} initialized.")
