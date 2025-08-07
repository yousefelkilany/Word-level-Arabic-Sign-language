from mediapipe.tasks.python import BaseOptions, vision
from mediapipe import solutions

DATA_DIR = "/kaggle/input/karsl-502"
KPS_DIR = "/kaggle/working/karsl-kps"
MS_30FPS = 1000 / 30

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
mp_pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options, running_mode=running_mode
)
mp_face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options, running_mode=running_mode, num_faces=1
)
mp_hands_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options, running_mode=running_mode, num_hands=2
)

mp_pose_landmark = solutions.pose.PoseLandmark
mp_facemesh = solutions.face_mesh_connections
mp_hand_landmark = solutions.hands.HandLandmark

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
HAND_KPS_CONNECTIONS = solutions.hands_connections.HAND_CONNECTIONS

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

POSE_KB2SLICE = (slice(0, POSE_NUM),)
FACE_KB2SLICE = (slice(POSE_NUM, POSE_NUM + FACE_NUM),)
RH_KB2SLICE = (slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM),)
LH_KB2SLICE = (
    slice(POSE_NUM + FACE_NUM + HAND_NUM, POSE_NUM + FACE_NUM + HAND_NUM * 2),
)
KP2SLICE = {
    "pose": POSE_KB2SLICE,
    "face": FACE_KB2SLICE,
    "rh": RH_KB2SLICE,
    "lh": LH_KB2SLICE,
}
