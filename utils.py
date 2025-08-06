from mediapipe.tasks.python import BaseOptions, vision
from mediapipe import solutions

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

pose_kps_idx = tuple(
    (
        solutions.pose.PoseLandmark.LEFT_SHOULDER,
        solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        solutions.pose.PoseLandmark.LEFT_ELBOW,
        solutions.pose.PoseLandmark.RIGHT_ELBOW,
        solutions.pose.PoseLandmark.LEFT_WRIST,
        solutions.pose.PoseLandmark.RIGHT_WRIST,
    )
)
face_kps_idx = tuple(
    sorted(
        set(
            point
            for edge in [
                *solutions.face_mesh_connections.FACEMESH_CONTOURS,
                *solutions.face_mesh_connections.FACEMESH_IRISES,
            ]
            for point in edge
        )
    )
)
hand_kps_idx = tuple(range(len(solutions.hands.HandLandmark)))

mp_pose_nose_idx = solutions.pose.PoseLandmark.NOSE
mp_pose_shoulders_idx = (
    solutions.pose.PoseLandmark.LEFT_SHOULDER,
    solutions.pose.PoseLandmark.RIGHT_SHOULDER,
)

POSE_KPS_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
FACE_KPS_CONNECTIONS = [
    *solutions.face_mesh_connections.FACEMESH_CONTOURS,
    *solutions.face_mesh_connections.FACEMESH_IRISES,
]
HAND_KPS_CONNECTIONS = solutions.hands_connections.HAND_CONNECTIONS


mp_face_nose_idx = sorted(solutions.face_mesh_connections.FACEMESH_NOSE)[0][0]
left_iris = solutions.face_mesh_connections.FACEMESH_LEFT_IRIS
right_iris = solutions.face_mesh_connections.FACEMESH_RIGHT_IRIS
mp_face_eyes_idx = (list(sorted(left_iris))[0][0], list(sorted(right_iris))[0][0])

mp_hand_wrist_idx = solutions.hands.HandLandmark.WRIST
mp_hands_palm_idx = (
    solutions.hands.HandLandmark.THUMB_MCP,
    solutions.hands.HandLandmark.PINKY_MCP,
)

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
