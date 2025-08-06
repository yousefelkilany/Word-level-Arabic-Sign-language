import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os_join = os.path.join

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
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    )
)
face_kps_idx = tuple(
    sorted(
        set(
            point
            for edge in [
                *mp.solutions.face_mesh_connections.FACEMESH_CONTOURS,
                *mp.solutions.face_mesh_connections.FACEMESH_IRISES,
            ]
            for point in edge
        )
    )
)
hand_kps_idx = tuple(range(len(mp.solutions.hands.HandLandmark)))

mp_pose_nose_idx = mp.solutions.pose.PoseLandmark.NOSE
mp_pose_shoulders_idx = (
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
)

mp_face_nose_idx = sorted(mp.solutions.face_mesh_connections.FACEMESH_NOSE)[0][0]
left_iris = mp.solutions.face_mesh_connections.FACEMESH_LEFT_IRIS
right_iris = mp.solutions.face_mesh_connections.FACEMESH_RIGHT_IRIS
mp_face_eyes_idx = (list(sorted(left_iris))[0][0], list(sorted(right_iris))[0][0])

mp_hand_wrist_idx = mp.solutions.hands.HandLandmark.WRIST
mp_hands_palm_idx = (
    mp.solutions.hands.HandLandmark.THUMB_MCP,
    mp.solutions.hands.HandLandmark.PINKY_MCP,
)

POSE_NUM = len(pose_kps_idx)
FACE_NUM = len(face_kps_idx)
HAND_NUM = len(hand_kps_idx)

KP2SLICE = {
    "pose": slice(0, POSE_NUM),
    "face": slice(POSE_NUM, POSE_NUM + FACE_NUM),
    "rh": slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM),
    "lh": slice(POSE_NUM + FACE_NUM + HAND_NUM, POSE_NUM + FACE_NUM + HAND_NUM * 2),
}

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


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
    POSE_KPS_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    lms_num = len(mp.solutions.pose.PoseLandmark)
    return draw_kps_on_image(
        rgb_image, pose_kps, pose_kps_idx, lms_num, POSE_KPS_CONNECTIONS
    )


DATA_DIR = "/kaggle/input/karsl-502"
KPS_DIR = "/kaggle/working/karsl-kps"

signer, split, word = "03", "test", f"{502:04}"
kps_path = os.path.join(KPS_DIR, "all_kps", f"{signer}-{split}", f"{word}.npz")

npz0001 = np.load(kps_path, allow_pickle=True)
video_name, video_kps = list(npz0001.items())[0]
# print(video_kps.min(), video_kps.max())
video_dir = os.path.join(DATA_DIR, signer, signer, split, word, video_name)
video_frames = sorted(os.listdir(video_dir))

# video_frames[0], video_kps[0]

frame_idx = 0
frame = cv2.imread(os.path.join(video_dir, video_frames[frame_idx]))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

kps = video_kps[frame_idx][KP2SLICE["pose"]]
annotated_frame = draw_pose_kps_on_image(frame, kps)
cv2.imwrite("annotated_frame.jpg", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
