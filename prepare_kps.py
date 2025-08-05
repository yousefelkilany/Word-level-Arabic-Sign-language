from collections import defaultdict
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from mediapipe.tasks.python import BaseOptions, vision

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os_join = os.path.join

DATA_DIR = "/kaggle/input/karsl-502"
KPS_DIR = "/kaggle/working/karsl-kps"
MS_30FPS = 1000 / 30

VisionRunningMode = vision.RunningMode

pose_base_options = BaseOptions(model_asset_path="pose_landmarker.task")
face_base_options = BaseOptions(
    model_asset_path="face_landmarker_v2_with_blendshapes.task"
)
hand_base_options = BaseOptions(model_asset_path="hand_landmarker.task")
mp_pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options, running_mode=VisionRunningMode.VIDEO
)
mp_face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options, running_mode=VisionRunningMode.VIDEO, num_faces=1
)
mp_hands_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options, running_mode=VisionRunningMode.VIDEO, num_hands=2
)

mp_pose_nose_idx = mp.solutions.pose.PoseLandmark.NOSE
mp_face_nose_idx = sorted(mp.solutions.face_mesh_connections.FACEMESH_NOSE)[0][0]
mp_hand_wrist_idx = mp.solutions.hands.HandLandmark.WRIST

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

POSE_NUM = len(pose_kps_idx)
FACE_NUM = len(face_kps_idx)
HAND_NUM = len(hand_kps_idx)

KP2SLICE = {
    "pose": slice(0, POSE_NUM),
    "face": slice(POSE_NUM, POSE_NUM + FACE_NUM),
    "rh": slice(POSE_NUM + FACE_NUM, POSE_NUM + FACE_NUM + HAND_NUM),
    "lh": slice(POSE_NUM + FACE_NUM + HAND_NUM, POSE_NUM + FACE_NUM + HAND_NUM * 2),
}


def init_worker():
    global pose_model, face_model, hands_model
    pose_model = vision.PoseLandmarker.create_from_options(mp_pose_options)
    face_model = vision.FaceLandmarker.create_from_options(mp_face_options)
    hands_model = vision.HandLandmarker.create_from_options(mp_hands_options)
    print(f"Worker process {os.getpid()} initialized.")


def extract_frame_keypoints(frame_rgb, timestamp):
    # TODO: normalize(?) keypoints after adjustment

    # define numpy views, pose -> face -> rh -> lh
    all_kps = np.zeros((184, 3))  # (pose=6 + face=136 + rh+lh=42), xyz=3
    pose_kps = all_kps[KP2SLICE["pose"]]
    face_kps = all_kps[KP2SLICE["face"]]
    rh_kps = all_kps[KP2SLICE["rh"]]
    lh_kps = all_kps[KP2SLICE["lh"]]
    np_xyz = np.dtype((float, 3))

    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    def get_pose():
        results = pose_model.detect_for_video(frame, timestamp)
        if results.pose_landmarks is None:
            return

        lms = results.pose_landmarks[0]
        pose_kps[:] = np.fromiter(
            ((lms[idx].x, lms[idx].y, lms[idx].z) for idx in pose_kps_idx), dtype=np_xyz
        )
        # pose_kps -= pose_kps[mp_pose_nose_idx]

    def get_face():
        results = face_model.detect_for_video(frame, timestamp)
        if results.face_landmarks is None:
            return

        lms = results.face_landmarks[0]
        face_kps[:] = np.fromiter(
            ((lms[idx].x, lms[idx].y, lms[idx].z) for idx in face_kps_idx), dtype=np_xyz
        )
        # face_kps -= face_kps[mp_face_nose_idx]

    def get_hands():
        results = hands_model.detect_for_video(frame, timestamp)
        if results.hand_landmarks is None:
            return

        for handedness, hand_lms in zip(results.handedness, results.hand_landmarks):
            target_hand = lh_kps if handedness[0].category_name == "Left" else rh_kps
            target_hand[:] = np.fromiter(
                ((lm.x, lm.y, lm.z) for lm in hand_lms), dtype=np_xyz
            )
            # target_hand -= target_hand[mp_hand_wrist_idx]

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(get_pose)
        executor.submit(get_face)
        executor.submit(get_hands)

    return all_kps


def process_video(video_dir):
    video_kps = []
    for idx, frame in enumerate(sorted(os.listdir(video_dir))):
        frame_path = os_join(video_dir, frame)

        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp = int(idx * MS_30FPS)
        video_kps.append(extract_frame_keypoints(frame_rgb, timestamp))

    return np.array(video_kps) if video_kps else None


def process_video_wrapper(task_info):
    """
    A wrapper that calls the real worker and returns the result
    along with the identifiers needed for grouping.
    """
    word_dir, video_name, signer, split, word = task_info

    video_dir = os_join(word_dir, video_name)
    video_group_key = (signer, split, word)
    keypoints_array = process_video(video_dir)
    return (video_group_key, keypoints_array)


def save_grouped_results(result):
    try:
        video_group_key, videos_kps = result
        signer, split, word = video_group_key

        word_kps_path = os_join(KPS_DIR, "all_kps", f"{signer}-{split}", word)
        os.makedirs(os.path.dirname(word_kps_path), exist_ok=True)

        final_keypoints = np.concatenate(videos_kps, axis=0)
        np.savez_compressed(word_kps_path, keypoints=final_keypoints)
        return True

    except Exception as e:
        print(f"Error saving file for key {video_group_key}: {e}")


def extract_keypoints_from_frames(
    splits=None, signers=None, selected_words=None, max_videos=None
):
    splits = splits or ["train", "test"]
    signers = signers or ["01", "02", "03"]
    selected_words = selected_words or tuple((f"{v:04}" for v in range(1, 503)))

    print("Stage 1: Generating task list...")
    videos_tasks = []
    for word in tqdm(selected_words, desc="Words"):
        for signer in signers:
            for split in splits:
                word_dir = os_join(DATA_DIR, signer, signer, split, word)
                for video_name in os.listdir(word_dir)[:max_videos]:
                    videos_tasks.append((word_dir, video_name, signer, split, word))

    if not videos_tasks:
        print("No videos found to process. Exiting.")
        return

    print(f"Generated {len(videos_tasks)} video processing tasks.")

    num_workers = os.cpu_count()
    print(f"\nStage 2: Executing tasks with {num_workers} workers...")

    # chunksize = int(len(videos_tasks) / num_workers + 0.5)  # ceiling
    # print(f"Using {num_workers} workers with chunksize={chunksize}")
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_worker
    ) as executor:
        # experiment with chunksize=chunksize
        results_itr = executor.map(process_video_wrapper, videos_tasks)
        videos_results = [
            result
            for result in tqdm(
                results_itr, total=len(videos_tasks), desc="Processing Videos"
            )
        ]

    print("\nStage 3: Grouping results...")
    grouped_results = defaultdict(list)
    for video_group_key, keypoints_array in tqdm(videos_results, desc="Grouping"):
        grouped_results[video_group_key].append(keypoints_array)

    print(f"\nStage 4: Saving {len(grouped_results)} NPZ files...")
    save_tasks = grouped_results.items()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        save_itr = executor.map(save_grouped_results, save_tasks)
        for _ in tqdm(save_itr, total=len(grouped_results), desc="Saving NPZ files"):
            ...


if __name__ == "__main__":
    extract_keypoints_from_frames()
