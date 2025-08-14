import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import mediapipe as mp

from utils import (
    DATA_DIR,
    KPS_DIR,
    KP2SLICE,
    init_mediapipe_worker,
    pose_kps_idx,
    mp_pose_nose_idx,
    mp_pose_shoulders_idx,
    face_kps_idx,
    mp_face_nose_idx,
    mp_face_eyes_idx,
    mp_hand_wrist_idx,
    mp_hands_palm_idx,
)


# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os_join = os.path.join

# set by `init_mediapipe_worker`
pose_model, face_model, hands_model = None, None, None


def extract_frame_keypoints(frame_rgb, adjusted=False):
    # define numpy views, pose=6 -> face=136 -> rh=21 -> lh=21
    all_kps = np.zeros((184, 3))
    pose_kps = all_kps[KP2SLICE["pose"]]
    face_kps = all_kps[KP2SLICE["face"]]
    rh_kps = all_kps[KP2SLICE["rh"]]
    lh_kps = all_kps[KP2SLICE["lh"]]
    np_xyz = np.dtype((float, 3))

    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    def landmarks_distance(lms_list, lm_idx):
        p1, p2 = lms_list[lm_idx[0]], lms_list[lm_idx[1]]
        # return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
        # EPS = 1e-10
        # return (abs(p1.x - p2.x) + EPS, abs(p1.y - p2.y) + EPS, abs(p1.z - p2.z) + EPS)
        return (abs(p1.x - p2.x), abs(p1.y - p2.y), abs(p1.z - p2.z))

    def get_pose():
        nonlocal pose_kps
        results = pose_model.detect(frame)
        if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
            return

        lms = results.pose_landmarks[0]
        pose_kps[:] = np.fromiter(
            ((lms[idx].x, lms[idx].y, lms[idx].z) for idx in pose_kps_idx), dtype=np_xyz
        )
        if adjusted:
            pose_kps -= pose_kps[mp_pose_nose_idx]
            pose_kps /= landmarks_distance(lms, mp_pose_shoulders_idx)

    def get_face():
        nonlocal face_kps
        results = face_model.detect(frame)
        if results.face_landmarks is None or len(results.face_landmarks) == 0:
            return

        lms = results.face_landmarks[0]
        face_kps[:] = np.fromiter(
            ((lms[idx].x, lms[idx].y, lms[idx].z) for idx in face_kps_idx), dtype=np_xyz
        )
        if adjusted:
            face_kps -= face_kps[mp_face_nose_idx]
            face_kps /= landmarks_distance(lms, mp_face_eyes_idx)

    def get_hands():
        nonlocal rh_kps, lh_kps
        results = hands_model.detect(frame)
        if results.hand_landmarks is None:
            return

        for handedness, hand_lms in zip(results.handedness, results.hand_landmarks):
            target_hand = lh_kps if handedness[0].category_name == "Left" else rh_kps
            target_hand[:] = np.fromiter(
                ((lm.x, lm.y, lm.z) for lm in hand_lms), dtype=np_xyz
            )
            if adjusted:
                target_hand -= target_hand[mp_hand_wrist_idx]
                target_hand /= landmarks_distance(hand_lms, mp_hands_palm_idx)

    with ThreadPoolExecutor(max_workers=3) as executor:
        _pose_res = executor.submit(get_pose)
        _face_res = executor.submit(get_face)
        _hand_res = executor.submit(get_hands)
        _pose_res.result()
        _face_res.result()
        _hand_res.result()

    return all_kps


def process_video(video_dir, adjusted):
    video_kps = []
    for idx, frame in enumerate(sorted(os.listdir(video_dir))):
        frame_path = os_join(video_dir, frame)

        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_kps.append(extract_frame_keypoints(frame_rgb, adjusted))

    return np.array(video_kps) if video_kps else None


def process_video_wrapper(task_info):
    """
    A wrapper that calls the real worker and returns the result
    along with the identifiers needed for grouping.
    """
    word_dir, video_name, signer, split, word, adjusted = task_info

    video_dir = os_join(word_dir, video_name)
    video_group_key = (signer, split, word)
    video_kps = process_video(video_dir, adjusted)
    return (video_group_key, video_name, video_kps)


def save_grouped_results(result):
    try:
        video_group_key, videos_name_kps = result
        signer, split, word = video_group_key

        word_kps_path = os_join(KPS_DIR, "all_kps", f"{signer}-{split}", word)
        os.makedirs(os.path.dirname(word_kps_path), exist_ok=True)

        final_keypoints = {video_name: kps for video_name, kps in videos_name_kps}
        np.savez_compressed(word_kps_path, **final_keypoints)
        return True

    except Exception as e:
        print(f"Error saving file for key {video_group_key}: {e}")


def extract_keypoints_from_frames(
    splits=None, signers=None, selected_words=None, max_videos=None, adjusted=False
):
    splits = splits or ["train", "test"][-1:]
    signers = signers or ["01", "02", "03"][-1:]
    selected_words = list(selected_words or range(1, 2))

    print(
        f"Stage 1: Generating task list for words {selected_words[0]} to {selected_words[-1]}..."
    )
    videos_tasks = []
    for word in tqdm(selected_words, desc="Words"):
        if 1 > word or word > 502:
            break
        word = f"{word:04}"
        for signer, split in product(signers, splits):
            word_dir = os_join(DATA_DIR, signer, signer, split, word)
            for video_name in os.listdir(word_dir)[:max_videos]:
                videos_tasks.append(
                    (word_dir, video_name, signer, split, word, adjusted)
                )

    if not videos_tasks:
        print("No videos found to process. Exiting.")
        return

    print(f"Generated {len(videos_tasks)} video processing tasks.")

    num_workers = os.cpu_count()
    print(f"\nStage 2: Executing tasks with {num_workers} workers...")

    # chunksize = int(len(videos_tasks) / num_workers + 0.5)  # ceiling
    # print(f"Using {num_workers} workers with chunksize={chunksize}")
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_mediapipe_worker()
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
    for video_group_key, video_name, video_kps in tqdm(videos_results, desc="Grouping"):
        grouped_results[video_group_key].append((video_name, video_kps))

    print(f"\nStage 4: Saving {len(grouped_results)} NPZ files...")
    save_tasks = grouped_results.items()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        save_itr = executor.map(save_grouped_results, save_tasks)
        for _ in tqdm(save_itr, total=len(grouped_results), desc="Saving NPZ files"):
            ...


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--signers", nargs="+", default=None)
    parser.add_argument("--selected_words_from", type=int, default=0)
    parser.add_argument("--selected_words_to", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--adjusted", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = cli()
    print("Extracting keypoints from frames...")
    print("Arguments:", cli_args)
    extract_keypoints_from_frames(
        splits=cli_args.splits,
        signers=cli_args.signers,
        selected_words=range(
            cli_args.selected_words_from, cli_args.selected_words_to + 1
        ),
        max_videos=cli_args.max_videos,
        adjusted=cli_args.adjusted,
    )
