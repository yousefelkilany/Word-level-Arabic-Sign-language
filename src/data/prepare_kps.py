import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import cv2
import numpy as np
from tqdm import tqdm

from core.constants import KARSL_DATA_DIR, DATA_OUTPUT_DIR

# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os_join = os.path.join


def process_video(video_dir, adjusted):
    video_kps = []
    for idx, frame in enumerate(sorted(os.listdir(video_dir))):
        frame_path = os_join(video_dir, frame)

        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # FIXME: use LandmarkerProcessor instead, AND save visibility alongside x,y,z coords
        # video_kps.append(extract_frame_keypoints(frame_rgb, adjusted))

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
    video_group_key, videos_name_kps = result
    signer, split, word = video_group_key

    try:
        word_kps_path = os_join(DATA_OUTPUT_DIR, "all_kps", f"{signer}-{split}", word)
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
            word_dir = os_join(KARSL_DATA_DIR, signer, signer, split, word)
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

    # FiXME: use LandmarkerProcessor.create instead of init_mediapipe_worker function
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_mediapipe_worker
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
