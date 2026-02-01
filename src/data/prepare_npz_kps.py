import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import cv2
import numpy as np
from tqdm import tqdm

from core.constants import DATA_OUTPUT_DIR, KARSL_DATA_DIR, os_join
from core.mediapipe_utils import LandmarkerProcessor

# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_worker_processor = None


def init_worker(landmarkers=None):
    """
    This function runs once when each worker process starts.
    It initializes the MediaPipe models globally for that process.
    """
    global _worker_processor
    _worker_processor = LandmarkerProcessor.create(
        landmarkers=landmarkers, inference_mode=False
    )


def process_video(video_dir, adjusted):
    global _worker_processor
    if _worker_processor is None:
        raise RuntimeError(
            "Worker processor not initialized. Check ProcessPoolExecutor."
        )

    video_kps = []
    for frame in sorted(os.listdir(video_dir)):
        frame_path = os_join(video_dir, frame)
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kps = _worker_processor.extract_frame_keypoints(frame_rgb, adjusted=adjusted)
        video_kps.append(kps)

    return np.array(video_kps) if video_kps else None


def process_sign_wrapper(sign_processing_info):
    (sign, signers, splits, adjusted) = sign_processing_info

    if 1 > sign or sign > 502:
        return
    sign = f"{sign:04}"
    print(f"Processing and Saving videos - sign {sign}")

    for signer, split in tqdm(
        product(signers, splits),
        total=len(signers) * len(splits),
        desc=f"Sign {sign} videos",
    ):
        sign_dir = os_join(KARSL_DATA_DIR, signer, signer, split, sign)
        if not os.path.exists(sign_dir):
            print(f"Skipping {sign_dir}, since it doesn't exist.")
            continue

        sign_keypoints = dict()
        for video_name in os.listdir(sign_dir):
            video_dir = os_join(sign_dir, video_name)
            if os.path.isdir(video_dir):
                video_kps = process_video(video_dir, adjusted)
                if video_kps is not None:
                    sign_keypoints[video_name] = video_kps

        try:
            kps_path = os_join(DATA_OUTPUT_DIR, "karsl-kps", f"{signer}-{split}", sign)
            os.makedirs(os.path.dirname(kps_path), exist_ok=True)
            np.savez_compressed(kps_path, **sign_keypoints)

        except Exception as e:
            print(f"Error saving file for key {({signer}, {split}, {sign})}: {e}")

    print(f"[DONE] sign {sign} - Processed and Saved videos")
    return True


def extract_keypoints_from_frames(
    splits=None, signers=None, signs=None, adjusted=False
):
    splits = splits or ["train", "test"][-1:]
    signers = signers or ["01", "02", "03"][-1:]
    signs = signs or range(1, 2)

    num_workers = os.cpu_count() or 2
    print(f"Starting processing Signs tasks with {num_workers} workers...")

    signs_tasks = [(s, signers, splits, adjusted) for s in signs]
    target_landmarkers = ["pose", "face", "hands"]

    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_worker, initargs=(target_landmarkers,)
    ) as executor:
        list(
            tqdm(
                executor.map(process_sign_wrapper, signs_tasks),
                total=len(signs_tasks),
                desc="Processing Tasks",
            )
        )


def npz_kps_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument("--signers", nargs="*", default=["01", "02", "03"])
    parser.add_argument("--selected_signs_from", default=1, type=int)
    parser.add_argument("--selected_signs_to", default=1, type=int)
    parser.add_argument("--adjusted", required=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = npz_kps_cli()
    print("Extracting keypoints from frames, then storing as NPZ files...")
    print("Arguments:", cli_args)
    extract_keypoints_from_frames(
        splits=cli_args.splits,
        signers=cli_args.signers,
        signs=range(cli_args.selected_signs_from, cli_args.selected_signs_to + 1),
        adjusted=cli_args.adjusted,
    )
