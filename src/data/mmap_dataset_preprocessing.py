import argparse
import gc as garbage_collect
import os
from itertools import product

import numpy as np
from tqdm import tqdm

from core.constants import MMAP_OUTPUT_PREPROCESSED_DIR, NPZ_KPS_DIR, SplitType, os_join


def load_raw_kps(
    split: SplitType, signers: list[str], signs: range
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[int, list]]]:
    X, y = [], []
    X_map_samples_lens = dict()

    for sign, signer in tqdm(
        product(signs, signers), desc=f"Loading Raw KPS - {split}"
    ):
        sign_kps_path = os_join(NPZ_KPS_DIR, f"{signer}-{split}", f"{sign:04}.npz")
        try:
            sign_kps = np.load(sign_kps_path, allow_pickle=True)
            X.extend([kps.astype(np.float16) for kps in sign_kps.values()])
            y.extend([sign] * len(sign_kps))
            if X_map_samples_lens.get(sign) is None:
                X_map_samples_lens[sign] = dict()
            X_map_samples_lens[sign][int(signer)] = [
                kps.shape[0] for kps in sign_kps.values()
            ]
        except FileNotFoundError as e:
            print(f"[ERROR] NO NPZ FOUND. error: {e}")

    X = np.concatenate(X, dtype=np.float32, axis=0)

    return X, np.array(y), dict(X_map_samples_lens)


def mmap_process_and_save_split(
    split: SplitType,
    signers: list[str],
    signs: range,
):
    print(f"--- Processing split: {split} ---")

    X, y, X_map_samples_lens = load_raw_kps(split, signers, signs)
    y = np.array(y, dtype=np.longlong) - 1
    print(f"Final shape for {split} X: {X.shape}")
    print(f"Final total size: {X.nbytes / 1024**3:.2f} GB")
    print(f"Final shape for {split} y: {y.shape}")

    os.makedirs(MMAP_OUTPUT_PREPROCESSED_DIR, exist_ok=True)
    mmap_path = os_join(MMAP_OUTPUT_PREPROCESSED_DIR, f"{split}_X.mmap")
    fp = np.memmap(mmap_path, dtype="float32", mode="w+", shape=X.shape)
    fp[:] = X[:]
    fp.flush()

    np.savez_compressed(os_join(MMAP_OUTPUT_PREPROCESSED_DIR, f"{split}_y.npz"), y)
    np.save(os_join(MMAP_OUTPUT_PREPROCESSED_DIR, f"{split}_X_shape.npy"), X.shape)
    np.save(
        os_join(MMAP_OUTPUT_PREPROCESSED_DIR, f"{split}_X_map_samples_lens.npy"),
        X_map_samples_lens,  # ty:ignore[invalid-argument-type]
    )

    print(f"Successfully saved {split} data to {MMAP_OUTPUT_PREPROCESSED_DIR}")

    del X, y
    garbage_collect.collect()
    print("Memory cleared.")
    print(f"--- Finished processing split: {split} ---\n")


def mmap_dataset_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--signers", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--selected_signs_from", type=int, default=1)
    parser.add_argument("--selected_signs_to", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = mmap_dataset_cli()
    print("Extracting keypoints from frames...")
    print("Arguments:", cli_args)
    for split in cli_args.splits:
        mmap_process_and_save_split(
            split=split,
            signers=cli_args.signers,
            signs=range(cli_args.selected_signs_from, cli_args.selected_signs_to + 1),
        )
