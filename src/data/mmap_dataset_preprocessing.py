import argparse
import gc as garbage_collect
import os
import pickle
from collections import defaultdict
from os.path import join as os_join

import cv2
import numpy as np
from tqdm import tqdm

from core.constants import DATA_OUTPUT_DIR, FEAT_NUM, NPZ_KPS_DIR, SEQ_LEN, SplitType


def load_raw_kps(
    split: SplitType, signers: list[str], signs: range
) -> tuple[list[np.ndarray], np.ndarray, dict[int, dict[int, int]]]:
    X, y = [], []
    signer_word_samples = defaultdict(lambda: defaultdict(int))
    for word in tqdm(signs, desc=f"Loading Raw KPS - {split}"):
        vids_cnt = 0
        for signer in signers:
            word_kps_path = os_join(
                NPZ_KPS_DIR, "all_kps", f"{signer}-{split}", f"{word:04}.npz"
            )
            try:
                word_kps = np.load(word_kps_path, allow_pickle=True)
                X.extend([kps.astype(np.float16) for kps in word_kps.values()])
                signer_word_samples[int(signer)][word] = len(word_kps)
                vids_cnt += len(word_kps)
            except FileNotFoundError as e:
                print(f"[ERROR] NO NPZ FOUND. error: {e}")
        y.extend([word] * vids_cnt)

    return X, np.array(y), signer_word_samples


def calculate_num_chunks(kps_len: int):
    """Calculates how many samples a sequence of a given length will produce."""
    if kps_len <= 1.15 * SEQ_LEN:
        return 1
    if kps_len <= 1.85 * SEQ_LEN:
        return 2
    return 3


def prepare_raw_kps(X: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    def fix_missing_kps(kps: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(kps)
        df = df.replace([np.inf, -np.inf, 0.0], np.nan)  # treat exact 0.0 as missing
        df = df.interpolate(method="linear", limit_direction="both", axis=0)
        df = df.fillna(0.0)

        if not np.isfinite(df.values).all():
            print("[ERROR - fix_missing_kps] some bad values in df")
        return df.values.astype(np.float32)

    def prepare_seq(kps: np.ndarray) -> np.ndarray:
        kps_len = kps.shape[0]
        flat_kps = kps.reshape(kps_len, FEAT_NUM * 3)
        flat_kps = fix_missing_kps(flat_kps)

        if kps_len <= 1.15 * SEQ_LEN:
            kps = cv2.resize(
                flat_kps,
                (SEQ_LEN, FEAT_NUM * 3)[::-1],
                interpolation=cv2.INTER_LINEAR,
            )

        elif kps_len <= 1.85 * SEQ_LEN:
            kps = np.array([kps[:SEQ_LEN, :], kps[-SEQ_LEN:, :]])

        else:
            step = (kps_len - SEQ_LEN) // 2
            kps = np.array(
                [kps[:SEQ_LEN, :], kps[step : step + SEQ_LEN, :], kps[-SEQ_LEN:, :]]
            )

        return kps.reshape(-1, SEQ_LEN, FEAT_NUM * 3)

    arr = [
        prepare_seq(kps)
        for kps in tqdm(X, desc="Raw KPS => Sequences", disable=(len(X) < 10))
    ]
    arr_lens = np.array([arr_i.shape[0] for arr_i in arr])
    arr = np.concatenate(arr, dtype=np.float32, axis=0)
    return arr, arr_lens


def prepare_labels(y: np.ndarray, X: list[np.ndarray]) -> np.ndarray:
    labels = [
        y_ for y_, x in zip(y, X) for _ in range(calculate_num_chunks(x.shape[0]))
    ]
    return np.array(labels, dtype=np.longlong) - 1


def mmap_process_and_save_split(
    split: SplitType,
    signers: list[str],
    signs: range,
    output_dir: str = f"{DATA_OUTPUT_DIR}/preprocessed_data",
):
    print(f"--- Processing split: {split} ---")

    X, y, signer_words_map = load_raw_kps(split, signers, signs)
    X_final, X_final_lens = prepare_raw_kps(X)
    y_final = prepare_labels(y, X)
    print(f"Final shape for {split} X: {X_final.shape}")
    print(f"Final total size: {X_final.nbytes / 1024**3:.2f} GB")
    print(f"Final shape for {split} y: {y_final.shape}")

    os.makedirs(output_dir, exist_ok=True)
    mmap_path = os_join(output_dir, f"{split}_X.mmap")
    fp = np.memmap(mmap_path, dtype="float32", mode="w+", shape=X_final.shape)
    fp[:] = X_final[:]
    fp.flush()

    np.save(os_join(output_dir, f"{split}_y.npy"), y_final)
    np.save(os_join(output_dir, f"{split}_X_shape.npy"), X_final.shape)
    np.save(os_join(output_dir, f"{split}_X_lens.npy"), X_final_lens)
    with open(os_join(output_dir, f"{split}_X_signer_words_map.pkl"), "wb") as f:
        pickle.dump(signer_words_map, f)

    print(f"Successfully saved {split} data to {output_dir}")

    del X, y, X_final, X_final_lens, y_final
    garbage_collect.collect()
    print("Memory cleared.")
    print(f"--- Finished processing split: {split} ---\n")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", required=False, default=["train", "test"]
    )
    parser.add_argument(
        "--signers", nargs="+", required=False, default=["01", "02", "03"]
    )
    parser.add_argument("--selected_words_from", required=False, type=int, default=1)
    parser.add_argument("--selected_words_to", required=False, type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = cli()
    print("Extracting keypoints from frames...")
    print("Arguments:", cli_args)
    for split in cli_args.splits:
        mmap_process_and_save_split(
            split=split,
            signers=cli_args.signers,
            signs=range(cli_args.selected_words_from, cli_args.selected_words_to + 1),
        )
