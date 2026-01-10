import argparse
import gc as garbage_collect
import os
from os.path import join as os_join

import numpy as np
from tqdm import tqdm

KAGGLE_KPS_DIR = "/kaggle/input"
LOCAL_DEV = int(os.environ.get("LOCAL_DEV", 0))
KPS_DIR = os_join([KAGGLE_KPS_DIR, "data"][LOCAL_DEV], "karsl-kps")
SEQ_LEN = 60
FEAT_NUM = 184


def load_raw_kps(split, signers, selected_words):
    X, y = [], []
    for word in tqdm(selected_words, desc=f"Loading Raw KPS - {split}"):
        vids_cnt = 0
        for signer in signers:
            word_kps_path = os_join(
                KPS_DIR, "all_kps", f"{signer}-{split}", f"{word:04}.npz"
            )
            try:
                word_kps = np.load(word_kps_path, allow_pickle=True)
                # X.extend(list(word_kps.values()))
                # lower float precision to save memory
                X.extend([kps.astype(np.float16) for kps in word_kps.values()])
                vids_cnt += len(word_kps)
            except FileNotFoundError:
                continue
        y.extend([word] * vids_cnt)

    return X, np.array(y)


def calculate_num_chunks(kps_len):
    """Calculates how many samples a sequence of a given length will produce."""
    if SEQ_LEN > kps_len:
        return 1
    if kps_len % SEQ_LEN >= (SEQ_LEN * 2 // 3):
        return kps_len // SEQ_LEN + 1
    return kps_len // SEQ_LEN


def prepare_raw_kps(X):
    def pad_split_seq(kps):
        # Pad sequences (with length < SEQ_LEN) to SEQ_LEN, no matter what is its length.
        kps_len = kps.shape[0]
        if SEQ_LEN > kps_len:
            kps = np.concatenate([kps, np.tile(kps[-1], (SEQ_LEN - kps_len, 1, 1))])

        # If sequence is longer, slice it into x sequences with the last slice filled if
        # it's 2/3 of SEQ_LEN, otherwise it's too short to be padded and is dropped.
        elif kps_len % SEQ_LEN >= (SEQ_LEN * 2 // 3):
            tile_cnt = (kps_len // SEQ_LEN + 1) * SEQ_LEN - kps_len
            kps = np.concatenate([kps, np.tile(kps[-1], (tile_cnt, 1, 1))])

        else:
            kps = kps[: (kps_len // SEQ_LEN) * SEQ_LEN]

        # Collapse last two dimensions, 184x3 to 552
        kps = kps.reshape(-1, SEQ_LEN, FEAT_NUM * 3)
        return np.nan_to_num(kps, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate(
        [pad_split_seq(kps) for kps in tqdm(X, desc="Raw KPS => Sequences")],
        dtype=np.float32,
    )


def prepare_labels(y, X):
    y = [y_ for y_, x in zip(y, X) for _ in range(calculate_num_chunks(x.shape[0]))]
    return np.array(y, dtype=np.longlong) - 1


def process_and_save_split(
    split, signers, selected_words, output_dir="preprocessed_data"
):
    print(f"--- Processing split: {split} ---")

    X, y = load_raw_kps(split, signers, selected_words)
    X_final = prepare_raw_kps(X)
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
    np.save(os_join(output_dir, f"{split}_X_shape.npy"), np.array(X_final.shape))
    print(f"Successfully saved {split} data to {output_dir}")

    del X, y, X_final, y_final
    garbage_collect.collect()
    print("Memory cleared.")
    print(f"--- Finished processing split: {split} ---\n")


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=splits)
    parser.add_argument("--signers", nargs="+", default=signers)
    parser.add_argument("--selected_words_from", type=int, default=1)
    parser.add_argument("--selected_words_to", type=int, default=num_words)
    return parser.parse_args()


if __name__ == "__main__":
    splits = ["train", "test"]
    signers = ["01", "02", "03"]
    num_words = 502

    args = cli_args()
    signers = args.signers
    selected_words = range(args.selected_words_from, args.selected_words_to + 1)

    for split in args.splits:
        process_and_save_split(split, signers, selected_words)
