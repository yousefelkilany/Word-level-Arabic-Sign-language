import cv2

from data.dataloader import prepare_mmap_dataloaders, prepare_lazy_dataloaders
from data.debug_live_viz import visualize_debug_skeleton


def test_mmap_dataset():
    train_dl, val_dl, test_dl = prepare_mmap_dataloaders()

    for kps, labels in train_dl:
        kps, labels = kps, labels
        print(f"{kps.shape = }")
        print(f"{labels.shape = }")

        sample_frame = kps[0][30].reshape(-1, 3)
        print(f"{sample_frame.shape = }")

        img, report = visualize_debug_skeleton(sample_frame)
        print("--- TRAINING DATA STATS ---")
        print(report)
        cv2.imwrite("debug_training_sample.jpg", img)

        break


def test_lazy_dataset():
    train_dl, val_dl, test_dl = prepare_lazy_dataloaders(range(2), batch_size=64)

    for kps, labels in train_dl:
        kps, labels = kps, labels
        print(f"{kps.shape = }")
        print(f"{labels.shape = }")

        # sample_frame = kps[0][30].reshape(-1, 3)
        # print(f"{sample_frame.shape = }")

        # img, report = visualize_debug_skeleton(sample_frame)
        # print("--- TRAINING DATA STATS ---")
        # print(report)
        # cv2.imwrite("debug_training_sample.jpg", img)

        break


if __name__ == "__main__":
    # test_mmap_dataset()
    test_lazy_dataset()
