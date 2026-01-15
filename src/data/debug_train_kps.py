import cv2
import numpy as np

from core.constants import DATA_DIR
from data.debug_live_viz import visualize_debug_skeleton

data = np.load(f"{DATA_DIR}/train_X.mmap", mmap_mode="r")
sample_frame = data[0][30].reshape(-1, 3)
print(f"{sample_frame.shape = }")

img, report = visualize_debug_skeleton(sample_frame)
print("--- TRAINING DATA STATS ---")
print(report)
cv2.imwrite("debug_training_sample.jpg", img)
