import cv2
import numpy as np

from utils import get_default_logger


class MotionDetector:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        sigma_factor: float = 0.005,
        thresh_val: int = 20,
        motion_thresh: float = 0.001,
    ):
        self.width = width
        self.height = height
        self.sigma_factor = sigma_factor
        self.thresh_val = thresh_val
        self.motion_thresh = motion_thresh

        self.sigmay = sigma_factor * height
        self.ksize = int(6 * self.sigmay) + int(self.sigmay % 2)
        self.ksize = max(3, self.ksize)

        self.gaussian_kernel = cv2.getGaussianKernel(self.ksize, self.sigmay)

        self.logger = get_default_logger()

    def convert_small_gray(self, frame):
        if len(frame.shape) > 2 and frame.shape[2] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (h := frame.shape[0]) and abs(h - self.height) > 1:
            scale = self.height / h
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return frame

    def detect(
        self, prev_frame: np.ndarray, frame: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        if (
            self.gaussian_kernel is None
            or (prev_frame is None or prev_frame.size == 0)
            or (frame is None or frame.size == 0)
        ):
            return False, prev_frame

        # use resized frames from client side, and although the image after
        # resizing, while keeping the ratio, is different from the size of
        # images used in training which were 1:1, the extracted kps were
        # normalized, so this might have minimal effect
        small_curr = self.convert_small_gray(frame)
        small_prev = self.convert_small_gray(prev_frame)

        diff = cv2.absdiff(small_prev, small_curr)

        blur = cv2.sepFilter2D(diff, -1, self.gaussian_kernel, self.gaussian_kernel)

        _, thresh = cv2.threshold(blur, self.thresh_val, 255, cv2.THRESH_BINARY)

        motion_ratio = cv2.countNonZero(thresh) / cv2.countNonZero(small_curr)
        return motion_ratio > self.motion_thresh, small_curr
