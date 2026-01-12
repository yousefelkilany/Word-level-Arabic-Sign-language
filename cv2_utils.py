import cv2
import numpy as np


class MotionDetector:
    def __init__(
        self,
        width: int = 320,
        sigma_factor: float = 0.005,
        thresh_val: int = 20,
        motion_thresh: float = 0.1,
    ):
        self.width = width
        self.sigma_factor = sigma_factor
        self.thresh_val = thresh_val
        self.motion_thresh = motion_thresh

        self.sigmax = sigma_factor * width
        self.ksize = int(6 * self.sigmax) + int(self.sigmax % 2)
        self.ksize = max(3, self.ksize)

        self.gaussian_kernel = cv2.getGaussianKernel(self.ksize, self.sigmax)

    def convert_small_gray(self, frame):
        if len(frame.shape) > 2 and frame.shape[2] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        w = frame.shape[1]
        if abs(w - self.width) > 1:
            scale = self.width / w
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return frame

    def detect(
        self, prev_frame: np.ndarray, frame: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        if not self.gaussian_kernel or not prev_frame or not frame:
            return False, prev_frame

        small_curr = self.convert_small_gray(frame)
        small_prev = self.convert_small_gray(prev_frame)

        diff = cv2.absdiff(small_prev, small_curr)

        blur = cv2.sepFilter2D(diff, -1, self.gaussian_kernel, self.gaussian_kernel)

        _, thresh = cv2.threshold(blur, self.thresh_val, 255, cv2.THRESH_BINARY)

        motion_ratio = cv2.countNonZero(thresh) / thresh.size
        return motion_ratio > self.motion_thresh, small_curr
