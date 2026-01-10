import cv2
import numpy as np


def gaussian_blur(diff, sigma_factor=0.005, kernel_dim=1):
    def ksize(sigma):
        return max(3, sigma + 1 - sigma % 2)

    h, w = diff.shape
    sigmax = sigma_factor * w
    sigmay = sigma_factor * h
    ksizex = ksize(int(6 * sigmax))
    ksizey = ksize(int(6 * sigmay))

    if kernel_dim == 2:
        return cv2.GaussianBlur(diff, (ksizex, ksizey), sigmaX=sigmax, sigmaY=sigmay)

    gaussian_x = cv2.getGaussianKernel(ksizex, sigmax)
    gaussian_y = cv2.getGaussianKernel(ksizey, sigmay)
    return cv2.sepFilter2D(diff, -1, gaussian_x, gaussian_y)


def detect_motion(prev_gray, gray, motion_thresh=0.1):
    if prev_gray is None or gray is None:
        return (None,) * 3

    diff = cv2.absdiff(prev_gray, gray)

    # (ksizex, ksizey), (sigmax, sigmay) = get_gaussian_kernel(frame, kernel_dim=2)
    # blur = cv2.GaussianBlur(diff, (ksizex, ksizey), sigmaX=sigmax, sigmaY=sigmay)

    # if not gaussian_x or not gaussian_y:
    #     gaussian_x, gaussian_y = get_gaussian_kernel(frame)
    # blur = cv2.sepFilter2D(diff, -1, gaussian_x, gaussian_y)

    blur = gaussian_blur(diff, kernel_dim=2)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    return blur, thresh, np.count_nonzero(thresh) // np.size(thresh) > motion_thresh
