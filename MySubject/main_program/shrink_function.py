import cv2
import numpy as np


def division(src, dst):
    div = src.copy()
    non_zero = dst != 0
    div[non_zero] = src[non_zero] / dst[non_zero]
    div[~non_zero] = 0.0
    return div

def shrink(img, bh, bv, lam):
    dv = np.array([[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]])
    dh = dv.T

    grad_h = cv2.filter2D(img, cv2.CV_64F, dh)
    grad_v = cv2.filter2D(img, cv2.CV_64F, dv)

    forward_h = grad_h + bh
    forward_v = grad_v + bv

    dh = division(forward_h, np.abs(forward_h)) * np.maximum(np.abs(forward_h) - 1/(2.0*lam), 0.0)
    dv = division(forward_v, np.abs(forward_v)) * np.maximum(np.abs(forward_v) - 1 / (2.0 * lam), 0.0)

    return dh, dv

