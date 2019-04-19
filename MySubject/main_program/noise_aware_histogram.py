import cv2
import numpy as np

# ある領域での局所コントラスト
def local_contrast(img):
    pow_img = np.square(img)
    g_img = cv2.GaussianBlur(img, (5, 5), 2.0)
    g_pow_img = cv2.GaussianBlur(pow_img, (5, 5), 2.0)

    return np.square(cv2.divide(g_pow_img.astype(dtype=np.float32), g_img.astype(dtype=np.float32)))

# Relative Noise Levelを計算
def culcurate_rnl(img):
