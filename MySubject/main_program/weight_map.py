import cv2
import numpy as np

sigma = 0.3
ave = 0.5

def weight_map(img):
    img_norm = img / np.max(img)
    weight = np.exp(-1 * (img_norm - ave) ** 2 / (2.0 * sigma ** 2))
    return weight