"""
Bright Channelを作成する関数
"""
import numpy as np

def getBrightChannel(I, w):
    M, N, _ = I.shape
    padded = np.pad(I, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    bright = np.zeros((M, N))
    for i, j in np.ndindex(bright.shape):
        bright[i, j] = np.max(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return bright