import numpy as np

# JND-Based Illumination Weakening Factor
def get_jnd_threshold(I, k1, k2, lam1, lam2):
    """
    :param I: Image (Input or Enhanced)
    :param k1: the weight of first term (I <= 128)
    :param k2: the weight of first term (I > 128)
    :param lam1: the weight of first multiplier (I <= 128)
    :param lam2: the weight of fist multiplier (I > 128)
    :return: V: the visibility threshold
    """
    H, W = I.shape[:2]
    V = np.zeros((H, W), dtype=np.float32)
    # visibility threshold を計算
    V[I <= 128] = k1 * ((1. - (2.*I)/256) ** lam1) + 1.
    V[I > 128] = k2 * ((2.*I/256 - 1) ** lam2) + 1.
    return V
