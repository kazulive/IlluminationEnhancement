import numpy as np

# JND-Based Illumination Weakening Factor
def getWeight(I, k):
    H, W = I.shape[:2]
    T = np.zeros((H, W), dtype=np.float32)
    # visibility threshold を計算
    T[I <= 127] = 17. * (1. - np.sqrt(I[I <= 127] / 127.)) + 3.
    T[I > 127] = 3. / 128. * (I[I > 127]) + 3
    # betaを求める
    alpha = k * (-1. / 17. * T + 20. / 17.)
    return alpha
