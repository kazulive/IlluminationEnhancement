import cv2
import numpy as np
############################################################################
##                            CLAHE関数                                   ##
##             (Contrast Limited Adaptive Histogram Equalization)         ##
############################################################################
def cleary(img, clip_limit=2, grid=(8,8)):
    # ヒストグラム平均化
    clahe = cv2.createCLAHE(clip_limit, grid)
    img = np.copy(img)
    clahe_adaptive = clahe.apply(img)
    th = clahe_adaptive.copy()
    return th