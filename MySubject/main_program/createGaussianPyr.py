import cv2
import numpy as np

# ガウシアンピラミッド作成
def createGaussianPyr(img):
    dst = list()
    dst.append(img)
    for i in range(1, 4):
        nowdst = cv2.pyrDown(dst[i-1])
        dst.append(nowdst)
    return dst
