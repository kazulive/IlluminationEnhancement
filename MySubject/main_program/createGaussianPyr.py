import cv2
import numpy as np

############################################################################
##                       ガウシアンピラミッド作成                         ##
############################################################################
def createGaussianPyr(img, num):
    dst = list()
    dst.append(img)
    for i in range(1, num):
        nowdst = cv2.pyrDown(dst[i-1])
        dst.append(nowdst)
    return dst
