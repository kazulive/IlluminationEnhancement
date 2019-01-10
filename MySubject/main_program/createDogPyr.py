import cv2
import numpy as np

def createDogPyr(imgPyr, pyr_num):
    dst = [imgPyr[pyr_num-1]]
    for i in range(pyr_num - 1, 0, -1):
        GE = cv2.pyrUp(imgPyr[i])
        L = cv2.subtract(imgPyr[i - 1], GE)
        dst.append(L)
    return dst