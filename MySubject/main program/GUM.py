import cv2
import numpy as np

############################################################################
##                       NonLinearFunction関数                            ##
############################################################################
def nonLinear(img, inv):
    H, W = img.shape[:2]
    phi = np.zeros((H, W))
    if inv != 1:
        phi[img == 0.0] = 0.0
        phi[img != 0.0] = np.log((1.0 - img[img != 0.0]) / img[img != 0.0])
        return phi
    else:
        return 1.0 / (1.0 + np.exp(img))

############################################################################
##                  The Ratio of Negative to Original                     ##
############################################################################
def getRation(img):
    H, W = img.shape[:2]
    guzai = np.zeros((H, W))
    guzai[img == 0.0] = 0.0
    guzai[img != 0.0] = (1.0 - img[img != 0.0]) / img[img != 0.0]
    return guzai

############################################################################
##                  The Ratio of Negative to Original                     ##
############################################################################



