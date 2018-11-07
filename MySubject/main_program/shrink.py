import cv2
import numpy as np

############################################################################
##                            shrink関数                                  ##
############################################################################
def nonLinearStretch(img):
    mean = cv2.mean(img)
    a = 10 + (1 - mean[0]) / mean[0]
    luminance_Adjusted = 2 * np.arctan(a * img) / np.pi
    return luminance_Adjusted