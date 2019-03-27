# coding : utf-8

import cv2
import numpy as np

############################################################################
##                       画像全体の勾配割合を求める                       ##
############################################################################
def averageGradient(img):
    H, W = img.shape[0], img.shape[1]
    # カーネル(縦横の輪郭検出)
    kernelX = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    kernelY = kernelX.T
    aveg = np.sum(np.square(cv2.filter2D(img, cv2.CV_32F, kernelX) ** 2 + cv2.filter2D(img, cv2.CV_32F, kernelY) ** 2)) / (((H-1) * (W-1)))
    return aveg

############################################################################
##                       それぞれのLを合成する関数                        ##
############################################################################
def gradientFusion(luminance_GF, luminance_VF):
    # それぞれのLへの重み係数
    weightGF = averageGradient(luminance_GF) / (averageGradient(luminance_GF) + averageGradient(luminance_VF))
    weightVF = averageGradient(luminance_VF) / (averageGradient(luminance_GF) + averageGradient(luminance_VF))

    print('weightGF : ', weightGF)
    print('weightVF : ', weightVF)

    # 融合式
    luminance_Final = luminance_VF * weightVF + luminance_GF * weightGF
    return luminance_Final