# coding : utf-8

import cv2
import numpy as np

############################################################################
##                       画像全体の勾配割合を求める                       ##
############################################################################
def averageGradient(img):
    H, W = img.shape[0], img.shape[1]
    # 総和
    sum = 0.0
    for y in range(0, H-2):
        for x in  range(0, W-2):
            A = np.square((img[y][x] - img[y+1][x]))
            B = np.square((img[y][x] - img[y][x+1]))
            sum += np.sqrt((A + B) / 2.0)
    return sum / ((H-1) * (W-1))

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

    ############################################################################
    ##                             照明成分調整                               ##
    ############################################################################
    """""""""
    print('----Luminance Adjustment----')
    luminance = cv2.cvtColor(luminance, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(reflectance, cv2.COLOR_BGR2GRAY)
    guidedImg = guidedFilter(gray, gray, 7, 0.001)
    cv2.imshow('luminance(VF)', luminance)
    cv2.imshow('luminance(GF)', guidedImg)
    cv2.waitKey()
    luminance_final = gradientFusion(guidedImg, luminance)
    cv2.normalize(luminance_final, luminance_final, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("luminance_final", luminance_final)
    cv2.waitKey()
    """""""""""
