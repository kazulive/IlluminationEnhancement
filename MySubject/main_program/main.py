# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import time

"""
time関数
start = time.time()
elapsed_time = time.time() - start
fout.writelines(imgName + ":Speed = " + format(elapsed_time) + "[sec]\n")
"""

"""
画像保存
cv2.imwrite(dirNameF + "0" + str(imgName) + "_shrink.bmp", (nonLinearStretch(luminance)).astype(dtype=np.uint8))
cv2.imwrite(dirNameF + "0" + str(imgName) + "_contrast.bmp", (cleary(nonLinearStretch(luminance).astype(dtype = np.uint8))).astype(dtype=np.uint8))
cv2.imwrite(dirNameF + "0" + str(imgName) + ".bmp", (reflectance).astype(dtype=np.uint8))
cv2.imshow("Luminance", (luminance).astype(dtype=np.uint8))
cv2.imshow("Reflectance", (255.0 * v_reflectance).astype(dtype=np.uint8))
cv2.imshow("Conv Result", (reflectance).astype(dtype=np.uint8))
cv2.waitKey()

cv2.imwrite(dirNameF + "0" + str(imgName) + ".bmp", (guidedImg).astype(dtype=np.uint8))

cv2.imwrite(dirNameR + "0" + str(imgName) + ".bmp", (reflectance).astype(dtype = np.uint8))
cv2.imwrite(dirNameL + "0" + str(imgName) + ".bmp", (luminance).astype(dtype = np.uint8))

cv2.imwrite(dirNameF + "reflectance0" + str(imgName) + ".bmp", (reflectance_final).astype(dtype=np.uint8))
"""


############################################################################
##                     同じディレクトリにあるファイル                     ##
############################################################################
from variational_retinex import *
from shrink import *
from guidedfilter import *
from gradient_fusion import *
from clahe import *


def main(imgName, dirNameF, dirNameR, dirNameL):
    img = cv2.imread("testdata/BMP/0" + imgName + ".bmp")
    img = img.astype(dtype = np.uint8)
    ############################################################################
    ##                               BGR → HSV                               ##
    ############################################################################
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ############################################################################
    ##                         Variational Retinex Model                      ##
    ############################################################################
    channel = len(v.shape)
    v_reflectance, luminance = variationalRetinex(v, 10.0, 0.1, 0.001, imgName, dirNameR, dirNameL, pyr_num=4)
    hsv_reflectance = cv2.merge((h, s, (cleary(nonLinearStretch(luminance).astype(dtype = np.uint8)) * v_reflectance).astype(dtype=np.uint8)))
    reflectance = cv2.cvtColor(hsv_reflectance, cv2.COLOR_HSV2BGR)
    cv2.imwrite(dirNameF + "0" + str(imgName) + ".bmp", (reflectance).astype(dtype=np.uint8))
    ############################################################################
    ##                         Guided Fusion                                  ##
    ############################################################################
    guidedImg = (guidedFilter(v.astype(dtype=np.float32) / 255.0, v.astype(dtype=np.float32)/255.0, 7, 0.04) * 255.0).astype(dtype=np.uint8)
    cv2.imshow("guided", guidedImg)
    cv2.waitKey()
    luminance_final = gradientFusion(guidedImg.astype(dtype=np.float32), luminance.astype(dtype=np.float32))
    reflectance_new = cv2.divide((v).astype(dtype = np.float32), (luminance_final).astype(dtype = np.float32))
    reflectance_new = np.minimum(1.0, np.maximum(reflectance_new, 0.0))
    hsv_reflectance_new = cv2.merge((h, s, (nonLinearStretch(luminance_final) * reflectance_new).astype(dtype=np.uint8)))
    reflectance_final = cv2.cvtColor(hsv_reflectance_new, cv2.COLOR_HSV2BGR)
    cv2.imwrite(dirNameF + "reflectance0" + str(imgName) + ".bmp", (reflectance_final).astype(dtype=np.uint8))
    print('----Variational Retinex End----')
    cv2.waitKey()

if __name__ == '__main__':

    ############################################################################
    ##                        画像読み込み，ファイル作成                      ##
    ############################################################################

    imgName = input('Start Image Name : ')
    finName = input('Finish Image Name: ')
    dirName = input('Input Directry Name : ')
    dirNameR = "result/reflectance/" + dirName + "/"
    dirNameL = "result/luminance/" + dirName + "/"
    dirNameF = "result/proposal/" + dirName + "/"
    speedName = "result/" + dirName
    fout = open(speedName + "_speed_time.txt", "w")

    ############################################################################
    ##                        　 ディレクトリ作成   　　                      ##
    ############################################################################
    if not os.path.exists(dirNameR):
        os.mkdir(dirNameR)
    if not os.path.exists(dirNameL):
        os.mkdir(dirNameL)
    if not os.path.exists(dirNameF):
        os.mkdir(dirNameF)

    i = int(imgName)
    f = int(finName)
    while(True):
        print('----Input 0' + str(i) + '.bmp-----')
        main(str(i), dirNameF, dirNameR, dirNameL)
        if(i == f):
            print('----Finish----')
            break
        i += 1
