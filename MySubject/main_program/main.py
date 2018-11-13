# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import time

############################################################################
##                     同じディレクトリにあるファイル                     ##
############################################################################
from variational_retinex import *
from multi_fusion import *


def main(imgName, dirNameF, dirNameR, dirNameL):
    img = cv2.imread("testdata/BMP/0" + imgName + ".bmp")
    img = img.astype(dtype = np.uint8)
    ############################################################################
    ##                           画像サイズ、配列定義                         ##
    ############################################################################
    start = time.time()
    ############################################################################
    ##                               BGR → HSV                               ##
    ############################################################################
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ############################################################################
    ##                         Variational Retinex Model                      ##
    ############################################################################
    channel = len(v.shape)
    v_reflectance, luminance = variationalRetinex(v, 1000.0, 0.01, 0.1, imgName, dirNameR, dirNameL)
    cv2.imshow("luminance", luminance.astype(dtype = np.uint8))
    hsv_reflectance = cv2.merge((h, s, (255.0 * v_reflectance).astype(dtype = np.uint8)))
    reflectance = cv2.cvtColor(hsv_reflectance, cv2.COLOR_HSV2BGR)
    elapsed_time = time.time() - start
    print("speed : ", elapsed_time)
    cv2.imshow("reflectance", reflectance)
    cv2.waitKey()
    #cv2.imwrite(dirNameR + "0" + str(imgName) + ".bmp", (255 * reflectance).astype(dtype = np.uint8))
    #cv2.imwrite(dirNameL + "0" + str(imgName) + ".bmp", (255 * luminance).astype(dtype = np.uint8))
    ############################################################################
    ##                       Proposal Multi Fusion                            ##
    ############################################################################
    """""""""
    luminance_result = component_fusion(luminance, imgName, dirNameF)
    cv2.normalize(luminance_result, luminance_result, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

    result = (cv2.divide((255*img).astype(dtype = np.uint8), (luminance_result* 255).astype(dtype = np.uint8), scale=255).astype(dtype = np.uint8))
    cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
    #elapsed_time = time.time() - start
    #fout.writelines(imgName + "Speed = " + format(elapsed_time) + "[sec]\n")
    cv2.imshow("Proposal Result", result)
    cv2.waitKey()
    """""""""

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
    fout = open("speed_time.txt", "w")
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
