# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import time

############################################################################
##                     同じディレクトリにあるファイル                     ##
############################################################################
from guidedfilter import *
from bright_channel import *
from variational_retinex import *
from gradient_fusion import *
from multi_fusion import *

def main(imgName, dirNameF, dirNameR, dirNameL):
    img = cv2.imread("testdata/BMP/0" + imgName + ".bmp")
    img = img.astype(dtype = np.uint8)
    ############################################################################
    ##                           画像サイズ、配列定義                         ##
    ############################################################################
    H, W = img.shape[0], img.shape[1]
    #start = time.time()
    ############################################################################
    ##                               BGR → HSV                               ##
    ############################################################################
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ############################################################################
    ##                           照明成分の初期化                             ##
    ############################################################################
    print('----Initial Luminance----')
    #luminance = cv2.GaussianBlur(img, (7, 7), 2.0)
    luminance = cv2.GaussianBlur(v, (9, 9), 5.0)
    v = v.astype(dtype = np.float32)
    img = img.astype(dtype=np.float32)
    luminance = luminance.astype(dtype=np.float32)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(v, v, 0, 1, cv2.NORM_MINMAX)
    ############################################################################
    ##                            Bright Channel生成                          ##
    ############################################################################
    """""""""
    print('----Get Bright Channel----')
    bright = getBrightChannel(img, 3)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    ############################################################################
    ##                              Guided Filter                             ##
    ############################################################################
    print('----Guided Filter----')
    bright = guidedFilter(img, bright, 7, 0.001)
    bright = bright.astype(dtype=np.float32)
    """""""""
    ############################################################################
    ##                                正規化                                  ##
    ############################################################################
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    #cv2.normalize(bright, bright, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)
    init_luminance= luminance.copy()
    ############################################################################
    ##                         Variational Retinex Model                      ##
    ############################################################################
    #channel = len(img.shape)
    channel = len(v.shape)
    v_reflectance, luminance = variationalRetinex(v, init_luminance,init_luminance, 10.0, 0.01, 0.0001, channel, imgName, dirNameR, dirNameL)
    cv2.imshow("Conv Luminance", (255 * luminance).astype(dtype = np.uint8))
    hsv_reflectance = cv2.merge((h, s, (255 * v_reflectance).astype(dtype = np.uint8)))
    reflectance = cv2.cvtColor(hsv_reflectance, cv2.COLOR_HSV2BGR)
    cv2.imshow("Conv Result", reflectance.astype(dtype = np.uint8))
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
