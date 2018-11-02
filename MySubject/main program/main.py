# -*- coding:utf-8 -*-
############################################################################
##                     同じディレクトリにあるファイル                     ##
############################################################################
import cv2
import os
import numpy as np
import time

from division import *
from guidedfilter import *
from brightChannel import *
from retinex import *
from gradient_fusion import *
from fusion import *

def main(imgName, iteration, dirNameF, dirNameR, dirNameL):
    img = cv2.imread("testdata/BMP/0" + imgName + ".bmp")
    img = img.astype(dtype = np.uint8)
    ############################################################################
    ##                           画像サイズ、配列定義                         ##
    ############################################################################
    H, W = img.shape[0], img.shape[1]
    result = np.zeros((H, W, 3), np.uint8)

    start = time.time()
    ############################################################################
    ##                           照明成分の初期化                             ##
    ############################################################################
    print('----Initial Luminance----')
    luminance = cv2.GaussianBlur(img, (7, 7), 2.0)
    img = img.astype(dtype=np.float32)
    luminance = luminance.astype(dtype=np.float32)

    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    ############################################################################
    ##                            Bright Channel生成                          ##
    ############################################################################
    print('----Get Bright Channel----')
    bright = getBrightChannel(img, 3)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

    ############################################################################
    ##                              Guided Filter                             ##
    ############################################################################
    print('----Guided Filter----')
    bright = guidedFilter(img, bright, 7, 0.001)
    bright = bright.astype(dtype=np.float32)

    ############################################################################
    ##                                正規化                                  ##
    ############################################################################
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(bright, bright, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)
    init_luminance= luminance.copy()

    ############################################################################
    ##                         Variational Retinex Model                      ##
    ############################################################################
    channel = len(img.shape)
    reflectance, luminance = variationalRetinex(img, init_luminance, bright, 10.0, 0.1, 0.001, iteration, channel, imgName, dirNameR, dirNameL)

    ############################################################################
    ##                       Proposal Multi Fusion                            ##
    ############################################################################
    luminance = (255 * luminance).astype(dtype=np.uint8)
    cv2.normalize(luminance, luminance, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Conv Result", reflectance)

    luminance_result = component_fusion(luminance, imgName, dirNameF)
    cv2.normalize(luminance_result, luminance_result, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

    result = (cv2.divide((255*img).astype(dtype = np.uint8), (luminance_result* 255).astype(dtype = np.uint8), scale=255).astype(dtype = np.uint8))
    cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
    elapsed_time = time.time() - start
    fout.writelines(imgName + "Speed = " + format(elapsed_time) + "[sec]\n")
    cv2.imshow("Proposal Result", result)
    cv2.waitKey()

if __name__ == '__main__':
    ############################################################################
    ##                        画像読み込み，ファイル作成                      ##
    ############################################################################
    imgName = input('Start Image Name : ')
    finName = input('Finish Image Name: ')
    dirName = input('Input Directry Name : ')
    iteration = input('Iteration : ')
    dirNameR = "result/reflectance/" + dirName + "/"
    dirNameL = "result/luminance/" + dirName + "/"
    dirNameF = "result/final_Image/prop/" + dirName + "/"
    fout = open("evaluate_data/speed_time.txt", "w")
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
        main(str(i), int(iteration), dirNameF, dirNameR, dirNameL)
        if(i == f):
            print('----Finish----')
            break
        i += 1
