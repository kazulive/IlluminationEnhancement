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
from clahe import *
from unsharpmask import *
from shrink import *
from fusion import *

def main(imgName, iteration, dirNameF, dirNameR, dirNameL):

    img = cv2.imread("testdata/0" + imgName + ".bmp")
    img = img.astype(dtype = np.uint8)

    # グレー変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HSV変換
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # H, S, V分割
    h, s, v = cv2.split(hsvImg)

    v = v.astype(dtype = np.float32)

    ############################################################################
    ##                           画像サイズ、配列定義                         ##
    ############################################################################
    H, W = img.shape[0], img.shape[1]
    #reflectance = np.zeros((H, W, 3), np.float32)
    #luminance = np.zeros((H, W, 3), np.float32)
    result = np.zeros((H, W, 3), np.uint8)

    start = time.time()
    ############################################################################
    ##                           照明成分の初期化                             ##
    ############################################################################
    print('----Initial Luminance----')
    luminance = cv2.GaussianBlur(img, (7, 7), 2.0)
    #cv2.imwrite("result/luminance.bmp", luminance)
    img = img.astype(dtype=np.float32)
    luminance = luminance.astype(dtype=np.float32)

    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    ############################################################################
    ##                            Bright Channel生成                          ##
    ############################################################################
    print('----Get Bright Channel----')
    bright = getBrightChannel(img, 3)
    # cv2.imwrite("result/bright.bmp",(255 * bright).astype(dtype=np.uint8))
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
    cv2.normalize(v, v, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)

    #cv2.imwrite(dirNameR + "bright0" + imgName +".bmp", (255 * bright).astype(dtype=np.uint8))
    #cv2.imwrite(dirNameR + "init_luminance0" + imgName + ".bmp", (255 * luminance).astype(dtype=np.uint8))
    init_luminance= luminance.copy()
    ############################################################################
    ##                         Variational Retinex Model                      ##
    ############################################################################
    channel = len(img.shape)
    reflectance, luminance = variationalRetinex(img, init_luminance, bright, 10.0, 0.1, 0.001, iteration, channel, imgName, dirNameR, dirNameL)
    #cv2.imshow("luminance", (255*luminance).astype(dtype = np.uint8))
    #cv2.imshow("reflectance", cv2.divide((255*img).astype(dtype = np.uint8), (255*luminance).astype(dtype = np.uint8), scale=255).astype(dtype = np.uint8))
    #cv2.waitKey()
    #cv2.imwrite(dirNameR + "reflectance0" + imgName + "_iteration_" + str(i) + ".bmp", cv2.divide((255*img).astype(dtype = np.uint8), (255*luminance).astype(dtype = np.uint8), scale=255).astype(dtype = np.uint8))
    #cv2.imwrite(dirNameL + "luminance0" + imgName + "_iteration_" + str(i) + ".bmp", (luminance*255).astype(dtype=np.uint8))
    """""""""
    #result = cv2.merge((h, s, (reflectance*255).astype(dtype = np.uint8)))#v_new.astype(dtype=np.uint8)))#
    #result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    #cv2.imwrite(dirNameF + "result0" + imgName + "_iteration_" + str(i) + ".bmp", result.astype(dtype=np.uint8))
    #cv2.imshow("reflectance", reflectance)
    #cv2.imshow("luminance", luminance)
    #cv2.waitKey()
    """"""""

    print('----Program End----')
    """""""""
    ############################################################################
    ##                         NonLinearStretch関数                           ##
    ############################################################################
    #luminance = (255 * luminance).astype(dtype=np.uint8)
    #cv2.normalize(luminance, luminance, 0, 255, cv2.NORM_MINMAX)

    lb, lg, lr = cv2.split(luminance)
    lb_adjusted = nonLinearStretch(lb)
    lg_adjusted = nonLinearStretch(lg)
    lr_adjusted = nonLinearStretch(lr)

    cv2.normalize(lb_adjusted, lb_adjusted, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(lg_adjusted, lg_adjusted, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(lr_adjusted, lr_adjusted, 0, 1, cv2.NORM_MINMAX)

    luminance_adjusted = cv2.merge((lb, lg, lr))
    #cv2.imwrite(dirNameF + "nonLinearStretch0" + imgName + ".bmp", (255 * luminance_adjusted).astype(dtype = np.uint8))
    #cv2.imshow('Non Linear Stretch',(255 * luminance_adjusted).astype(dtype = np.uint8))
    #cv2.waitKey()
    ############################################################################
    ##                              CLAHE関数                                 ##
    ############################################################################
    luminance = (255 * luminance).astype(dtype=np.uint8)
    cv2.normalize(luminance, luminance, 0, 255, cv2.NORM_MINMAX)
    lb, lg, lr = cv2.split(luminance)
    print('----CLAHE----')
    lb_clahe = cleary(lb)
    lg_clahe = cleary(lg)
    lr_clahe = cleary(lr)

    cv2.normalize(lb_clahe, lb_clahe, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lg_clahe, lg_clahe, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lr_clahe, lr_clahe, 0, 255, cv2.NORM_MINMAX)

    luminance_clahe = cv2.merge((lb_clahe, lg_clahe, lr_clahe))
    #cv2.imwrite(dirNameF + "CLAHE0" + imgName + ".bmp", luminance_clahe)
    #cv2.imshow('CLAHE', luminance_clahe.astype(dtype = np.uint8))
    #cv2.waitKey()
    ############################################################################
    ##                            Unsharp Masking                             ##
    ############################################################################
    #luminance = luminance.astype(dtype = np.uint8)
    #cv2.normalize(luminance, luminance, 0, 255, cv2.NORM_MINMAX)
    lb, lg, lr = cv2.split(luminance)
    print('----Unsharp Mask----')
    lb_unsharp = unsharp_mask(lb)
    lg_unsharp = unsharp_mask(lg)
    lr_unsharp = unsharp_mask(lr)

    cv2.normalize(lb_unsharp, lb_unsharp, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lg_unsharp, lg_unsharp, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lr_unsharp, lr_unsharp, 0, 255, cv2.NORM_MINMAX)

    luminance_unsharp = cv2.merge((lb_unsharp, lg_unsharp, lr_unsharp))
    #cv2.imwrite(dirNameF + "Unsharp_Mask0" + imgName + ".bmp", luminance_unsharp)
    #cv2.imshow('Unsharp Mask', luminance_unsharp.astype(dtype = np.uint8))
    #cv2.waitKey()
    ############################################################################
    ##                        Fusion of three components                      ##
    ############################################################################
    luminance = luminance.copy().astype(dtype = np.float32)
    luminance_adjusted = luminance_adjusted.copy().astype(dtype = np.float32)
    luminance_clahe = luminance_clahe.copy().astype(dtype=np.float32)
    luminance_unsharp = luminance_unsharp.copy().astype(dtype=np.float32)

    lb, lg, lr = cv2.split(luminance)
    lb_adjusted, lg_adjusted, lr_adjusted = cv2.split(luminance_adjusted)
    lb_clahe, lg_clahe, lr_clahe = cv2.split(luminance_clahe)
    lb_unsharp, lg_unsharp, lr_unsharp = cv2.split(luminance_unsharp)

    b_result = component_fusion(lb, (255*lb_adjusted), lb_unsharp, lb_clahe)
    g_result = component_fusion(lg, (255*lg_adjusted), lg_unsharp, lg_clahe)
    r_result = component_fusion(lr, (255 * lr_adjusted), lr_unsharp, lr_clahe)

    cv2.normalize(b_result, b_result, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(g_result, g_result, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(r_result, r_result, 0, 255, cv2.NORM_MINMAX)

    luminance_result = cv2.merge((b_result, g_result, r_result))
    #cv2.imshow('B', b_result.astype(dtype=np.uint8))
    #cv2.imshow('G', g_result.astype(dtype=np.uint8))
    #cv2.imshow('R', r_result.astype(dtype=np.uint8))
    #cv2.waitKey()

    cv2.normalize(luminance_result, luminance_result, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

    result = (cv2.divide((255*img).astype(dtype = np.uint8), (luminance_result* 255).astype(dtype = np.uint8), scale=255).astype(dtype = np.uint8))
    cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
    #cv2.imwrite(dirNameF + "0" + imgName + ".bmp", result.astype(dtype = np.uint8))
    #cv2.imshow('Result', result)
    #cv2.waitKey()
    elapsed_time = time.time() - start
    fout.writelines(imgName + "Speed = " + format(elapsed_time) + "[sec]\n")

if __name__ == '__main__':
    imgName = input('input image name : ')
    dirName = input('input directry Name : ')
    iteration = input('Iteration : ')
    dirNameR = "result/reflectance/" + dirName + "/"
    dirNameL = "result/luminance/" + dirName + "/"
    dirNameF = "result/final_Image/prop/" + dirName + "/"
    fout = open("evaluate_data/speed_time.txt", "w")
    #os.mkdir(dirNameR)
    #os.mkdir(dirNameL)
    #os.mkdir(dirNameF)
    i = int(imgName)
    while(True):
        print('----Input 0' + str(i) + '.bmp-----')
        main(str(i), int(iteration), dirNameF, dirNameR, dirNameL)
        if(i == 5):
            print('----Finish----')
            break
        i += 1

"""""""""
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
"""""""""""
