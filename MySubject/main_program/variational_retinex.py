import cv2
import numpy as np

from createGaussianPyr import *
from guidedfilter import *
from padding import *
from gradient_fusion import *

############################################################################
##                       各チャネルのFFT計算                              ##
############################################################################
def culcFFT(img, sum):
    fimg = np.fft.fft2(img) / sum
    real = np.real(np.fft.ifft2(fimg))
    return real

def upSampling(img, luminance, init_luminance):
    up_luminance = cv2.pyrUp(luminance, (img.shape))

    up_luminance = cv2.resize(up_luminance, (img.shape[1], img.shape[0]))
    up_init_luminance = np.copy(up_luminance)

    return up_luminance, up_init_luminance

############################################################################
##                       Variational Retinex Model                        ##
############################################################################
def variationalRetinex(img, alpha, beta, gamma, imgName, dirNameR, dirNameL, pyr_num):
    H, W = img.shape[:2]
    init_luminance = cv2.GaussianBlur(img, (5, 5), 5.0)
    luminance = np.copy(init_luminance)
    img = img.astype(dtype=np.float32)
    luminance = luminance.astype(dtype=np.float32)
    reflectance = np.zeros((H, W), np.float32)
    ############################################################################
    ##                           各処理の前準備                               ##
    ############################################################################
    count = 0
    ############################################################################
    ##                           デルタ関数定義                               ##
    ############################################################################
    # delta = np.ones((H, W), np.float32)
    # gdelta = delta + gamma
    ############################################################################
    ##                           微分オペレータ                               ##
    ############################################################################
    sumR = getKernel(img, gamma)[0] + beta * getKernel(img, gamma)[2]
    sumL = getKernel(img, gamma)[1] + alpha * getKernel(img, gamma)[2]
    ############################################################################
    ##                           最適化問題の反復試行                         ##
    ############################################################################
    flag = 0
    while (flag != 1):
        count += 1
        reflectance_prev = np.copy(reflectance)
        luminance_prev = np.copy(luminance)
        # I / Lの計算
        IL = cv2.divide(img, (luminance).astype(dtype=np.float32))
        reflectance = culcFFT(IL, sumR)
        reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))
        # I / Rの計算
        IR = cv2.divide(img, (reflectance).astype(dtype=np.float32))
        IR += gamma * init_luminance
        luminance = culcFFT(IR, sumL)
        luminance = np.maximum(luminance, img)

        #cv2.imwrite(dirNameR + "0" + str(imgName) + str(count) + ".bmp", (255.0 * reflectance).astype(dtype=np.uint8))
        #cv2.imwrite(dirNameL + "0" + str(imgName) + str(count) + ".bmp", (luminance).astype(dtype=np.uint8))

        if (count != 1):
            eps_r = cv2.divide(np.abs(np.sum(255.0 * reflectance) - np.sum(255.0 * reflectance_prev)),
                               np.abs(np.sum(255.0 * reflectance_prev)))
            eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
            if (eps_r[0] <= 0.05 and eps_l[0] <= 0.05):
                flag = 1

    return reflectance, luminance

    """""""""
    imgPyr = createGaussianPyr(image, pyr_num)
    for i in range(pyr_num - 1, -1, -1):
        img = np.copy(imgPyr[i])
        if (i == pyr_num - 1):
            H, W = img.shape[:2]
            img = img.astype(dtype=np.float32)
            reflectance = np.zeros((H, W), np.float32)
            init_luminance = cv2.GaussianBlur(img, (5, 5), 2.0)
            luminance = np.copy(init_luminance)
            print('----Variational Retinex Model(1 channel)----')
        else:
            H, W = img.shape[:2]
            reflectance = np.zeros((H, W), dtype=np.float32)
            luminance, init_luminance = upSampling(img, luminance, init_luminance)
        count = 0
        ############################################################################
        ##                        微分オペレータのフーリエ                        ##
        ############################################################################
        sumR = getKernel(img, gamma)[0] + beta * getKernel(img, gamma)[2]
        sumL = getKernel(img, gamma)[1] + alpha * getKernel(img, gamma)[2]
        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        flag = 0
        while (flag != 1):
            count += 1
            reflectance_prev = np.copy(reflectance)
            luminance_prev = np.copy(luminance)
            # I / Lの計算 その後、分割
            IL = cv2.divide((img).astype(dtype=np.float32), (luminance).astype(dtype=np.float32))
            reflectance = culcFFT(IL, sumR)
            reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))
            IR = cv2.divide((img).astype(dtype=np.float32), (reflectance).astype(dtype=np.float32))
            #np.savetxt("luminance" + str(i) + str(count) + ".csv", reflectance, fmt="%0.2f", delimiter=",")
            IR += gamma * init_luminance

            luminance = culcFFT(IR, sumL)
            luminance = np.maximum(luminance, img)

            #if(i==0):
            #    cv2.imwrite(dirNameR + "0" + str(imgName) + str(count) + ".bmp",
            #            (255.0 * reflectance).astype(dtype=np.uint8))
            #cv2.imwrite(dirNameL + str(i) + "_0" + str(imgName) + str(count) + ".bmp", (luminance).astype(dtype=np.uint8))

            if (count != 1):
                eps_r = cv2.divide(np.abs(np.sum(255 * reflectance) - np.sum(255 * reflectance_prev)),
                                   np.abs(np.sum(255 * reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
                if (eps_r[0] <= 0.1 and eps_l[0] <= 0.1):
                    flag = 1
    return reflectance, luminance
    """""""""