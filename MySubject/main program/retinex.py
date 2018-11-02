"""
Variational Retinex Model関数
エネルギー関数の最小化問題
"""
import cv2
import numpy as np
from division import *

############################################################################
##                       各チャネルのFFT計算                              ##
############################################################################
def culcFFT(img, sum):
    fimg = np.fft.fft2(img)
    fimg = fimg / sum
    ifimg = np.fft.ifft2(fimg)
    ifimg = ifimg.real
    return ifimg

def variationalRetinex(img, luminance0, bright, alpha, beta, gamma, number, channel, imgName, dirNameR, dirNameL):
    # 3チャネル用処理
    if(channel == 3):
        print('----Variational Retinex Model(3 channel)----')
        ############################################################################
        ##                           各処理の前準備                               ##
        ############################################################################
        count = 0
        # BGR分割
        b, g, r = cv2.split(img)
        # 画像サイズ
        H, W = img.shape[:2]
        # 照明成分の初期化
        luminance = luminance0.copy()
        reflectance = np.zeros((H, W, 3), np.float32)

        ############################################################################
        ##                           デルタ関数定義                               ##
        ############################################################################
        delta = np.ones((H, W), np.float32)
        gdelta = delta + gamma
        fdimage = delta
        fgdimage = gdelta

        ############################################################################
        ##                           FFT用配列                                    ##
        ############################################################################
        flimage = np.zeros((H, W, 3), np.float32)
        iflimage = np.zeros((H, W, 3), np.float32)
        frimage = np.zeros((H, W, 3), np.float32)
        ifrimage = np.zeros((H, W, 3), np.float32)

        ############################################################################
        ##                           微分オペレータ                               ##
        ############################################################################
        sobel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])

        ############################################################################
        ##                           F(dx), F(dy)                                 ##
        ############################################################################
        fsximage = np.fft.fft2(sobel_x)
        fsyimage = np.fft.fft2(sobel_y)
        fsx_shift = np.fft.fftshift(fsximage)
        fsy_shift = np.fft.fftshift(fsyimage)
        fsx_mag_spectrum = np.abs(fsx_shift) / np.amax(np.abs(fsx_shift))
        fsy_mag_spectrum = np.abs(fsy_shift) / np.amax(np.abs(fsy_shift))
        fsx_mag_spectrum = cv2.resize(fsx_mag_spectrum, (W, H))
        fsy_mag_spectrum = cv2.resize(fsy_mag_spectrum, (W, H))

        ############################################################################
        ##                           R, Lの計算時の分母　                         ##
        ############################################################################
        sum = np.square(fsx_mag_spectrum) + np.square(fsy_mag_spectrum)
        sumR = fdimage + beta * sum
        sumL = fgdimage + alpha * sum

        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        while (count < number):
            count += 1

            # I / Lの計算 その後、分割
            IL = cv2.divide((img * 255).astype(dtype=np.float32), (255 * luminance).astype(dtype=np.float32))
            ILB, ILG, ILR = cv2.split(IL)

            reflectance = cv2.merge((culcFFT(ILB, sumR), culcFFT(ILG, sumR), culcFFT(ILR, sumR)))
            cv2.normalize(reflectance, reflectance, 0, 1, cv2.NORM_MINMAX)

            IR = cv2.divide((img * 255).astype(dtype=np.float32), (255 * reflectance).astype(dtype=np.float32))
            IRB, IRG, IRR = cv2.split(IR)
            IRB += gamma * bright
            IRG += gamma * bright
            IRR += gamma * bright

            luminance = cv2.merge((culcFFT(IRB, sumL), culcFFT(IRG, sumL), culcFFT(IRR, sumL)))
            cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)

            lb, lg, lr = cv2.split(luminance)
            maxb = np.maximum(lb, b)
            maxg = np.maximum(lg, g)
            maxr = np.maximum(lr, r)
            luminance = cv2.merge((maxb, maxg, maxr))

            #cv2.imwrite(dirNameR + "reflectance0" + imgName + "_iteration_" + str(count) + ".bmp", cv2.divide((255*img).astype(dtype = np.uint8), (255*luminance).astype(dtype = np.uint8), scale=255).astype(dtype = np.uint8))
            #cv2.imwrite(dirNameL + "luminance0" + imgName + "_iteration_" + str(count) + ".bmp", (luminance*255).astype(dtype=np.uint8))

    # 1チャネル用処理
    else:
        print('----Variational Retinex Model(1 channel)----')
        ############################################################################
        ##                           各処理の前準備                               ##
        ############################################################################
        count = 0
        # 画像サイズ
        H, W = img.shape[:2]
        # 照明成分の初期化
        luminance = luminance0.copy()
        reflectance = np.zeros((H, W), np.float32)

        ############################################################################
        ##                           デルタ関数定義                               ##
        ############################################################################
        delta = np.ones((H, W), np.float32)
        gdelta = delta + gamma
        fdimage = delta
        fgdimage = gdelta

        ############################################################################
        ##                           FFT用配列                                    ##
        ############################################################################
        flimage = np.zeros((H, W), np.float32)
        iflimage = np.zeros((H, W), np.float32)
        frimage = np.zeros((H, W), np.float32)
        ifrimage = np.zeros((H, W), np.float32)

        ############################################################################
        ##                           微分オペレータ                               ##
        ############################################################################
        sobel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])

        ############################################################################
        ##                           F(dx), F(dy)                                 ##
        ############################################################################
        fsximage = np.fft.fft2(sobel_x)
        fsyimage = np.fft.fft2(sobel_y)
        fsx_shift = np.fft.fftshift(fsximage)
        fsy_shift = np.fft.fftshift(fsyimage)
        fsx_mag_spectrum = np.abs(fsx_shift) / np.amax(np.abs(fsx_shift))
        fsy_mag_spectrum = np.abs(fsy_shift) / np.amax(np.abs(fsy_shift))
        fsx_mag_spectrum = cv2.resize(fsx_mag_spectrum, (W, H))
        fsy_mag_spectrum = cv2.resize(fsy_mag_spectrum, (W, H))

        ############################################################################
        ##                           R, Lの計算時の分母　                         ##
        ############################################################################
        sum = cv2.add(np.square(fsx_mag_spectrum), np.square(fsy_mag_spectrum))
        sumR = cv2.add(fdimage , beta * sum)
        sumL = cv2.add(fgdimage, alpha * sum)

        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        while (count < number):
            count += 1

            # I / Lの計算 その後、分割
            IL = division((img * 255).astype(dtype=np.float32), (255 * luminance).astype(dtype=np.float32))
            #IL = division(img.astype(dtype=np.float32), luminance.astype(dtype=np.float32))

            reflectance = culcFFT(IL, sumR)
            reflectance = reflectance.copy()
            cv2.normalize(reflectance, reflectance, 0, 1, cv2.NORM_MINMAX)
            #cv2.imwrite("result/reflectance" + str(count) + ".jpg", (255 * reflectance).astype(dtype=np.uint8))

            IR = division((img * 255).astype(dtype=np.float32), (255 * reflectance).astype(dtype=np.float32))
            #IR = division(img.astype(dtype=np.float32), reflectance.astype(dtype=np.float32))
            IR += gamma * bright

            luminance = culcFFT(IR, sumL)
            luminance = luminance.copy()
            cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)
            luminance = np.maximum(luminance, img)

            #cv2.imshow("reflectance", (255 * reflectance).astype(dtype=np.uint8))
            #cv2.imshow("luminance", luminance.astype(dtype = np.uint8))
            #cv2.waitKey()

    return reflectance, luminance