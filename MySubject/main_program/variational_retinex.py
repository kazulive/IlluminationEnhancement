import cv2
import numpy as np

from createGaussianPyr import *

def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape

    psf_pad = zero_pad(psf, shape, position='corner')

    for axis, axis_size in enumerate(inshape):
        psf_pad = np.roll(psf_pad, -int(axis_size / 2), axis=axis)

    otf = np.fft.fft2(psf_pad)

    return otf

def getKernel(img):
    sizeF = np.shape(img)
    diff_kernelX = np.expand_dims(np.array([-1, 1]), axis=1)
    diff_kernelY = np.expand_dims(np.array([[-1], [0], [1]]), axis=0)
    eigsDtD = np.abs(psf2otf(diff_kernelX, sizeF))   + np.abs(psf2otf(diff_kernelX.T, sizeF))
    return eigsDtD

############################################################################
##                       各チャネルのFFT計算                              ##
############################################################################
def culcFFT(img, sum):
    fimg = np.fft.fft2(img) / sum
    real = np.real(np.fft.ifft2(fimg))
    return real

############################################################################
##                       Variational Retinex Model                        ##
############################################################################
def variationalRetinex(image, alpha, beta, gamma, imgName, dirNameR, dirNameL):
    imgPyr = createGaussianPyr(image)
    for i in range(3, -1, -1):
        img = imgPyr[i].copy()
        if(i == 3):
            print('----Initial Luminance----')
            init_luminance = cv2.GaussianBlur(img, (5, 5), 5.0)
            luminance = init_luminance.copy()
            print('----Variational Retinex Model(1 channel)----')
        else:
            init_luminance = cv2.GaussianBlur(img, (5, 5), 5.0)
            luminance = cv2.pyrUp(luminance, (img.shape))
            luminance = cv2.resize(luminance, (img.shape[1], img.shape[0]))
        ############################################################################
        ##                           各処理の前準備                               ##
        ############################################################################
        count = 0
        # 画像サイズ
        H, W = img.shape[:2]
        reflectance = np.zeros((H, W), np.float32)
        ############################################################################
        ##                           デルタ関数定義                               ##
        ############################################################################
        delta = np.ones((H, W), np.float32)
        gdelta = delta + gamma
        ############################################################################
        ##                           微分オペレータ                               ##
        ############################################################################
        sumR = delta + beta * getKernel(img)
        sumL =  gdelta + alpha * getKernel(img)
        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        flag = 0
        while (flag != 1):
            count += 1
            reflectance_prev = reflectance.copy()
            luminance_prev = luminance.copy()
            # I / Lの計算 その後、分割
            IL = cv2.divide((img).astype(dtype=np.float32), (luminance).astype(dtype=np.float32))
            reflectance = culcFFT(IL, sumR)
            reflectance = reflectance.copy()
            reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))

            IR = cv2.divide((img).astype(dtype=np.float32), (reflectance).astype(dtype=np.float32))
            IR += gamma * init_luminance

            luminance = culcFFT(IR, sumL)
            luminance = luminance.copy()
            luminance = np.maximum(luminance, img)

            if (count != 1):
                eps_r = cv2.divide(np.abs(np.sum(reflectance) - np.sum(reflectance_prev)),
                                   np.abs(np.sum(reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
                if (eps_r[0] <= 0.01 and eps_l[0] <= 0.01):
                    #print('----Variational Retinex End----')
                    flag = 1

    return reflectance, luminance