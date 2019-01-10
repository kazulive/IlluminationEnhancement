import cv2
import numpy as np

from createGaussianPyr import *
from createDogPyr import *
from guidedfilter import *
from padding import *
from gradient_fusion import *

# カーネル(縦横の輪郭検出)
kernelX = np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]])
kernelY = kernelX.T

def shrink(R, r, bx, by, lam):
    tmpX = R * cv2.filter2D(r.astype(dtype=np.float32), cv2.CV_32F, kernelX) + bx
    tmpY = R * cv2.filter2D(r.astype(dtype=np.float32), cv2.CV_32F, kernelY) + by

    return (cv2.divide(tmpX , np.abs(tmpX))) * np.maximum(np.abs(tmpX) - 1.0 / (2.0 * lam), 0.0), (cv2.divide(tmpY, np.abs(tmpY))) * np.maximum(np.abs(tmpY) - 1.0 / (2.0 * lam), 0.0)

############################################################################
##                       Variational Retinex Model                        ##
############################################################################
def variationalRetinex(img, alpha, beta, lam, imgName, dirNameR, dirNameL):
    H, W = img.shape[:2]
    img = img.astype(dtype = np.uint8)
    Fx = psf2otf(np.expand_dims(np.array([1, -1]), axis=1), img.shape).conjugate()
    Fy = psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T, img.shape).conjugate()
    # s, r, l, R, L, d, bの初期設定
    reflectance = np.zeros((H, W), np.float32)
    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)
    init_luminance = cv2.GaussianBlur(img, (5, 5), 2.0)
    luminance = np.copy(init_luminance)
    log_image = np.log(img + 1.0)# * (255. / np.log(256))
    log_reflectance = np.log(reflectance + 1.0)# * (255. / np.log(256))
    log_luminance = np.log(luminance + 1.0)# * (255. / np.log(256))
    count = 0
    ############################################################################
    ##                           最適化問題の反復試行                         ##
    ############################################################################
    flag = 0
    while (flag != 1):
        count += 1
        if count == 1:
            # I / Lの計算 その後、分割
            log_reflectance = np.real(np.fft.ifft2(np.fft.fft2(log_image - log_luminance)))
            log_reflectance = np.minimum(log_reflectance, 0.0)
            bx = reflectance * cv2.filter2D(log_reflectance.astype(dtype = np.float32), cv2.CV_32F, kernelX) - dx
            by = reflectance * cv2.filter2D(log_reflectance.astype(dtype = np.float32), cv2.CV_32F, kernelY) - dy

            log_luminance = np.real(np.fft.ifft2((np.fft.fft2(log_image - log_reflectance)) / getKernel(img)[0] + beta * luminance * getKernel(img)[1]))
            log_luminance = np.maximum(log_image, log_luminance)

            reflectance_prev = np.copy(np.exp(log_reflectance))
            luminance_prev = np.copy(np.exp(log_luminance))
            log_reflectance_prev = np.copy(log_reflectance)
            log_luminance_prev = np.copy(log_luminance)
            bx_prev = np.copy(bx)
            by_prev = np.copy(by)
        else:
            ############################################################################
            ##                        微分オペレータのフーリエ                        ##
            ############################################################################
            sumR = getKernel(img)[0] + alpha * lam * reflectance_prev * getKernel(img)[1]
            sumL = getKernel(img)[0] + beta * luminance_prev * getKernel(img)[1]

            dx, dy = shrink(reflectance_prev, log_reflectance, bx_prev, by_prev, lam)
            phi = Fx * np.fft.fft2(dx - bx_prev) + Fy * np.fft.fft2(dy - by_prev)
            log_reflectance = np.real(np.fft.ifft2((np.fft.fft2(log_image - log_luminance_prev) + alpha * lam * phi) / sumR))
            log_reflectance = np.minimum(log_reflectance, 0.0)
            #np.savetxt("reflectance" + str(count) + ".csv", log_reflectance, fmt="%0.2f", delimiter=",")

            bx = bx_prev + reflectance * cv2.filter2D(log_reflectance.astype(dtype = np.float32), cv2.CV_32F, kernelX) - dx
            by = by_prev + reflectance *  cv2.filter2D(log_reflectance.astype(dtype = np.float32), cv2.CV_32F, kernelY) - dy

            log_luminance = np.real(np.fft.ifft2((np.fft.fft2(log_image - log_reflectance)) / sumL))
            log_luminance = np.maximum(log_image, log_luminance)
            #np.savetxt("luminance" + str(count) + ".csv", log_luminance, fmt="%0.2f", delimiter=",")

            eps_r = cv2.divide(np.abs(np.sum(log_reflectance) - np.sum(log_reflectance_prev)),np.abs(np.sum(log_reflectance_prev)))
            eps_l = cv2.divide(np.abs(np.sum(log_luminance) - np.sum(log_luminance_prev)), np.abs(np.sum(log_luminance_prev)))
            reflectance_prev = np.copy(np.exp(log_reflectance))
            luminance_prev = np.copy(np.exp(log_luminance))
            log_reflectance_prev = np.copy(log_reflectance)
            log_luminance_prev = np.copy(log_luminance)
            bx_prev = np.copy(bx)
            by_prev = np.copy(by)

            if (eps_r[0]<= 0.01 and eps_l[0] <= 0.01):
                flag = 1

    return np.exp(log_reflectance), np.exp(log_luminance)