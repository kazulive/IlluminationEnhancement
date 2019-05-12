import cv2
import numpy as np
import glob
import time
from natsort import natsorted

from padding import *

class LLIE(object):
    def __init__(self, img, beta, weight, delta, lam, sigma):
        H, W = img.shape[0], img.shape[1]
        self.height = H
        self.width = W
        self.beta = beta
        self.weight = weight
        self.delta = delta
        self.lam = lam
        self.sigma = sigma

        # 微分オペレータ
        self.kernel = np.array([[0, 0, 0],
                                [-1, 0, 1],
                                [0, 0, 0]])
        # 微分オペレータのFFT
        self.F_conj_h = psf2otf(np.expand_dims(np.array([1, -1]), axis=1), (self.height, self.width)).conjugate()
        self.F_conj_v = psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T, (self.height, self.width)).conjugate()
        self.F_div = getKernel(img)[1]

    # 反射画像を更新
    def get_reflectance(self, img, illumination, N_map, G):
        tmp1 = np.fft.fft2((img - N_map)/(illumination + 1.0))
        tmp2 = self.weight * (self.F_conj_h + self.F_conj_v) * np.fft.fft2(G)

        return np.real(np.fft.ifft2((tmp1 + tmp2) / (1.0 + self.weight * self.F_div)))

    # 照明画像を更新
    def get_illumination(self, img, reflectance, N_map, T, Z, meu):
        tmp1 = 2.0 * np.fft.fft2((img - N_map)/(np.maximum(reflectance, 0.001)))
        tmp2 = (self.F_conj_v + self.F_conj_h) * (meu * np.fft.fft2(T) - np.fft.fft2(Z))

        return np.real(np.fft.ifft2((tmp1 + tmp2) / (2.0 + meu * self.F_div)))

    # ノイズを更新
    def get_N_map(self, img, reflectance, illumination):
        return (img - reflectance * illumination) / (1.0 + self.delta)

    # Gを更新
    def get_G(self, img):
        grad_img = (cv2.filter2D(img.astype(dtype=np.float32), cv2.CV_32F, self.kernel) + cv2.filter2D(img.astype(dtype=np.float32), cv2.CV_32F, self.kernel.T)) / 2.0
        grad_img[np.abs(grad_img) < 2.] = 0.
        K = 1.0 + self.lam * np.exp(-np.abs(grad_img) / self.sigma)
        return K * grad_img

    # Tを更新
    def get_T(self, illumination, Z, meu):
        tmp = illumination + Z / meu
        return np.sign(tmp) * np.maximum(np.abs(tmp) - self.beta/meu, 0.)

    # Zを更新
    def get_Z(self, illumianton, T, Z, meu):
        return Z + meu * (illumianton - T)

    # main関数
    def build(self, img):
        # 配列の初期化
        reflectance = np.zeros((self.height, self.width), dtype=np.float32)
        illumination = np.copy(img)
        N_map = np.zeros((self.height, self.width), dtype=np.float32)
        G = self.get_G(img)
        T = np.zeros((self.height, self.width), dtype=np.float32)
        Z = np.zeros((self.height, self.width), dtype=np.float32)
        meu = 1.0
        p = 1.5
        k = 0

        while(k < 5):
            reflectance = self.get_reflectance(img, illumination, N_map, G)
            reflectance = reflectance.astype(dtype=np.float32)
            reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))
            cv2.imshow("Reflectance", (255 * reflectance).astype(dtype=np.uint8))
            cv2.waitKey(0)
            illumination = self.get_illumination(img, reflectance, N_map, T, Z, meu)
            illumination = illumination.astype(dtype=np.float32)
            cv2.imshow("Illumination", illumination.astype(dtype=np.uint8))
            cv2.waitKey(0)
            print(illumination)
            N_map = self.get_N_map(img, reflectance, illumination)
            grad_illumination = (cv2.filter2D(illumination.astype(dtype=np.float32), cv2.CV_32F, self.kernel) + cv2.filter2D(illumination.astype(dtype=np.float32), cv2.CV_32F, self.kernel.T)) / 2.0
            T = self.get_T(grad_illumination, Z, meu)
            Z = self.get_Z(grad_illumination, T, Z, meu)
            meu *= p
            k += 1

        return reflectance, illumination, N_map

def gamma_correction(img, gamma):
    return illumination ** (1. / gamma)

def fileRead():
    data = []
    for file in natsorted(glob.glob('./testdata/BMP/*.bmp')):
        data.append(cv2.imread(file, 1))
    return data

if __name__ == '__main__':
    img_list = fileRead()
    count = 0
    for img in img_list:
        count += 1
        print('input ' + str(count) + ' image')
        img = img.astype(dtype=np.float32)
        # HSV変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        reflectance, illumination, noise = LLIE(v, 0.05, 0.01, 1.0, 10, 10).build(v)
        #illumination = gamma_correction(illumination, 2.2)
        cv2.imshow("Output", (illumination * reflectance).astype(dtype=np.uint8))
        cv2.waitKey(0)