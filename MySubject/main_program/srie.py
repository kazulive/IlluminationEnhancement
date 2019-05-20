import os
import cv2
import csv
import glob
import time
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

# importファイル
from padding import *               # 点拡がり関数
from clahe import *                 # CLAHE関数
from shrink import *                # shrinkage関数
from guidedfilter import *          # Weighted Guided Filter関数
from gradient_fusion import *       # 合成関数
from createGaussianPyr import *     # 画像ピラミッド関数
from jnd_threshold import *         # beta計算関数
from agcwd import *                 # agcwd計算関数

"""
# shrinkage function
def shrinkage(reflectance, b_h, b_v, lam):
    former_h = cv2.filter2D(reflectance.astype(dtype=np.float32), cv2.CV_32F, kernel) + b_h
    former_v = cv2.filter2D(reflectance.astype(dtype=np.float32), cv2.CV_32F, kernel.T) + b_v
    latter = 1. / (2. * lam)

    tmp_h = np.maximum(np.abs(former_h) - latter, 0.0)
    tmp_v = np.maximum(np.abs(former_v) - latter, 0.0)

    d_h = cv2.divide(former_h, np.abs(former_h)) * tmp_h
    d_v = cv2.divide(former_v, np.abs(former_v)) * tmp_v1
    
def variationalRetinex(img, alpha, beta, gamma, lam):
    # 配列用意
    H, W = img.shape[:2]
    reflectance = np.zeros((H, W), dtype=np.float32)                                            # 反射画像              => (W, H, 1) float32型
    illumination = cv2.GaussianBlur(img, (5, 5), 2.0)                                           # 照明画像              => (W, H, 1) float32型
    dh = np.zeros((H, W), dtype=np.float32)                                                     # 補助変数 d_horizontal => (W, H, 1) float32型
    dv = np.zeros((H, W), dtype=np.float32)                                                     # 補助変数 d_vertical   => (W, H, 1) float32型
    bh = np.zeros((H, W), dtype=np.float32)                                                     # 誤差 b_horizontal     => (W, H, 1) float32型
    bv = np.zeros((H, W), dtype=np.float32)                                                     # 誤差 b_vertical       => (W, H, 1) float32型
    phi = np.zeros((H, W), dtype=np.float32)                                                    # 反射画像の計算に用いるΦ
    average_img = cv2.filter2D(img, -1, avg_kernel)                                             # I0:平均値画像

    # FFTの事前計算
    F_conj_h = psf2otf(np.expand_dims(np.array([1, -1]), axis=1), img.shape[:2]).conjugate()    # FFT derivative operateor horizontal
    F_conj_v = psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T, img.shape[:2]).conjugate()  # FFT derivative operateor verical
    F_delta, F_div = getKernel(img)[0], getKernel(img)[1]                                       # FFT delta function, F(∇h)*・F(∇h) + F(∇v)*・F(∇v)

    # 最適化式を解く d -> reflectance -> b -> illumination
    flag = 0
    count = 0
    while(flag != 1):
        # 以前の画像を保存
        reflectance_prev = np.copy(reflectance)
        illumination_prev = np.copy(illumination)
        bh_prev = np.copy(bh)
        bv_prev = np.copy(bv)

        # Step 1
        if (count != 0):
            # dh, dvを求める
            dh, dv = shrinkage(reflectance_prev, bh_prev, bv_prev, lam)

        # Step 2
        if (count != 0):
            # Φを求める
            phi = F_conj_h * np.fft.fft2(dh - bh_prev) + F_conj_v * np.fft.fft2(dv - bv_prev)
        # 分子・分母の計算
        top = np.fft.fft2(img / (illumination_prev + 0.1)) + beta * lam * phi
        bottom = F_delta + beta * lam * F_div
        # Reflectance(反射画像)を求める
        reflectance = np.real(np.fft.ifft2(top / bottom))
        # Reflectanceの値調整
        reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))

        # Step 3
        # errorを求める
        b_h, b_v = diff(bh_prev, bv_prev, dh, dv, reflectance)

        # Step 4
        # 分子・分母の計算
        top = np.fft.fft2(gamma * average_img + img / (reflectance + 0.1))
        bottom = F_delta + np.ones((H, W)) * gamma + alpha * F_div
        # Illumination(照明画像)を求める
        illumination = np.real(np.fft.ifft2(top / bottom))
        np.savetxt("luminance" + ".csv", illumination, fmt="%0.2f", delimiter=",")
        # Illuminationの値調整
        illumination = np.maximum(illumination, img)

        #cv2.imshow("reflectance", (255 * reflectance).astype(dtype=np.uint8))
        #cv2.imshow("illumination", illumination.astype(dtype=np.uint8))
        #cv2.waitKey()

        if(count != 0):
            # 収束条件
            eps_r = cv2.divide(np.abs(np.sum(255 * reflectance) - np.sum(255 * reflectance_prev)),
                           np.abs(np.sum(255 * reflectance_prev)))
            eps_l = cv2.divide(np.abs(np.sum(illumination) - np.sum(illumination)), np.abs(np.sum(illumination)))

            if (eps_r[0] <= 0.1 and eps_l[0] <= 0.1):
                flag = 1
        count += 1

    return reflectance, illumination
"""

# 微分画像カーネル
kernel = np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]])

# 平均値画像カーネル
avg_kernel = np.ones((5, 5), np.float32) / 25.

class SRIE:
    def __init__(self, img, image, alpha, beta, gamma, lam, pyr_num):
        self.init = np.sqrt(img[:,:,0]**2 + img[:,:,1]**2 + img[:,:,2]**2)
        self.img = image
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.pyr_num = pyr_num

        self.imgPyr = []

        # 微分画像カーネル
        self.kernel = np.array([[0, 0, 0],
                                [-1, 0, 1],
                                [0, 0, 0]])

        # 平均値画像カーネル
        self.avg_kernel = np.ones((5, 5), np.float32) / 25.

    # VFアップサンプリング
    def upSampling(self, img, illumination, init_illumination):
        up_illumination = cv2.pyrUp(illumination, (img.shape))

        up_illumination = cv2.resize(up_illumination, (img.shape[1], img.shape[0]))
        up_init_illumination = np.copy(up_illumination)

        return up_illumination, up_init_illumination

    # shrinkage function
    def shrinkage(self, reflectance, bh, bv, lam):
        former_h = cv2.filter2D(reflectance.astype(dtype=np.float32), cv2.CV_32F, self.kernel) + bh
        former_v = cv2.filter2D(reflectance.astype(dtype=np.float32), cv2.CV_32F, self.kernel.T) + bv
        latter = 1. / (2. * lam)

        tmp_h = np.maximum(np.abs(former_h) - latter, 0.0)
        tmp_v = np.maximum(np.abs(former_v) - latter, 0.0)

        dh = cv2.divide(former_h, np.abs(former_h)) * tmp_h
        dv = cv2.divide(former_v, np.abs(former_v)) * tmp_v

        return dh, dv

    # error culculation
    def diff(self, bh, bv, dh, dv, reflectance):
        new_bh = bh + cv2.filter2D(reflectance.astype(dtype=np.float32), cv2.CV_32F, self.kernel) - dh
        new_bv = bv + cv2.filter2D(reflectance.astype(dtype=np.float32), cv2.CV_32F, self.kernel.T) - dv

        return new_bh, new_bv

    def srie(self):
        # 配列用意
        H, W = self.img.shape[:2]
        reflectance = np.zeros((H, W), dtype=np.float32)                            # 反射画像              => (W, H, 1) float32型
        illumination = self.init#cv2.GaussianBlur(self.img, (5, 5), 2.0)            # 照明画像              => (W, H, 1) float32型
        dh = np.zeros((H, W), dtype=np.float32)                                     # 補助変数 d_horizontal => (W, H, 1) float32型
        dv = np.zeros((H, W), dtype=np.float32)                                     # 補助変数 d_vertical   => (W, H, 1) float32型
        bh = np.zeros((H, W), dtype=np.float32)                                     # 誤差 b_horizontal     => (W, H, 1) float32型
        bv = np.zeros((H, W), dtype=np.float32)                                     # 誤差 b_vertical       => (W, H, 1) float32型
        phi = np.zeros((H, W), dtype=np.float32)                                    # 反射画像の計算に用いるΦ
        average_img = cv2.filter2D(self.img, -1, self.avg_kernel)                   # I0:平均値画像

        # FFTの事前計算
        F_conj_h = psf2otf(np.expand_dims(np.array([1, -1]), axis=1),
                           self.img.shape[:2]).conjugate()                          # FFT derivative operateor horizontal
        F_conj_v = psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T,
                           self.img.shape[:2]).conjugate()                          # FFT derivative operateor verical
        F_delta, F_div = getKernel(self.img)[0], getKernel(self.img)[1]             # FFT delta function, F(∇h)*・F(∇h) + F(∇v)*・F(∇v)

        # 最適化式を解く d -> reflectance -> b -> illumination
        flag = 0
        count = 0
        while (flag != 1):
            # 以前の画像を保存
            reflectance_prev = np.copy(reflectance)
            illumination_prev = np.copy(illumination)
            bh_prev = np.copy(bh)
            bv_prev = np.copy(bv)

            # Step 1
            if (count != 0):
                # dh, dvを求める
                dh, dv = self.shrinkage(reflectance_prev, bh_prev, bv_prev, self.lam)

            # Step 2
            if (count != 0):
                # Φを求める
                phi = F_conj_h * np.fft.fft2(dh - bh_prev) + F_conj_v * np.fft.fft2(dv - bv_prev)
            # 分子・分母の計算
            top = np.fft.fft2(self.img / (illumination_prev + 0.01)) + self.beta * self.lam * phi
            bottom = F_delta + self.beta * self.lam * F_div
            # Reflectance(反射画像)を求める
            reflectance = np.real(np.fft.ifft2(top / bottom))
            # Reflectanceの値調整
            reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))

            # Step 3
            # errorを求める
            bh, bv = self.diff(bh_prev, bv_prev, dh, dv, reflectance)

            # Step 4
            # 分子・分母の計算
            top = np.fft.fft2(self.gamma * average_img + self.img / (reflectance + 0.01))
            bottom = F_delta + np.ones((H, W)) * self.gamma + self.alpha * F_div
            # Illumination(照明画像)を求める
            illumination = np.real(np.fft.ifft2(top / bottom))
            np.savetxt("luminance" + ".csv", illumination, fmt="%0.2f", delimiter=",")
            # Illuminationの値調整
            illumination = np.maximum(illumination, self.img)

            # cv2.imshow("reflectance", (255 * reflectance).astype(dtype=np.uint8))
            # cv2.imshow("illumination", illumination.astype(dtype=np.uint8))
            # cv2.waitKey()

            if (count != 0):
                # 収束条件
                eps_r = cv2.divide(np.abs(np.sum(255 * reflectance) - np.sum(255 * reflectance_prev)),
                                   np.abs(np.sum(255 * reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(illumination) - np.sum(illumination)), np.abs(np.sum(illumination)))

                if (eps_r[0] <= 0.1 and eps_l[0] <= 0.1):
                    flag = 1
            count += 1

        return reflectance, illumination

    def pyramid_srie(self):
        # 画像ピラミッド生成
        imgPyr = createGaussianPyr(self.img, self.pyr_num)
        for i in range(self.pyr_num - 1, -1, -1):
            img = np.copy(imgPyr[i])
            if (i == self.pyr_num - 1):
                H, W = img.shape[:2]
                img = img.astype(dtype=np.float32)
                reflectance = np.zeros((H, W), np.float32)                                              # 反射画像              => (W, H, 1) float32型
                init_illumination = cv2.GaussianBlur(img, (5, 5), 2.0)                                  # 初期照明画像          => (W, H, 1) float32型
                illumination = np.copy(init_illumination)                                               # 照明画像              => (W, H, 1) float32型
                dh = np.zeros((H, W), dtype=np.float32)                                                 # 補助変数 d_horizontal => (W, H, 1) float32型
                dv = np.zeros((H, W), dtype=np.float32)                                                 # 補助変数 d_vertical   => (W, H, 1) float32型
                bh = np.zeros((H, W), dtype=np.float32)                                                 # 誤差 b_horizontal     => (W, H, 1) float32型
                bv = np.zeros((H, W), dtype=np.float32)                                                 # 誤差 b_vertical       => (W, H, 1) float32型
                phi = np.zeros((H, W), dtype=np.float32)                                                # 反射画像の計算に用いるΦ
                print('----Variational Retinex Model(1 channel)----')
            else:
                H, W = img.shape[:2]
                reflectance = np.zeros((H, W), dtype=np.float32)                                        # 反射画像              => (W, H, 1) float32型
                illumination, init_illumination = self.upSampling(img, illumination, init_illumination) # 照明画像              => (W, H, 1) float32型
                dh = np.zeros((H, W), dtype=np.float32)                                                 # 補助変数 d_horizontal => (W, H, 1) float32型
                dv = np.zeros((H, W), dtype=np.float32)                                                 # 補助変数 d_vertical   => (W, H, 1) float32型
                bh = np.zeros((H, W), dtype=np.float32)                                                 # 誤差 b_horizontal     => (W, H, 1) float32型
                bv = np.zeros((H, W), dtype=np.float32)                                                 # 誤差 b_vertical       => (W, H, 1) float32型
                phi = np.zeros((H, W), dtype=np.float32)                                                # 反射画像の計算に用いるΦ

            # FFTの事前計算
            F_conj_h = psf2otf(np.expand_dims(np.array([1, -1]), axis=1), img.shape[:2]).conjugate()    # FFT derivative operateor horizontal
            F_conj_v = psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T, img.shape[:2]).conjugate()  # FFT derivative operateor verical
            F_delta, F_div = getKernel(img)[0], getKernel(img)[1]                                       # FFT delta function, F(∇h)*・F(∇h) + F(∇v)*・F(∇v)

            # 最適化式を解く d -> reflectance -> b -> illumination
            flag = 0
            count = 0
            while(flag != 1):
                # 以前の画像を保存
                reflectance_prev = np.copy(reflectance)
                illumination_prev = np.copy(illumination)
                bh_prev = np.copy(bh)
                bv_prev = np.copy(bv)

                # alphaの計算
                alpha = getWeight(init_illumination.astype(dtype=np.float32), 1.0)

                #plt.imshow(reflectance, cmap='gray')
                #plt.show()

                """Step 1"""
                if (count != 0):
                    # dh, dvを求める
                    dh, dv = self.shrinkage(reflectance_prev, bh_prev, bv_prev, self.lam)

                """Step 2"""
                if (count != 0):
                    # Φを求める
                    phi = F_conj_h * np.fft.fft2(dh - bh_prev) + F_conj_v * np.fft.fft2(dv - bv_prev)
                # 分子・分母の計算
                top = np.fft.fft2(img / (illumination_prev + 0.01)) + self.beta * self.lam * phi
                bottom = F_delta + self.beta * self.lam * F_div
                # Reflectance(反射画像)を求める
                reflectance = np.real(np.fft.ifft2(top / bottom))
                # Reflectanceの値調整
                reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))

                """Step 3"""
                # errorを求める
                bh, bv = self.diff(bh_prev, bv_prev, dh, dv, reflectance)

                """Step 4"""
                # 分子・分母の計算
                top = np.fft.fft2(self.gamma * init_illumination + img / (reflectance + 0.01))
                bottom = F_delta + np.ones((H, W)) * self.gamma + alpha * F_div
                # Illumination(照明画像)を求める
                illumination = np.real(np.fft.ifft2(top / bottom))
                np.savetxt("luminance" + ".csv", illumination, fmt="%0.2f", delimiter=",")
                # Illuminationの値調整
                illumination = np.maximum(illumination, img)

                #cv2.imshow("reflectance", (255 * reflectance).astype(dtype=np.uint8))
                #cv2.imshow("illumination", illumination.astype(dtype=np.uint8))
                #cv2.waitKey()

                if(count != 0):
                    # 収束条件
                    eps_r = cv2.divide(np.abs(np.sum(255 * reflectance) - np.sum(255 * reflectance_prev)),
                                np.abs(np.sum(255 * reflectance_prev)))
                    eps_l = cv2.divide(np.abs(np.sum(illumination) - np.sum(illumination_prev)), np.abs(np.sum(illumination)))

                    if (eps_r[0] <= 0.1 and eps_l[0] <= 0.1):
                        flag = 1
                count += 1

        return reflectance, illumination

def fileRead():
    data = []
    for file in natsorted(glob.glob('./testdata/BMP/*.bmp')):
        data.append(cv2.imread(file, 1))
    return data

if __name__=='__main__':
    img_list = fileRead()
    fout = open("./pyramid_speed_time.csv", "w")
    writer = csv.writer(fout, lineterminator='\n')
    time_list = []
    print('画像数 : ', len(img_list))
    count = 0
    for img in img_list:
        count += 1
        print('input image file')
        start = time.time()
        img = img.astype(dtype=np.uint8)
        # HSV変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Variational Retinex
        reflectance, illumination = SRIE(img, v, 1000, 0.01, 0.1, 10., 3).srie()
        cv2.imwrite("result/reflectance/0" + str(count) + ".bmp",
                    (255.0 * reflectance).astype(dtype=np.uint8))
        cv2.imwrite("result/illumination/conv0" + str(count) + ".bmp", (illumination).astype(dtype=np.uint8))
        # RGB変換
        #illumination_final = agcwd(illumination.astype(dtype=np.uint8))
        #cv2.imwrite("result/illumination/0" + str(count) + ".bmp", (illumination_final).astype(dtype=np.uint8))
        hsv = cv2.merge((h, s, (cleary(nonLinearStretch(illumination).astype(dtype=np.uint8)) * reflectance).astype(dtype=np.uint8)))
        #prop_hsv = cv2.merge((h, s, (illumination_final * reflectance).astype(dtype=np.uint8)))
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #prop_output = cv2.cvtColor(prop_hsv, cv2.COLOR_HSV2BGR)
        #cv2.imwrite("result/proposal/0" + str(count) + ".bmp", (prop_output).astype(dtype=np.uint8))
        cv2.imwrite("result/proposal/conv0" + str(count) + ".bmp", (output).astype(dtype=np.uint8))
        #plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        #plt.show()
        elapsed_time = time.time() - start
        time_list.append(elapsed_time)
    writer.writerow(time_list)
    fout.close()
    """
    # Weighted Guided Filter
    guidedImg = (guidedFilter(v.astype(dtype=np.float32) / 255.0, v.astype(dtype=np.float32) / 255.0, 16,
                              0.25) * 255.0).astype(dtype=np.uint8)
    illumination_final = gradientFusion(guidedImg.astype(dtype=np.float32), illumination.astype(dtype=np.float32))
    reflectance_final = cv2.divide((v).astype(dtype=np.float32), (illumination_final).astype(dtype=np.float32))
    reflectance_final = np.minimum(1.0, np.maximum(reflectance_final, 0.0))
    np.savetxt("reflectance_final" + ".csv", reflectance_final, fmt="%0.2f", delimiter=",")
    cv2.imshow("reflectance_Final", reflectance_final.astype(dtype=np.float32))
    cv2.waitKey()
    cv2.imwrite("result/reflectance/reflectance0" + str(imgName) + ".bmp",
                (255.0 * np.minimum(1.0, np.maximum(reflectance_final, 0.0))).astype(dtype=np.uint8))
    cv2.imwrite("result/luminance/illumination0" + str(imgName) + ".bmp", (illumination_final).astype(dtype=np.uint8))
    hsv_final = cv2.merge((h, s, (nonLinearStretch(illumination_final) * reflectance_final).astype(dtype=np.uint8)))
    output2 = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    cv2.imwrite("result/proposal/reflectance0" + str(imgName) + ".bmp", (output2).astype(dtype=np.uint8))
    """
