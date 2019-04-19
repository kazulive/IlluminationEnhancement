import cv2
import glob
import numpy as np
from natsort import natsorted

from padding import *

kernel = np.array([[0, 0, 0],
                   [0, -1, 1],
                   [0, 0, 0]])

def max_function(I, lam, beta, delta):

    sobel_h = cv2.filter2D(I.astype(dtype=np.float32), cv2.CV_32F, kernel)
    sobel_v = cv2.filter2D(I.astype(dtype=np.float32), cv2.CV_32F, kernel.T)

    tmp_h = np.maximum(np.abs(sobel_h) - lam / beta, 0.)
    tmp_v = np.maximum(np.abs(sobel_v) - lam / beta, 0.)

    h = cv2.divide(sobel_h, np.abs(sobel_h)) * tmp_h
    v = cv2.divide(sobel_v, np.abs(sobel_v)) * tmp_v

    return h.astype(dtype=np.float32), v.astype(dtype=np.float32)

def get_auxiliary_variables(I, beta, max_beta, lam, sigma):
    F_delta, F_div = getKernel(I)[0], getKernel(I)[1]
    F_conj_h = psf2otf(np.expand_dims(np.array([1, -1]), axis=1),
                       I.shape[:2]).conjugate()  # FFT derivative operateor horizontal
    F_conj_v = psf2otf(np.expand_dims(np.array([1, -1]), axis=1).T,
                       I.shape[:2]).conjugate()  # FFT derivative operateor verical
    Is = np.copy(I)
    while(beta < max_beta):
        #cv2.imshow("is", (255 * Is).astype(dtype=np.uint8))
        #cv2.waitKey(0)
        h, v = max_function(Is, lam=lam, beta=beta, delta=1.0)
        former = np.fft.fft2(I) + beta * (F_conj_h * np.fft.fft2(h) + F_conj_v * np.fft.fft2(v))
        latter = F_delta + beta * F_div
        Is = np.real(np.fft.ifft2(former.astype(dtype=np.float32) / latter.astype(dtype=np.float32)))
        beta = beta * sigma
    return Is.astype(dtype=np.float32)

def fileRead():
    data = []
    for file in natsorted(glob.glob('./testdata/BMP/*.bmp')):
        data.append(cv2.imread(file, 1))
    return data

if __name__ == '__main__':
    img_list = fileRead()
    max_beta = 1e5
    lam = 0.015
    beta0 = 2.0 * lam
    sigma = 2.0
    count = 0
    for img in img_list:
        count += 1
        print('input image file')
        img = img.astype(dtype=np.float32) / 255.
        b, g, r = cv2.split(img)
        b_new = get_auxiliary_variables(b, beta0, max_beta, lam, sigma)
        g_new = get_auxiliary_variables(g, beta0, max_beta, lam, sigma)
        r_new = get_auxiliary_variables(r, beta0, max_beta, lam, sigma)
        result = cv2.merge((b_new, g_new, r_new))
        cv2.imwrite("result/conv/0" + str(count) + ".bmp", (255. * result).astype(dtype=np.uint8))