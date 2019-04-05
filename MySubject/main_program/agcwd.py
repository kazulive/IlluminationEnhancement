import cv2
import numpy as np


def agcwd(img, alpha=0.5):
    H, W = img.shape
    enhanced_img = np.copy(img)                                                     # 画像コピー
    pdf = get_pdf(img)                                                              # 正規化ヒストグラム
    max_pdf = np.max(pdf)                                                           # ヒストグラムの最大値
    min_pdf = np.min(pdf)                                                           # ヒストグラムの最小値
    weight_pdf = max_pdf * (((pdf - min_pdf) / (max_pdf - min_pdf)) ** alpha)       # pdf_wの計算
    weight_cdf = np.cumsum(weight_pdf) / np.sum(weight_pdf)                         # cdf_wの計算
    img_intensity = np.arange(0, 256)                                               # [0-255]の輝度値配列
    img_intensity = np.array([255 * (intensity / 255) ** (1 - weight_cdf[intensity]) for intensity in img_intensity], dtype = np.uint8)     # T(intensity)の計算
    for i in range(0, H):
        for j in range(0, W):
            intensity = enhanced_img[i, j]
            enhanced_img[i, j] = img_intensity[intensity]

    return enhanced_img

# 正規化ヒストグラム
def get_pdf(img):
    H, W = img.shape                                                                # 縦横サイズ抽出
    N = H * W                                                                       # 画像サイズ計算
    hist = cv2.calcHist([img.astype(dtype=np.uint8)], [0], None, [256], [0, 256])                          # ヒストグラム計算

    return hist / N                                                                 # 正規化
