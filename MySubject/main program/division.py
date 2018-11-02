"""
画像全体の除算を行う
対象画像は[0, 1]ゆえに分母が0の場合は0
"""
import numpy as np

def division(bg_img, fg_img):
    result = np.zeros(bg_img.shape)

    non_zero = fg_img != 0.0000

    result[non_zero] = bg_img[non_zero] / fg_img[non_zero]
    result[~non_zero] = 0.00000

    return result
