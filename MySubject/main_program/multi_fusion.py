import cv2
from clahe import *
from unsharpmask import *
from shrink import *
import numpy as np

def sum(img_adjusted, img_clahe, img_unsharp):
    ############################################################################
    ##                           画像の足し合わせ                             ##
    ############################################################################
    result = cv2.add(cv2.add(0.2*img_clahe, 0.2*img_unsharp), 0.6*img_adjusted)
    return result

def component_fusion(luminance, imgName, dirName):
    ############################################################################
    ##                         NonLinearStretch関数                           ##
    ############################################################################
    luminance = luminance.astype(dtype = np.float32)
    cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)
    lb, lg, lr = cv2.split(luminance)
    lb_adjusted = nonLinearStretch(lb)
    lg_adjusted = nonLinearStretch(lg)
    lr_adjusted = nonLinearStretch(lr)

    cv2.normalize(lb_adjusted, lb_adjusted, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(lg_adjusted, lg_adjusted, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(lr_adjusted, lr_adjusted, 0, 1, cv2.NORM_MINMAX)

    luminance_adjusted = cv2.merge((lb, lg, lr))
    ############################################################################
    ##                              CLAHE関数                                 ##
    ############################################################################
    luminance = (255 * luminance).astype(dtype=np.uint8)
    cv2.normalize(luminance, luminance, 0, 255, cv2.NORM_MINMAX)
    hsv = cv2.cvtColor(luminance, cv2.COLOR_BGR2HSV)
    lh, ls, lv = cv2.split(hsv)
    print('----CLAHE(V chのみ)----')
    lv_clahe = cleary(lv)

    hsv = cv2.merge((lh, ls, lv_clahe))
    luminance_clahe = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    ############################################################################
    ##                            Unsharp Masking                             ##
    ############################################################################
    lb, lg, lr = cv2.split(luminance)
    print('----Unsharp Mask----')
    lb_unsharp = unsharp_mask(lb)
    lg_unsharp = unsharp_mask(lg)
    lr_unsharp = unsharp_mask(lr)

    cv2.normalize(lb_unsharp, lb_unsharp, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lg_unsharp, lg_unsharp, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(lr_unsharp, lr_unsharp, 0, 255, cv2.NORM_MINMAX)

    luminance_unsharp = cv2.merge((lb_unsharp, lg_unsharp, lr_unsharp))

    cv2.imwrite(dirName + "shrink" + str(imgName) + ".bmp", (255 * luminance_adjusted))
    cv2.imwrite(dirName + "clahe" + str(imgName) + ".bmp", luminance_clahe)
    cv2.imwrite(dirName + "unsharp mask" + str(imgName) + ".bmp", luminance_unsharp)

    luminance = luminance.copy().astype(dtype = np.float32)
    luminance_adjusted = luminance_adjusted.copy().astype(dtype = np.float32)
    luminance_clahe = luminance_clahe.copy().astype(dtype=np.float32)
    luminance_unsharp = luminance_unsharp.copy().astype(dtype=np.float32)

    lb, lg, lr = cv2.split(luminance)
    lb_adjusted, lg_adjusted, lr_adjusted = cv2.split(luminance_adjusted)
    lb_clahe, lg_clahe, lr_clahe = cv2.split(luminance_clahe)
    lb_unsharp, lg_unsharp, lr_unsharp = cv2.split(luminance_unsharp)

    result_b = sum((255*lb_adjusted), lb_clahe, lb_unsharp)
    result_g = sum((255*lg_adjusted), lg_clahe, lg_unsharp)
    result_r = sum((255*lr_adjusted), lr_clahe, lr_unsharp)

    cv2.normalize(result_b, result_b, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(result_g, result_g, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(result_r, result_r, 0, 255, cv2.NORM_MINMAX)

    result = cv2.merge((result_b, result_g, result_r))
    return result