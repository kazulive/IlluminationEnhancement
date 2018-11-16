import cv2
from clahe import *
from unsharpmask import *
from shrink import *
import numpy as np

def sum(img_adjusted, img_unsharp):
    ############################################################################
    ##                           画像の足し合わせ                             ##
    ############################################################################
    result = cv2.add(0.5 * img_unsharp, 0.5 * img_adjusted)
    return result

def component_fusion(luminance, imgName, dirName):
    ############################################################################
    ##                         NonLinearStretch関数                           ##
    ############################################################################
    luminance = luminance.astype(dtype = np.float32)
    print('----NonLinearStretch----')
    l_adjusted = nonLinearStretch(luminance)
    #cv2.imshow("luminance_adjusted", l_adjusted.astype(dtype = np.uint8))
    #cv2.waitKey()
    ############################################################################
    ##                              CLAHE関数                                 ##
    ############################################################################
    print('----CLAHE----')
    l_clahe = cleary(l_adjusted.astype(dtype = np.uint8))
    #cv2.imshow("luminance_clahe", l_clahe.astype(dtype = np.uint8))
    #cv2.waitKey()
    ############################################################################
    ##                            Unsharp Masking                             ##
    ############################################################################
    #print('----Unsharp Mask----')
    #l_unsharp = unsharp_mask(luminance)

    #cv2.imwrite(dirName + "shrink" + str(imgName) + ".bmp", (luminance_adjusted).astype(dtype = np.uint8))
    #cv2.imwrite(dirName + "clahe" + str(imgName) + ".bmp", (luminance_clahe).astype(dtype = np.uint8))
    #cv2.imwrite(dirName + "unsharp mask" + str(imgName) + ".bmp", (luminance_unsharp).astype(dtype = np.uint8))

    #cv2.imshow("luminance_unsharp", l_unsharp.astype(dtype=np.uint8))
    #cv2.waitKey()
    #result = sum(l_adjusted.astype(dtype = np.float32), l_clahe.astype(dtype = np.float32), l_unsharp.astype(dtype = np.float32))
    #result = sum(l_adjusted.astype(dtype=np.float32), l_unsharp.astype(dtype=np.float32))
    #result = np.copy(result)
    #print(np.max(result))
    #cv2.normalize(result, result, 0.0, 255.0, cv2.NORM_MINMAX)
    return l_clahe