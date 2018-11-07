import cv2
import numpy as np
import sys

#############################################################################
##                           重み画像生成                                  ##
#############################################################################
def getWeight(img, eps=10.0):
    print('--------Product Weight Image---------')
    print('eps:' + str(eps))

    weightImg = 1.0 / (np.abs(img) + eps)
    return weightImg

#############################################################################
##                           勾配画像生成                                  ##
#############################################################################
def getGradient(img, Lambda=6.0, sigma=10.0, eps=10.0):
    print('--------Product Gradient Image---------')
    print('lambda:' + str(eps))
    print('sigma:' + str(eps))
    print('eps:' + str(eps))

    dst = img.copy()
    dst[dst < eps] = 0.0

    gradientImg = (1 + Lambda * np.exp(-np.abs(dst)/sigma)) * dst
    return gradientImg

def weightedMatrices(img, kernelX, kernelY, flag):

    if(flag == True):
        difImgX = cv2.filter2D(img, cv2.CV_64F, kernelX)
        weightImg = getWeight(difImgX, eps=10.0)
        gradientImg = getGradient(difImgX, Lambda=6.0, sigma=10.0, eps=10.0)

    else:
        difImgY = cv2.filter2D(img, cv2.CV_64F, kernelY)
        weightImg = getWeight(difImgY, eps=10.0)
        gradientImg = getGradient(difImgY, Lambda=6.0, sigma=10.0, eps=10.0)

    return weightImg, gradientImg


    ############################################################################
    ##         シグモイド関数                                                 ##
    ##         暗い部分を明るく詳細を強調し、明るい部分を強調して強調しすぎ   ##
    ############################################################################
    luminanceAdjusted = 2.0 * atan2(a * luminance) / 3.14

    ############################################################################
    ##         CLAHE(Contrast Limites Adaptive Histogram Equalization)        ##
    ##         最終的な強調結果をより自然にする                               ##
    ############################################################################
    clahe = cv2.createCLAHE(clipLimit=2.0, titleGridSize=(8, 8))
    maxb = clahe.apply(maxb)
    maxg = clahe.apply(maxg)
    maxr = clahe.apply(maxr)
    luminance = cv2.merge((maxb, maxg, maxr))

if __name__ == '__main__':
    imgName = sys.argv[1]
    img = cv2.imread('testdata/' + imgName)
    cv2.imshow("Input Image", img)
    cv2.waitKey()

    # BGRに分割
    b, g, r = cv2.split(img)

    # カーネルX, カーネルY
    kernelX = np.array([[0, 0, 0],
                        [-1, 0, 1],
                        [0, 0, 0]])

    kernelY = np.array([[0, -1, 0],
                        [0, 0, 0],
                        [0, 1, 0]])

    weightB, gradientB = weightedMatrices(b, kernelX, kernelY, True)
    weightG, gradientG = weightedMatrices(g, kernelX, kernelY, True)
    weightR, gradientR = weightedMatrices(r, kernelX, kernelY, True)

    weightImg = cv2.merge((weightB, weightG, weightR))
    gradientImg = cv2.merge((gradientB, gradientG, gradientR))

    print('--------Show Weight Image---------')
    cv2.imshow("Weight Image", weightImg)

    print('--------Show Gradient Image---------')
    cv2.imshow("Gradient Image", gradientImg)

    cv2.waitKey(0)






