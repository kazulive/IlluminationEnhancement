#coding: utf-8

#############################################################################
##                          3つの定量的数値評価                            ##
##                          1. Mean(平均値)                                ##
##                          2. Clarify(明瞭さ)                             ##
##                          3. CCI(カラフルさ)                             ##
#############################################################################

import cv2
import numpy as np
from numpy import unique
from scipy.stats import entropy as scipy_entropy

#############################################################################
##                          画素値の総和                                   ##
#############################################################################
def Sum(img):
    H, W = img.shape[:2]
    sum = np.sum(img)
    mean = sum / (H * W)
    return mean

#############################################################################
##                          平均値(MEAN)                                   ##
#############################################################################
def Mean(img):
    H, W = img.shape[0], img.shape[1]
    b, g, r = cv2.split(img)
    Mean = (Sum(b) + Sum(g) + Sum(r)) / 3.0
    return Mean

#############################################################################
##                         明瞭度(CLARIFY)                                 ##
#############################################################################
def Clarify(img, dst):
    sigmaImg = np.std(img)
    sigmaDst = np.std(dst)
    return np.log10(sigmaDst / (sigmaImg + sigmaDst)) - np.log10(1/2)

#############################################################################
##                   カラフルさ(COLOR COLORFULNESS INDEX)                  ##
#############################################################################
def CCI(img):
    b, g, r = cv2.split(img)
    rg = np.absolute(r - g)
    yb = np.absolute((r + g) / 2.0 - b)

    (rgMean, rgStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

    return stdRoot + (0.3 * meanRoot)

def image_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    entropyValue = 0.0
    totalSize = img.shape[0] * img.shape[1]
    for i in range(0, 256):
        if(hist[i] > 0):
            entropyValue += (hist[i]/totalSize) * (np.log2(hist[i]/totalSize))
    return np.array(entropyValue) * (-1)

if __name__ == '__main__':
    imgName = input('input image name : ')
    fileName = input('file name : ')
    fout = open("evaluate_data/" + fileName + "/text.txt", "w")
    i = int(imgName)
    print('----Start To Evaluate----')

    while (i <= 5):
        img = cv2.imread("testdata/0" + str(i) + ".bmp" )
        result = cv2.imread("evaluate_data/" + fileName + "/0" + str(i) + ".bmp")
        print('----Input 0' + str(i) + '.bmp-----')
        fout.writelines("----Evaluate 0" + str(i) + ".bmp-----\n")
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        print('----Mean----')
        print('Mean : ', Mean(result))
        fout.writelines("Mean = " + str(Mean(result)) + "\n")

        print('----Clarify----')
        print('Clarify : ', Clarify(img, result))
        fout.writelines("Clarify = " + str(Clarify(img, result)) + "\n")

        print('----Color Colorfulness Index----')
        print('CCI : ', CCI(result))
        fout.writelines("CCI = " + str(CCI(result)) + "\n")

        print('----Shannon_Entropy----')
        print('Entropy : ', image_entropy(gray))
        fout.writelines("Entropy = " + str(image_entropy(gray)) + "\n\n")
        i += 1
    print("---Program End----")