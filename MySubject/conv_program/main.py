import sys
import os

import cv2
import json

import retinex

if __name__ == '__main__':
    imgName = input('input image name : ')
    finName = input('Finish ImageName : ')
    i = int(imgName)
    N = int(finName)
    while(i <= N):
        print('----0' + str(i) + '.bmp----')
        img = cv2.imread("testdata/0" + str(i) + ".bmp")

        with open('config.json', 'r') as f:
            config = json.load(f)

        img_msrcr = retinex.MSRCR(
            img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )

        img_amsrcr = retinex.automatedMSRCR(
            img,
            config['sigma_list']
        )

        img_msrcp = retinex.MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )

        shape = img.shape
        #cv2.imshow('Image', img)
        cv2.imwrite('result/MSRCR/0' + str(i) + 'retinex.bmp', img_msrcr)
        cv2.imwrite('result/MSRCR/0' + str(i) + 'Automated_retinex.bmp', img_amsrcr)
        cv2.imwrite('result/MSRCR/0 ' + str(i) + 'MSRCP.bmp', img_msrcp)

        i += 1