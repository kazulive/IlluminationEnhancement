import cv2
import numpy as np

def division(src, dst):
    div = src.copy()
    non_zero = dst.real != 0
    print(non_zero)
    div[non_zero] = src[non_zero] / dst[non_zero]
    div[~non_zero] = 0.0 + 0.0j
    print(div)
    return div

############################################################################
##                       各チャネルのFFT計算                              ##
############################################################################
def culcFFT(img, sum):
    fimg = np.fft.fft2(img)
    #fimg = np.fft.fftshift(fimg)
    #fimg.real = division(fimg.real, sum.real)
    #fimg.imag = division(fimg.imag, sum.imag)
    #print(fimg)
    #print('\n')
    #print(sum)
    fimg = division(fimg , sum)
    #fimg.real[fimg.real == np.inf] = 0
    #fimg.real[fimg.real == -1*np.inf] = 0
    #fimg.imag[fimg.imag == np.inf] = 0
    #fimg.imag[fimg.imag == -1*np.inf] = 0

    #fimg = np.fft.fftshift(fimg)

    ifimg = np.fft.ifft2(fimg)
    ifimg = ifimg.real

    return ifimg

def edgeFFT(img):
    fimg = np.fft.fft2(img)
    fimg_conj = np.conj(fimg)
    fimg_result = fimg * fimg_conj
    return fimg_result.real

def edge(img):
    sobel_x = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    return cv2.filter2D(img, -1, sobel_x), cv2.filter2D(img, -1, sobel_y)
############################################################################
##                       Variational Retinex Model                        ##
############################################################################
def variationalRetinex(img, luminance0, bright, alpha, beta, gamma, channel, imgName, dirNameR, dirNameL):
    # 3チャネル用処理
    if(channel == 3):
        print('----Variational Retinex Model(3 channel)----')
        ############################################################################
        ##                           各処理の前準備                               ##
        ############################################################################
        count = 0
        eps_r = 0.0
        eps_l = 0.0
        # BGR分割
        b, g, r = cv2.split(img)
        # 画像サイズ
        H, W = img.shape[:2]
        # 照明成分の初期化
        luminance = luminance0.copy()
        reflectance = np.zeros((H, W, 3), np.float32)

        ############################################################################
        ##                           デルタ関数定義                               ##
        ############################################################################
        delta = np.ones((H, W), np.float32)
        gdelta = delta + gamma
        fdimage = delta
        fgdimage = gdelta

        ############################################################################
        ##                           微分オペレータ                               ##
        ############################################################################
        sobel_x = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1,  0, -1]])

        sobel_y = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])

        ############################################################################
        ##                           F(dx), F(dy)                                 ##
        ############################################################################
        fsximage = np.fft.fft2(sobel_x)
        fsyimage = np.fft.fft2(sobel_y)
        fsx_shift = np.fft.fftshift(fsximage)
        fsy_shift = np.fft.fftshift(fsyimage)
        fsx_mag_spectrum = np.abs(fsx_shift) / np.amax(np.abs(fsx_shift))
        fsy_mag_spectrum = np.abs(fsy_shift) / np.amax(np.abs(fsy_shift))
        fsx_mag_spectrum = cv2.resize(fsx_mag_spectrum, (W, H))
        fsy_mag_spectrum = cv2.resize(fsy_mag_spectrum, (W, H))

        ############################################################################
        ##                           R, Lの計算時の分母　                         ##
        ############################################################################
        sum = fsx_mag_spectrum**2 + fsy_mag_spectrum**2
        sumR = fdimage + beta * sum
        sumL = fgdimage + alpha * sum

        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        while (True):
            count += 1
            reflectance_prev = reflectance.copy()
            luminance_prev = luminance.copy()

            # I / Lの計算 その後、分割
            IL = cv2.divide((img * 255).astype(dtype=np.float32), (255 * luminance).astype(dtype=np.float32))
            ILB, ILG, ILR = cv2.split(IL)

            reflectance = cv2.merge((culcFFT(ILB, sumR), culcFFT(ILG, sumR), culcFFT(ILR, sumR)))
            cv2.normalize(reflectance, reflectance, 0, 1, cv2.NORM_MINMAX)

            IR = cv2.divide((img * 255).astype(dtype=np.float32), (255 * reflectance).astype(dtype=np.float32))
            IRB, IRG, IRR = cv2.split(IR)
            IRB += gamma * (bright)
            IRG += gamma * (bright)
            IRR += gamma * (bright)

            luminance = cv2.merge((culcFFT(IRB, sumL), culcFFT(IRG, sumL), culcFFT(IRR, sumL)))
            cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)

            lb, lg, lr = cv2.split(luminance)
            maxb = np.maximum(lb, b)
            maxg = np.maximum(lg, g)
            maxr = np.maximum(lr, r)
            luminance = cv2.merge((maxb, maxg, maxr))

            if(count != 1):
                eps_r = cv2.divide(np.abs(np.sum(reflectance) - np.sum(reflectance_prev)), np.abs(np.sum(reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
                print(eps_r[0])
                if(eps_r[0] <= 0.1 and eps_l[0] <= 0.1):
                    print('----Variational Retinex End----')
                    break

    # 1チャネル用処理
    else:
        print('----Variational Retinex Model(1 channel)----')
        ############################################################################
        ##                           各処理の前準備                               ##
        ############################################################################
        count = 0
        eps_r = 0.0
        eps_l = 0.0
        # 画像サイズ
        H, W = img.shape[:2]
        # 照明成分の初期化
        luminance = luminance0.copy()
        reflectance = np.zeros((H, W), np.float32)

        ############################################################################
        ##                           デルタ関数定義                               ##
        ############################################################################
        delta = np.ones((H, W), np.float32)
        gdelta = delta + gamma
        fdimage = delta
        fgdimage = gdelta

        ############################################################################
        ##                           微分オペレータ                               ##
        ############################################################################
        #sobel_x = np.array([[-1, 0, 1],
        #                    [-1, 0, 1],
        #                    [-1, 0, 1]])

        sobel_x = np.array([-1, 0, 1])
        sobel_y = sobel_x.T
        sum = np.dot(sobel_x, sobel_y)
        """""""""
        #sobel_y = sobel_x.T
        #sobel_y = np.array([[-1, -2, -1],
        #                    [0, 0, 0],
        #                    [1, 2, 1]])

        ############################################################################
        ##                           F(dx), F(dy)                                 ##
        ############################################################################
        fsximage = np.fft.fft2(sobel_x)
        fsyimage = np.fft.fft2(sobel_y)
        fsx_shift = np.fft.fftshift(fsximage)
        fsy_shift = np.fft.fftshift(fsyimage)
        fsx_mag_spectrum = np.abs(fsx_shift) / np.amax(np.abs(fsx_shift))
        fsy_mag_spectrum = np.abs(fsy_shift) / np.amax(np.abs(fsy_shift))
        fsx_mag_spectrum = cv2.resize(fsx_mag_spectrum, (W, H))
        fsy_mag_spectrum = cv2.resize(fsy_mag_spectrum, (W, H))
    
        ############################################################################
        ##                           R, Lの計算時の分母　                         ##
        ############################################################################
        sum = fsx_mag_spectrum ** 2 + fsy_mag_spectrum ** 2
        """""""""
        sumR = delta + beta * sum * delta
        sumL = delta + gamma + alpha * sum * delta
        sumR = np.fft.fft2(sumR)
        sumL = np.fft.fft2(sumL)
        #sumR = fdimage + beta * sum * fdimage
        #sumL = fgdimage + alpha * sum * fdimage

        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        while (True):
            count += 1
            reflectance_prev = reflectance.copy()
            luminance_prev = luminance.copy()

            # I / Lの計算 その後、分割
            IL = cv2.divide((255 * img).astype(dtype=np.float32), (255 * luminance).astype(dtype=np.float32))
            dxR, dyR = edge(reflectance)
            sum = edgeFFT(dxR) + edgeFFT(dyR)
            sumR = fdimage + beta * sum
            reflectance = culcFFT(IL, sumR)
            #reflectance = cv2.min(1.0, cv2.max(reflectance, 0.0))
            reflectance = reflectance.copy()
            cv2.normalize(reflectance, reflectance, 0, 1, cv2.NORM_MINMAX)

            IR = cv2.divide((255 * img).astype(dtype=np.float32), (reflectance).astype(dtype=np.float32))
            IR += gamma * 255 * bright
            dxL, dyL = edge(luminance)
            sum = edgeFFT(dxL) + edgeFFT(dyL)
            sumL = fgdimage + alpha * sum

            luminance = culcFFT(IR, sumL)
            luminance = luminance.copy()
            cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)
            luminance = np.maximum(luminance, img)

            cv2.imshow("Conv Luminance", (255 * luminance).astype(dtype=np.uint8))
            cv2.imshow("Conv Result", (255 * reflectance).astype(dtype=np.uint8))
            cv2.waitKey()
            if (count != 1):
                eps_r = cv2.divide(np.abs(np.sum(reflectance) - np.sum(reflectance_prev)),
                                   np.abs(np.sum(reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
                if (eps_r[0] <= 0.05 and eps_l[0] <= 0.05):
                    print('----Variational Retinex End----')
                    break

    return reflectance, luminance