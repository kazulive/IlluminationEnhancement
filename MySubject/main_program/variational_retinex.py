import cv2
import numpy as np

def division(src, dst):
    div = src.copy()
    non_zero = dst.real != 0
    div[non_zero] = src[non_zero] / dst[non_zero]
    div[~non_zero] = 0.0 + 0.0j
    return div

def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape

    psf_pad = zero_pad(psf, shape, position='corner')

    for axis, axis_size in enumerate(inshape):
        psf_pad = np.roll(psf_pad, -int(axis_size / 2), axis=axis)

    otf = np.fft.fft2(psf_pad)

    #n_ops = np.sum(psf_pad.size * np.log2(psf_pad.shape))
    #otf = np.real_if_close(otf, tol=n_ops)

    return otf

def getKernel(img):
    sizeF = np.shape(img)
    diff_kernelX = np.expand_dims(np.array([-1, 1]), axis=1)
    diff_kernelY = np.expand_dims(np.array([[-1], [0], [1]]), axis=0)
    eigsDtD = np.abs(psf2otf(diff_kernelX, sizeF)) ** 2  + np.abs(psf2otf(diff_kernelX.T, sizeF)) ** 2
    return eigsDtD

############################################################################
##                       各チャネルのFFT計算                              ##
############################################################################
def culcFFT(img, sum):
    fimg = np.fft.fft2(img) / sum
    real = np.real(np.fft.ifft2(fimg))
    return real

############################################################################
##                       Variational Retinex Model                        ##
############################################################################
def variationalRetinex(img, luminance0, bright, alpha, beta, gamma, channel, imgName, dirNameR, dirNameL):
    # 3チャネル用処理
    if (channel == 3):
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
        fdelta = np.fft.fft2(delta)
        gdelta = delta + gamma

        ############################################################################
        ##                           R, Lの計算時の分母　                         ##
        ############################################################################
        sumR = fdelta + beta * getKernel(delta)
        sumL = fdelta + gdelta + alpha * getKernel(delta)
        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        while (True):
            count += 1
            reflectance_prev = reflectance.copy()
            luminance_prev = luminance.copy()

            # I / Lの計算 その後、分割
            IL = cv2.divide((img).astype(dtype=np.float32), (luminance).astype(dtype=np.float32))
            ILB, ILG, ILR = cv2.split(IL)

            reflectance = cv2.merge((cv2.min(1, cv2.max(culcFFT(ILB, sumR), 0)), cv2.min(1, cv2.max(culcFFT(ILG, sumR), 0)), cv2.min(1, cv2.max(culcFFT(ILR, sumR), 0))))

            #cv2.normalize(reflectance, reflectance, 0, 1, cv2.NORM_MINMAX)

            IR = cv2.divide((img).astype(dtype=np.float32), (reflectance).astype(dtype=np.float32))
            IRB, IRG, IRR = cv2.split(IR)
            IRB += gamma * (bright[:,:,0])
            IRG += gamma * (bright[:,:,1])
            IRR += gamma * (bright[:,:,2])

            luminance = cv2.merge((culcFFT(IRB, sumL), culcFFT(IRG, sumL), culcFFT(IRR, sumL)))
            #cv2.normalize(luminance, luminance, 0, 1, cv2.NORM_MINMAX)

            lb, lg, lr = cv2.split(luminance)
            maxb = np.maximum(lb, b)
            maxg = np.maximum(lg, g)
            maxr = np.maximum(lr, r)
            luminance = cv2.merge((maxb, maxg, maxr))

            if (count != 1):
                eps_r = cv2.divide(np.abs(np.sum(reflectance) - np.sum(reflectance_prev)),
                                   np.abs(np.sum(reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
                if (eps_r[0] <= 0.01 and eps_l[0] <= 0.01):
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
        ############################################################################
        ##                           微分オペレータ                               ##
        ############################################################################
        sumR = delta + beta * getKernel(img)
        sumL =  gdelta + alpha * getKernel(img)
        ############################################################################
        ##                           最適化問題の反復試行                         ##
        ############################################################################
        while (True):
            count += 1
            reflectance_prev = reflectance.copy()
            luminance_prev = luminance.copy()

            # I / Lの計算 その後、分割
            IL = cv2.divide((img).astype(dtype=np.float32), (luminance).astype(dtype=np.float32))
            reflectance = culcFFT(IL, sumR)
            reflectance = reflectance.copy()
            #print(np.max(reflectance))
            reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))

            IR = cv2.divide((img).astype(dtype=np.float32), (reflectance).astype(dtype=np.float32))
            IR += gamma * bright

            luminance = culcFFT(IR, sumL)
            luminance = luminance.copy()
            #print(np.max(luminance))
            #luminance = 255.0 * luminance / np.max(luminance)
            luminance = np.maximum(luminance, img)
            #cv2.imshow("Conv Luminance", (luminance).astype(dtype=np.uint8))
            #cv2.imshow("Conv Result", (255 * reflectance).astype(dtype=np.uint8))
            #cv2.waitKey()
            if (count != 1):
                eps_r = cv2.divide(np.abs(np.sum(reflectance) - np.sum(reflectance_prev)),
                                   np.abs(np.sum(reflectance_prev)))
                eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
                if (eps_r[0] <= 0.01 and eps_l[0] <= 0.01):
                    print('----Variational Retinex End----')
                    break

    return reflectance, luminance