import cv2
import numpy as np

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

    return otf

def getKernel(img, gamma):
    sizeF = np.shape(img)
    kernel = np.expand_dims(np.array([1.0]), axis=1)
    gkernel = np.expand_dims(np.array([1.0 + gamma]), axis=1)
    eigsK = psf2otf(kernel, sizeF)
    eigsgK = psf2otf(gkernel, sizeF)
    diff_kernelX = np.expand_dims(np.array([-1, 1]), axis=1)
    diff_kernelY = np.expand_dims(np.array([[-1], [0], [1]]), axis=0)
    eigsDtD = np.abs(psf2otf(diff_kernelX, sizeF)) ** 2  + np.abs(psf2otf(diff_kernelX.T, sizeF)) ** 2
    return eigsK, eigsgK, eigsDtD

def culcFFT(img, sum):
    fimg = np.fft.fft2(img) / sum
    real = np.real(np.fft.ifft2(fimg))
    return real

def variationalRetinex(img, alpha, beta, gamma, imgName, dirNameR, dirNameL):
    H, W = img.shape[:2]
    init_luminance = cv2.GaussianBlur(img, (5, 5), 5.0)
    luminance = np.copy(init_luminance)
    img = img.astype(dtype=np.float32)
    luminance = luminance.astype(dtype=np.float32)
    reflectance = np.zeros((H, W), np.float32)
    ############################################################################
    ##                           各処理の前準備                               ##
    ############################################################################
    count = 0
    ############################################################################
    ##                           デルタ関数定義                               ##
    ############################################################################
    # delta = np.ones((H, W), np.float32)
    # gdelta = delta + gamma
    ############################################################################
    ##                           微分オペレータ                               ##
    ############################################################################
    sumR = getKernel(img, gamma)[0] + beta * getKernel(img, gamma)[2]
    sumL = getKernel(img, gamma)[1] + alpha * getKernel(img, gamma)[2]
    ############################################################################
    ##                           最適化問題の反復試行                         ##
    ############################################################################
    flag = 0
    while (flag != 1):
        count += 1
        reflectance_prev = np.copy(reflectance)
        luminance_prev = np.copy(luminance)
        # I / Lの計算
        IL = cv2.divide(img, (luminance).astype(dtype=np.float32))
        reflectance = culcFFT(IL, sumR)
        reflectance = np.minimum(1.0, np.maximum(reflectance, 0.0))
        # I / Rの計算
        IR = cv2.divide(img, (reflectance).astype(dtype=np.float32))
        IR += gamma * init_luminance
        luminance = culcFFT(IR, sumL)
        print(np.max(luminance))
        luminance = np.maximum(luminance, img)

        cv2.imwrite(dirNameR + "0" + str(imgName) + str(count) + ".bmp", (255.0 * reflectance).astype(dtype=np.uint8))
        cv2.imwrite(dirNameL + "0" + str(imgName) + str(count) + ".bmp", (luminance).astype(dtype=np.uint8))

        if (count != 1):
            eps_r = cv2.divide(np.abs(np.sum(255.0 * reflectance) - np.sum(255.0 * reflectance_prev)),
                               np.abs(np.sum(255.0 * reflectance_prev)))
            eps_l = cv2.divide(np.abs(np.sum(luminance) - np.sum(luminance_prev)), np.abs(np.sum(luminance_prev)))
            if (eps_r[0] <= 0.05 and eps_l[0] <= 0.05):
                flag = 1

    return reflectance, luminance