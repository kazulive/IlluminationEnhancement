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

def getKernel(img):
    sizeF = np.shape(img)
    diff_kernelX = np.expand_dims(np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), axis=1)
    diff_kernelY = np.expand_dims(np.array([[-1], [0], [1]]), axis=0)
    eigsDtD = np.abs(psf2otf(diff_kernelX[0], sizeF)) ** 2 + np.abs(psf2otf(diff_kernelX[0].T, sizeF)) ** 2
    return eigsDtD
