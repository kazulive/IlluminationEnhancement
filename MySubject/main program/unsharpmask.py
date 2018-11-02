import cv2
import numpy as np

def unsharp_mask(img, kernel_size=(7, 7), sigma=3.0, amount=0.25, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharppened = float(amount+1) * img - float(amount) * blurred
    sharppened = np.maximum(sharppened, np.zeros(sharppened.shape))
    sharppened = np.minimum(sharppened, 255 * np.ones(sharppened.shape))
    sharppend = sharppened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharppend, img, where=low_contrast_mask)
    return sharppened