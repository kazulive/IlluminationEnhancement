import cv2

############################################################################
##                            CLAHE関数                                   ##
##             (Contrast Limited Adaptive Histogram Equalization)         ##
############################################################################
def cleary(img, clip_limit=3, grid=(8,8), thresh=225):
    # ヒストグラム平均化
    clahe = cv2.createCLAHE(clip_limit, grid)
    img = img.copy()
    clahe_adaptive = clahe.apply(img)
    th = clahe_adaptive.copy()
    th[clahe_adaptive > thresh] = 255
    return th