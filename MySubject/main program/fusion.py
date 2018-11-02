import cv2
import numpy as np

def component_fusion(img_variational, img_shrink, img_unsharp, img_clahe):
    """3つの要素を足し合わせる"""
    result = cv2.add(cv2.add(cv2.add(0.7*img_variational, 0.1*img_clahe), 0.1*img_unsharp), 0.1*img_shrink)
    return result