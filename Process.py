import cv2
import numpy as np

from Rectangle import *


def process_naive(name):
    path = "Images/" + name
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    greyscale = cv2.GaussianBlur(img, (7, 7), 0)
    greyscale = cv2.cvtColor(greyscale, cv2.COLOR_RGB2GRAY)
    r, th = cv2.threshold(greyscale, 110, 255, cv2.THRESH_BINARY_INV)
    output = cv2.connectedComponentsWithStats(th, 8, cv2.CV_32S)
    stats = output[2]
    ans = []
    for ls in stats:
        x = ls[cv2.CC_STAT_LEFT]
        w = ls[cv2.CC_STAT_WIDTH]
        y = ls[cv2.CC_STAT_TOP]
        h = ls[cv2.CC_STAT_HEIGHT]
        if max(w, h) / min(w, h) < 3 and 10000 >= h * w >= 1000:
            ans.append(Rectangle(y, x, y + h, x + w))
    return ans


def process_byHSV(name):
    path = "Images/" + name
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img = cv2.medianBlur(hsv_img, 15)
    mask1 = hsv_img[:, :, 0] >= 35
    mask2 = hsv_img[:, :, 0] <= 85
    mask = mask1 & mask2
    mask = np.invert(mask)
    mask = 255 * mask.astype(np.uint8)
    components = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_16U)
    ans = []
    stats = components[2]
    for ls in stats:
        x = ls[cv2.CC_STAT_LEFT]
        w = ls[cv2.CC_STAT_WIDTH]
        y = ls[cv2.CC_STAT_TOP]
        h = ls[cv2.CC_STAT_HEIGHT]
        if max(w, h) / min(w, h) < 3 and 10000 >= h * w >= 1000:
            ans.append(Rectangle(y, x, y + h, x + w))
    return ans
