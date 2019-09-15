import cv2
import numpy as np

from Rectangle import *


def get_rects_from_binarized(th):
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


def process_naive(name):
    path = "Images/" + name
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    greyscale = cv2.medianBlur(img, 15)
    greyscale = cv2.cvtColor(greyscale, cv2.COLOR_RGB2GRAY)
    r, th = cv2.threshold(greyscale, 110, 255, cv2.THRESH_BINARY_INV)
    return get_rects_from_binarized(th)


def process_byHSV(name):
    path = "Images/" + name
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img = cv2.medianBlur(hsv_img, 15)
    mask1 = hsv_img[:, :, 0] >= 80
    mask2 = hsv_img[:, :, 0] <= 85
    mask = mask1 & mask2
    mask = 255 * mask.astype(np.uint8)
    return get_rects_from_binarized(mask)


def process_byBlue(name):
    path = "Images/" + name
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 2]
    img = cv2.medianBlur(img, 15)
    r, th = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    return get_rects_from_binarized(th)
