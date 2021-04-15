import os

import cv2
import numpy as np
from PIL import ImageEnhance, Image

names = os.listdir("images")
for name in names:
    img = Image.open("images/" + name)
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(5)
    img = np.array(img)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByConvexity = False
    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 100
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints " + name, im_with_keypoints)
    cv2.waitKey(0)