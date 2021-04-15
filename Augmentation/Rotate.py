import os

import cv2
import numpy as np


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


names = os.listdir("images")
for name in names:
    img = cv2.imread("images/"+name)
    for ang in range(4):
        rotated = rotateImage(img, 90*ang)
        cv2.imwrite("rot/" + name[:-4] + "_" + str(ang) + ".JPG", rotated)
