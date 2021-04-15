import os

import cv2


names = os.listdir("images")
for name in names:
    img = cv2.imread("images/"+name)
    cv2.imwrite("small/" + name[:-4] + ".PNG", img)
