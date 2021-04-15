import os

import cv2

path = "C:/Programs/Datasets/Bears_small/train/images/"
names = os.listdir(path)
print(*names)
for name in names:
    img = cv2.imread(path + name, cv2.IMREAD_COLOR)
    print(img.shape)
    img = img[96:-96, 96:-96]
    print(img.shape)
    cv2.imwrite("resized/" + name[:-4] + ".PNG", img)
