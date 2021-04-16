import os
import random

import cv2

emptynames = os.listdir("empty")
empty = []
cnt = 0
for name in emptynames:
    path = "empty/" + name
    full = cv2.imread(path, cv2.IMREAD_COLOR)
    h = full.shape[0]
    w = full.shape[1]
    for i in range(900):
        x = random.randint(0, h - 100)
        y = random.randint(0, w - 100)
        cnt += 1
        subrect = full[x:x + 100, y:y + 100]
        cv2.imwrite("in/" + str(cnt) + ".JPG", subrect)
