import json
import os

import cv2
import numpy as np

names = os.listdir("images")
for name in names:
    path = "annotations/" + name[:-3] + "json"
    data = json.load(open(path))
    shapes = data["shapes"]
    img = np.zeros((300, 300))
    for shape in shapes:
        points = shape["points"]
        for point in points:
            point[0] = int(point[0])
            point[1] = int(point[1])
        points = np.asarray(points)
        cv2.fillPoly(img, pts=[points], color=(255, 255, 255))
    cv2.imwrite("masks/"+name[:-4]+""+".PNG", img)
