import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

fin = open("names.txt", "r")
names = [fin.readline().strip() for i in range(22)]
print(*names)
random.shuffle(names)
n1 = names[:17]
n2 = names[17:]
train = []
test = []
cnt = 0
for name in os.listdir("bears"):
    f = False
    for pref in n1:
        if pref == name[:len(pref)]:
            cnt += 1
            img = cv2.imread("bears/" + name, cv2.IMREAD_COLOR)
            train.append(img.flatten())
            f = True
            break
    if not f and len(test) < 10:
        img = cv2.imread("bears/" + name, cv2.IMREAD_COLOR)
        test.append(img.flatten())
for name in os.listdir("empty"):
    img = cv2.imread("empty/" + name, cv2.IMREAD_COLOR)
    train.append(img.flatten())
train = np.asarray(train)
test = np.asarray(test)
print(train.shape)
print(test.shape)
pca = decomposition.PCA(n_components=100)
c1 = pca.fit_transform(train)
c2 = pca.transform(test)
idx = 0
for i in range(10):
    plt.clf()
    for point in c1:
        if idx < cnt:
            plt.scatter(point[i], point[i+1], color="blue")
        else:
            plt.scatter(point[i], point[i+1], color="red")
        idx += 1
    plt.show()
