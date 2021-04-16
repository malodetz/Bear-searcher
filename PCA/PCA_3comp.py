import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

imagenames = os.listdir("out")
random.shuffle(imagenames)
train = []
test = []
for i in range(124):
    path = "out/" + imagenames[i]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if i >= 115:
        test.append(img.flatten())
    else:
        train.append(img.flatten())
train = np.asarray(train)
print(train.shape)
pca = decomposition.PCA(n_components=3)
fig = plt.figure()
ax = plt.axes(projection='3d')
components = pca.fit_transform(train)
for point in components:
    ax.scatter3D(point[0], point[1], point[2], color="blue")
test = np.asarray(test)
testpoints = pca.transform(test)
for point in testpoints:
    ax.scatter3D(point[0], point[1], point[2], color="red")
epmtynames = os.listdir("in")
empty = []
for name in epmtynames:
    path = "in/" + name
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    empty.append(img.flatten())
empty = np.asarray(empty)
nonbears = pca.transform(empty)
for point in nonbears:
    ax.scatter3D(point[0], point[1], point[2], color="green")
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.show()
