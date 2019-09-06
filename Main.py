import os

from Converter import *
from Process import *

imagenames = os.listdir("Images")
for filename in imagenames:
    rects = process_byHSV(filename)
    bear = getRectangle(filename)
    ok = False
    for rect in rects:
        if bear.intersects(rect):
            ok = True
    if ok:
        print(filename + ": Bear was found")
        print("Number of mistakes: " + str(len(rects)))
    else:
        print(filename + ": Bear wasn' t found")