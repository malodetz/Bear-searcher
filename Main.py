import shutil

from Converter import *
from Process import *

imagenames = os.listdir("Images")
shutil.rmtree("Results")
os.mkdir("Results")
for filename in imagenames:
    rects = process_byHSV(filename)
    bear = getRectangle(filename)
    ok = False
    for rect in rects:
        if bear.intersects(rect):
            ok = True
            writeRectangle(filename, rect, True)
        else:
            writeRectangle(filename, rect, False)
    if ok:
        print(filename + ": Bear was found")
        print("Number of mistakes: " + str(len(rects)))
    else:
        print(filename + ": Bear wasn' t found")
