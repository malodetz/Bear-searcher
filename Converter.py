import os
import xml.etree.ElementTree as ET

from Rectangle import *


def getRectangle(filename):
    name = "Real Bears/" + filename.split(".")[0] + ".xml"
    tree = ET.parse(name)
    root = tree.getroot()
    ymin, xmin, ymax, xmax = None, None, None, None
    for boxes in root.iter('object'):
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
    return Rectangle(ymin, xmin, ymax, xmax)


def writeRectangle(filename, rectangle, type):
    name = "Results/" + filename.split(".")[0] + ".xml"
    print(os.path.exists(name))
    #     TODO
