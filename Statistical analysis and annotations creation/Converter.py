import os
import xml.etree.ElementTree as ET

from cv2 import imread, imwrite

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
    if not os.path.exists(name):
        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = "Results"
        ET.SubElement(root, "filename").text = filename
        path = ET.SubElement(root, "path")
        path.text = "Results/" + filename
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        img = imread("Images/" + filename)
        w, h, d = img.shape
        imwrite("Results/" + filename, img)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(d)
        ET.SubElement(root, "segmented").text = "0"
        tree = ET.ElementTree(root)
        tree.write(name)
    tree = ET.parse(name)
    root = tree.getroot()
    object = ET.SubElement(root, "object")
    if type:
        ET.SubElement(object, "name").text = "Bear"
    else:
        ET.SubElement(object, "name").text = "Fake Bear"
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"
    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(rectangle.xmin)
    ET.SubElement(bndbox, "ymin").text = str(rectangle.ymin)
    ET.SubElement(bndbox, "xmax").text = str(rectangle.xmax)
    ET.SubElement(bndbox, "ymax").text = str(rectangle.ymax)
    tree.write(name)
