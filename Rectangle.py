class Rectangle(object):

    def __init__(self, ymin, xmin, ymax, xmax):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def getarea(self):
        return (self.ymax - self.ymin) * (self.xmax - self.xmin)

    def intersects(self, o):
        ymax = min(self.ymax, o.ymax)
        xmin = max(self.xmin, o.xmin)
        ymin = max(self.ymin, o.ymin)
        xmax = min(self.xmax, o.xmax)
        if ymax <= ymin or xmax <= xmin:
            return False
        else:
            intersection = Rectangle(ymin, xmin, ymax, xmax)
            return intersection.getarea() >= 0.5 * min(self.getarea(), o.getarea())
