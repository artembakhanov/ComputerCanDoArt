import cv2
import numpy as np
from numpy import random as rd
from voronoi_layer import LayerVoronoiImage

size = 30

im = LayerVoronoiImage(
    [[rd.randint(0, 512, size=(size, 2)), rd.randint(0, 256, (size))], [rd.randint(0, 512, size=(size, 2)), rd.randint(0, 256, (size))],
     [rd.randint(0, 512, size=(size, 2)), rd.randint(0, 256, (size))]], None)

cv2.imshow("hello", im.draw())
cv2.waitKey()

cock2 = im.mutate(1)
cv2.imshow("hello", cock2.draw())
cv2.waitKey()
a = 2