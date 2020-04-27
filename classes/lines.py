import numpy as np
from numpy import random as rd
import cv2

from classes.base import Generator, Image


class LineImage(Image):
    def __init__(self, im, original_image):
        super().__init__(original_image)
        self.im = im

    def __deepcopy__(self, memo):
        copy = type(self)(None, self.original_image)
        memo[id(self)] = copy
        copy.im = np.copy(self.im)
        return copy

    def mutate(self):
        return LineImage(
            cv2.line(self.im, (rd.randint(0, 512), rd.randint(0, 512)), (rd.randint(0, 512), rd.randint(0, 512)),
                     color=(rd.randint(0, 512), rd.randint(0, 512), rd.randint(0, 512)), thickness=rd.randint(5, 20)),
            self.original_image)

    def draw(self):
        return self.im


class LineImageGenerator(Generator):
    def __init__(self, original_image):
        super().__init__(original_image)

    def generate(self, n=1, ):
        ims = []
        for _ in range(n):
            ims.append(LineImage(np.full((512, 512, 3), 255), self._original_image))
        return np.array(ims)
