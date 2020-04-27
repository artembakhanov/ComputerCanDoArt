from .layers import *
from .base import *
from numpy import random as rd
from copy import deepcopy


class SimpleImage(Image):
    def __init__(self, original_image, figures):
        super().__init__(original_image)
        self.figures = figures

    def mutate(self):
        pass


