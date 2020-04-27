from copy import deepcopy

import cv2
import numpy as np
from numpy import random as rd

from classes.base import Image, Generator


class LayerFigure:
    def __init__(self):
        self.mutations = None
        self.color = None
        self.layer_ind = None

    def mutate(self):
        rd.choice(self.mutations)()

    def mutate_color(self):
        self.color[self.layer_ind] = rd.randint(0, 256)  # rd.random()

    def draw(self, im):
        pass

    @staticmethod
    def random_color():
        return rd.randint(0, 256)  # rd.random()  #

    @staticmethod
    def random_pos(size, max_out):
        return [rd.randint(-max_out[0], size[0] + max_out[0]), rd.randint(-max_out[1], size[1] + max_out[1])]


class LayerCircle(LayerFigure):

    def __init__(self, size, position, r, color, layer, min_r=15, max_r=90, max_out=(20, 20), seed=None):
        super().__init__()
        self.size = size
        self.r = r
        self.position = position
        self.color = [0, 0, 0]
        self.layer = layer
        self.layer_ind = list("rgb").index(self.layer)
        self.color[self.layer_ind] = color
        self.min_r = min_r
        self.max_r = max_r
        self.max_out = max_out

        self.mutations = [self.mutate_r, self.mutate_color, self.mutate_pos]

        if seed:
            rd.seed(seed)

    def __deepcopy__(self, memodict):
        copy = type(self)(self.size, self.position.copy(), self.r, self.color[self.layer_ind], self.layer)
        memodict[id(self)] = copy
        copy.min_r = self.min_r
        copy.max_r = self.max_r
        copy.max_out = self.max_out
        return copy

    def mutate_r(self):
        self.r = rd.randint(self.min_r, self.max_r)

    def mutate_pos(self):
        coord = 0 if rd.random() < 0.5 else 1
        self.position[coord] = rd.randint(0 - self.max_out[coord], self.size[coord] + self.max_out[coord])

    def draw(self, im):
        cv2.circle(im, tuple(self.position), self.r, self.color[self.layer_ind], -1, cv2.LINE_AA)

    @staticmethod
    def random_r(min_r, max_r):
        return rd.randint(min_r, max_r)

    @staticmethod
    def generate(layer, number=1, size=(512, 512), min_r=25, max_r=90, max_out=(20, 20)):
        if number == 1:
            return LayerCircle(size, LayerCircle.random_pos(size, max_out), LayerCircle.random_r(min_r, max_r),
                               LayerCircle.random_color(), layer, min_r, max_r, max_out)
        else:
            return [LayerCircle.generate(layer, 1, size, min_r, max_r, max_out) for i in range(number)]


class LayerPolygon(LayerFigure):
    def __init__(self, size, positions, color, layer, max_out=(20, 20), seed=None):
        super().__init__()
        self.size = size
        self.positions = positions
        self.color = [0, 0, 0]
        self.layer = layer
        self.layer_ind = list("rgb").index(self.layer)
        self.color[self.layer_ind] = color
        self.max_out = max_out

        self.mutations = [self.mutate_color, self.mutate_positions]

        if seed:
            rd.seed(seed)

    def __deepcopy__(self, memodict):
        copy = type(self)(self.size, deepcopy(self.positions), self.color[self.layer_ind], self.layer, self.max_out)
        memodict[id(self)] = copy
        return copy

    def mutate_positions(self):
        self.positions[rd.randint(len(self.positions))] = LayerFigure.random_pos(self.size, self.max_out)

    def draw(self, im):
        cv2.fillPoly(im, [self.positions.reshape(-1, 1, 2)], self.color[self.layer_ind], cv2.LINE_AA)

    @staticmethod
    def generate(layer, number=1, size=(512, 512), max_out=(20, 20)):
        if number == 1:
            return LayerPolygon(size,
                                np.array([LayerPolygon.random_pos(size, max_out) for _ in range(rd.randint(3, 5))]),
                                LayerPolygon.random_color(), layer, max_out)
        else:
            return [LayerCircle.generate(layer, size=size, max_out=max_out) for _ in range(number)]


class LayerImage(Image):
    def __init__(self, layers, original_image):
        super().__init__(original_image)
        self.layers = layers
        self.im = None

    def __deepcopy__(self, memo):
        copy = type(self)(None, self.original_image)
        memo[id(self)] = copy
        copy.layers = deepcopy(self.layers, memo)
        return copy

    def draw(self, canvas=None):
        if self.im is None:
            layers = [np.zeros([512, 512], dtype=np.uint8) for _ in range(3)] if canvas is None else cv2.split(
                np.copy(canvas))
            for i, layer in enumerate(self.layers):
                for figure in layer:
                    figure.draw(layers[i])
            self.im = cv2.merge(layers)
        return self.im

    def mutate(self, gen):
        mutated = deepcopy(self)
        mutated.im = None
        rd.choice(mutated.layers[rd.randint(0, 3)]).mutate()
        return mutated

    def draw_layer(self, layer):
        im = np.zeros([512, 512], dtype=np.uint8)
        for figure in self.layers[layer]:
            figure.draw()
        return im

    def crossover(self, other):
        f1 = self.fitness()
        f2 = other.fitness()

        new_layers = np.array([[self.layers[i][j] if rd.random() < f1 / (f1 + f2) else other.layers[i][j] for j in
                                range(len(self.layers[i]))] for i in range(3)])

        return LayerImage(new_layers, self.original_image)


class LayerImageGenerator(Generator):
    def __init__(self, original_image, classes=None, distributions=None, class_args=None):
        super().__init__(original_image)
        if classes is None:
            classes = [LayerCircle, LayerPolygon]
        if distributions is None:
            distributions = [1 / len(classes)] * len(classes)
        if len(classes) != len(distributions):
            raise ValueError("The number of probabilities should be the same as the number of classes")

        self._classes = classes
        self._distr = distributions

    def generate(self, n=1, figures_num=10):
        ims = []
        for _ in range(n):
            layer_names = list("rgb")
            layers = [rd.choice(self._classes, figures_num, p=self._distr) for _ in range(3)]
            for j, layer in enumerate(layers):
                for k, figure in enumerate(layer):
                    layer[k] = figure.generate(layer_names[j], )
            ims.append(LayerImage(np.array(layers), self._original_image))
        return np.array(ims)
