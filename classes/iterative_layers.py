from .layers import *


class IterativeLayerImage(LayerImage):
    def __init__(self, layers, original_image, iteration_size=5000):
        super().__init__(layers, original_image)
        self.fixed_image = None
        self.iteration_size = iteration_size

    def __deepcopy__(self, memo):
        copy = super(IterativeLayerImage, self).__deepcopy__(memo)
        copy.fixed_image = self.fixed_image
        return copy

    def draw(self, **kwargs):
        return super(IterativeLayerImage, self).draw(self.fixed_image)

    def mutate(self, gen):
        if gen % 1000 == 0:
            im = self.draw()
            mutated = IterativeLayerImageGenerator(self.original_image).generate()[0]
            mutated.fixed_image = im
        else:
            mutated = super(IterativeLayerImage, self).mutate()
        return mutated


class IterativeLayerImageGenerator(LayerImageGenerator):
    def generate(self, n=1, figures_num=10):
        ims = []
        for _ in range(n):
            layer_names = list("rgb")
            layers = [rd.choice(self._classes, figures_num, p=self._distr) for _ in range(3)]
            for j, layer in enumerate(layers):
                for k, figure in enumerate(layer):
                    layer[k] = figure.generate(layer_names[j], )
            ims.append(IterativeLayerImage(np.array(layers), self._original_image))
        return np.array(ims)
