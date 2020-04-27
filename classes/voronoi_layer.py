from scipy.spatial.qhull import Voronoi

from .base import mse
from .layers import *


class LayerVoronoiImage(Image):
    def __init__(self, layers, original_image, pts_number, size):
        super().__init__(original_image)
        self.a_im = None
        self.layers = layers
        self.im = None
        self.pts_number = pts_number
        self.radius = 4  # size[0] // int(np.sqrt(pts_number)) // 2
        self.size = size
        self.rgb = [None] * 3
        self.polys = [None] * 3
        self.ch_l_point = None

    def __deepcopy__(self, memo):
        copy = type(self)(None, self.original_image, self.pts_number, self.size)
        memo[id(self)] = copy
        copy.layers = deepcopy(self.layers, memo)
        copy.rgb = deepcopy(self.rgb)
        copy.polys = [self.polys[i] for i in range(len(self.polys))]
        copy.ch_l_point = None
        return copy

    def draw(self, canvas=None, make_beautiful=False):
        if self.im is None:
            layers = [np.zeros(self.size, dtype=np.uint8) for _ in range(3)] if canvas is None else cv2.split(
                np.copy(canvas))
            for i, layer in enumerate(self.layers):
                # if there is no change in color
                if self.ch_l_point is None or self.ch_l_point[0] != i or self.rgb[i] is None:

                    # if change is in structure
                    if self.rgb[i] is None:
                        if self.polys[i] is None:
                            vor = Voronoi(layer[0], incremental=False)
                            self.polys[i] = vor
                        else:
                            vor = self.polys[i]

                        # draw from scratch
                        ver = np.int32(vor.vertices)
                        for j in range(0, len(vor.point_region)):
                            ri = vor.point_region[j]
                            if len(vor.regions[ri]) != 0 and -1 not in vor.regions[ri]:
                                cv2.fillPoly(layers[i], [ver[vor.regions[ri]].reshape((-1, 1, 2))],
                                             int(self.layers[i][1][j]), cv2.LINE_AA if make_beautiful else cv2.LINE_8)

                    # nothing changed this time
                    else:
                        layers[i] = self.rgb[i]
                else:
                    # change only color
                    vor = self.polys[i]
                    layers[i] = self.rgb[i]
                    j = self.ch_l_point[1]
                    ri = vor.point_region[j]
                    ver = np.int32(vor.vertices)
                    if len(vor.regions[ri]) != 0 and -1 not in vor.regions[ri]:
                        cv2.fillPoly(layers[i], [ver[vor.regions[ri]].reshape((-1, 1, 2))],
                                     int(self.layers[i][1][j]))
            self.rgb = layers
            self.im = cv2.merge(layers)
        return self.im

    # draw centers
    def draw_approximated(self):
        if self.a_im is None:
            self.a_im = LayerVoronoiImage.approximated(self.layers, self.radius, self.size)
        return self.a_im

    # fast method of drawing
    @staticmethod
    def approximated(pts, r, size):
        layers = np.zeros((3, size[0], size[1]), dtype=np.uint8)
        for i in range(len(pts)):
            layer = pts[i]
            for j in range(len(layer[0])):
                x, y = pts[i][0][j]
                c = pts[i][1][j]
                layers[i, y - r: y + r, x - r: x + r] = int(c)
        return cv2.merge(layers)

    def fitness(self, image=None, gen=0, **kwargs):
        if self.fit is None:
            if image is None:
                image = self.original_image
            self.fit = mse(image, self.draw())
            # self.draw_approximated() if gen < 3000 else self.draw())  # 500 / np.linalg.norm(self.draw() - image)
        return self.fit

    def mutate(self, gen):
        mutated = deepcopy(self)
        mutated.im = None
        self._mutate(mutated, gen)
        return mutated

    def _chance(self, gen):
        if gen < 50000:
            return 0.5
        else:
            return 0.5 + (gen - 50000) * 0.5 / 50000

    def _mutate(self, mutated, gen):
        changed_layer = rd.randint(0, 3)
        layer = mutated.layers[changed_layer]
        point = rd.randint(0, len(layer[0]))

        if rd.rand() > self._chance(gen):
            mutated.rgb[changed_layer] = None
            mutated.polys[changed_layer] = None
            xy = rd.randint(0, 2)
            layer[0][point][xy] = rd.randint(0, self.size[xy])
        else:
            layer[1][point] = rd.randint(0, 256)
            mutated.ch_l_point = (changed_layer, point)

    def save(self, name):
        cv2.imwrite(name, self.draw(make_beautiful=True))


class VoronoiLayerImageGenerator(Generator):
    def __init__(self, original_image):
        super().__init__(original_image)

    def generate(self, n=1, pts_numbers=1160):
        ims = []
        for _ in range(n):
            pts = [[[], []] for _ in range(3)]
            for i in range(3):
                for _ in range(3):
                    pts[i][0] = rd.randint(0, self._original_image.shape[0], size=(pts_numbers, 2))
                    pts[i][1] = rd.randint(0, 256, (pts_numbers))
            ims.append(LayerVoronoiImage(pts, self._original_image, pts_numbers, self._original_image.shape[:2]))
        return np.array(ims)
