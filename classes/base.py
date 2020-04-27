from numba import njit, prange


@njit(fastmath=True, parallel=True)
def mse(im1, im2):
    im1 = im1.reshape((1, -1))
    im2 = im2.reshape((1, -1))
    mse = 0
    for i in prange(im1.shape[1]):
        el1 = im1[0, i]
        el2 = im2[0, i]
        mse += (el1 - el2) ** 2

    return mse


class Image:
    def __init__(self, original_image, *args, **kwargs):
        self.original_image = original_image
        self.size = original_image.shape[:2]
        self.fit = None

    def crossover(self, other):
        """
        Crossover function.
        """

    def mutate(self, *args, **kwargs):
        """
        Mutate function
        :return: new mutated individual
        """

    def draw(self):
        """
        Draw the image.
        :return: the cv2 image.
        """

    def __mul__(self, other):
        return self.crossover(other)

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    def __gt__(self, other):
        return self.fitness() > other.fitness()

    def fitness(self, image=None, **kwargs):
        if self.fit is None:
            if image is None:
                image = self.original_image
            self.fit = mse(image, self.draw())  # 500 / np.linalg.norm(self.draw() - image)
        return self.fit


class Generator:
    def __init__(self, original_image):
        self._original_image = original_image
