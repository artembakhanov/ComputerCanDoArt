import cv2
import numpy as np
from classes import LayerImageGenerator

from ga import GeneticAlgorithm

if __name__ == '__main__':
    original_image = cv2.imread("random1.png")
    generator = LayerImageGenerator(original_image)  # LineImageGenerator(original_image)
    ga = GeneticAlgorithm(generator, "top", 500, population_size=1, kids_size=0, mutation_size=1)
    iterations = 500
    r = np.zeros((iterations, 512, 512))
    g = np.zeros((iterations, 512, 512))
    b = np.zeros((iterations, 512, 512))
    for i in range(iterations):
        a = ga.start()
        print(f"iteration {i}")
        r[i] = a[0].draw_layer(0)
        g[i] = a[0].draw_layer(1)
        b[i] = a[0].draw_layer(2)

    r = np.mean(r, 0)
    g = np.mean(g, 0)
    b = np.mean(b, 0)

    im = cv2.merge((r, g, b))
    cv2.imwrite("iterations_test3.png", im)