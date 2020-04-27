import cv2

from classes import VoronoiLayerImageGenerator
from ga import GeneticAlgorithm

if __name__ == '__main__':
    original_image = cv2.imread("images/inno.jpg")
    generator = VoronoiLayerImageGenerator(original_image)
    ga = GeneticAlgorithm(generator, "top", 200_000,
                          population_size=1,
                          kids_size=0,
                          mutation_size=1,
                          logging=True,
                          save_int_results=True)
    im = ga.start()[0]
    cv2.imwrite("voronoi_output/voronoi29.png", im.draw(make_beautiful=True))
    cv2.imwrite("voronoi_output/voronoi29_1.png", im.draw_approximated())
