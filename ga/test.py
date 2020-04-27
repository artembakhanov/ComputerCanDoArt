import unittest

import ga.classes


class LayerCircleMethods(unittest.TestCase):
    def test_color_mutations(self):
        circle = ga.classes.LayerCircle((512, 512), [256, 256], 50, 255, 'r', seed=123)
        circle.mutate()
        self.assertEqual(circle.color, [109, 0, 0])

    def test_r_mutations(self):
        circle = ga.classes.LayerCircle((512, 512), [256, 256], 50, 255, 'r', seed=129)
        circle.mutate()
        self.assertEqual(circle.r, 87)

    def test_pos_mutations(self):
        circle = ga.classes.LayerCircle((512, 512), [256, 256], 50, 255, 'r', seed=125)
        circle.mutate()
        self.assertEqual(circle.position, [256, 235])

    def test_random_generation(self):
        circle = ga.classes.LayerCircle.generate("r")
        self.assertEqual(circle.__class__, ga.classes.LayerCircle)
        circles = ga.classes.LayerCircle.generate("r", 10)
        self.assertEqual(circles.__class__, list)
        self.assertEqual(len(circles), 10)


class ImageGeneratorMethods(unittest.TestCase):
    def test_image_generation(self):
        ig = ga.classes.ImageGenerator()
        images = ig.generate(10)


if __name__ == '__main__':
    unittest.main()
