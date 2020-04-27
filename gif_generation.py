import os

import imageio

png_dir = 'int_output3'
images = []
for i in range(1, 199 * 2):
    file_path = os.path.join(png_dir, f"{i * 500}.png")
    images.append(imageio.imread(file_path))
imageio.mimsave('voronoi_output/apple.gif', images)
