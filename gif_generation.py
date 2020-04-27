import os

import imageio

png_dir = 'output'
images = []
for i in range(1, 18789):
    file_path = os.path.join(png_dir, f"{i * 100}.png")
    images.append(imageio.imread(file_path))
imageio.mimsave('output.gif', images, duration=5.55e-4)
