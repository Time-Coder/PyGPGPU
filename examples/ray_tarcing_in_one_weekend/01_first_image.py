import numpy as np
from pygpgpu import kernel
from pygf import color3f
import matplotlib.pyplot as plt


def write_color(image:np.ndarray, i:int, j:int, color:color3f):
    image[i, j, 0] = 255 * color.r
    image[i, j, 1] = 255 * color.g
    image[i, j, 2] = 255 * color.b

@kernel
def first_image(width:int, height:int)->np.ndarray:
    image:np.ndarray = np.zeros((width, height, 3), dtype=np.uint8)

    for j, i in np.ndindex(image.shape[0], image.shape[1]):
        r = i / (width - 1)
        g = j / (height - 1)
        b = 0.0
        write_color(image, i, j, color3f(r, g, b))

    return image

image = first_image(256, 256)
plt.imshow(image)
plt.show()