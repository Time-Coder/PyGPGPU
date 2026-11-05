import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel
from pygf import color3f
import math
import matplotlib.pyplot as plt


default_pixels_per_value = 100

def escape_steps(c: complex)->int:
    zn = 0j
    abs_zn = abs(zn)
    steps: int = 0
    while abs_zn <= 2 and steps <= 50:
        zn = zn**2 + c
        abs_zn = abs(zn)
        steps += 1

    return steps

def smooth_jet_colormap(x: float)->color3f:
    return color3f(
        math.exp( - 60 * abs(x-3/4)**3.2 ),
        math.exp( - 60 * abs(x-2/4)**3.2 ),
        math.exp( - 60 * abs(x-1/4)**3.2 )
    )

@kernel
def draw_MandelbrotSet(image:NDArray[np.float32], view_center:complex, scale:float):
    value_per_pixel = 1 / (scale * default_pixels_per_value)
    left_top_complex = complex(
        view_center.real - 0.5*value_per_pixel*image.shape[1],
        view_center.imag + 0.5*value_per_pixel*image.shape[0]
    )
    for i, j in np.ndindex(image.shape[0], image.shape[1]):
        c = complex(
            left_top_complex.real + j*value_per_pixel,
            left_top_complex.imag - i*value_per_pixel
        )
        steps = escape_steps(c)
        color = smooth_jet_colormap(steps/50)
        image[i, j, 0] = color.r
        image[i, j, 1] = color.g
        image[i, j, 2] = color.b

image = np.zeros((400, 600, 3), dtype=np.float32)
draw_MandelbrotSet(image, -0.74355+0.1268j, 20)

plt.imshow(image)
plt.show()