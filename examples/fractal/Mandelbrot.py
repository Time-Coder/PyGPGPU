import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel
from pygf import color3f
import matplotlib.pyplot as plt
import math


default_pixels_per_value = 200

def escape_steps(c: complex, max_iter: int) -> float:
    zn = 0j
    abs_zn = abs(zn)
    n: int = 0
    while abs_zn <= 2 and n < max_iter:
        zn = zn**2 + c
        abs_zn = abs(zn)
        n += 1

    if n == max_iter or abs_zn <= 1.0:
        return float(n)

    smooth = n + 1 - np.log2(np.log2(abs_zn))
    return smooth

def smooth_jet_colormap(x: float)->color3f:
    return color3f(
        math.exp( - 60 * abs(x-3/4)**3.2 ),
        math.exp( - 60 * abs(x-2/4)**3.2 ),
        math.exp( - 60 * abs(x-1/4)**3.2 )
    )

@kernel(float=np.float64)
def draw_MandelbrotSet(image:NDArray[np.uint8], view_center:complex, scale:float):
    value_per_pixel = 1 / (scale * default_pixels_per_value)
    left_top_complex = complex(
        view_center.real - 0.5*value_per_pixel*image.shape[1],
        view_center.imag + 0.5*value_per_pixel*image.shape[0]
    )
    max_iter: int = int(100 * np.log10(1 + scale) + 500)
    for i, j in np.ndindex(image.shape[0], image.shape[1]):
        c = complex(
            left_top_complex.real + j*value_per_pixel,
            left_top_complex.imag - i*value_per_pixel
        )
        steps = escape_steps(c, max_iter)
        color = smooth_jet_colormap(steps/max_iter)
        image[i, j, 0] = 255 * color.r
        image[i, j, 1] = 255 * color.g
        image[i, j, 2] = 255 * color.b

image = np.zeros((600, 800, 3), dtype=np.uint8)
draw_MandelbrotSet(image, -0.7618+0.0819j, 5900)

plt.rcParams['toolbar'] = 'None'
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(image)
ax.axis('off')
plt.show()