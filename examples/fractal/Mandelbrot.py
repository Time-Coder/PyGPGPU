import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, float3
import imageio.v3 as iio
import math


def escape_steps(c: complex, max_iter: int) -> int:
    zn = 0j
    abs_zn = abs(zn)
    n: int = 0
    while abs_zn <= 2 and n < max_iter:
        zn = zn**2 + c
        abs_zn = abs(zn)
        n += 1

    return n

def palette(x: float)->float3:
    return float3(
        math.exp( - 60 * abs(x-3/4)**3.2 ),
        math.exp( - 60 * abs(x-2/4)**3.2 ),
        math.exp( - 60 * abs(x-1/4)**3.2 )
    )

image_width, image_height = 600, 400
default_pixels_per_value = 150
max_iter = 890

@kernel(float=np.float32)
def draw_MandelbrotSet(view_center: complex, scale: float) -> NDArray[np.uint8]:
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
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
        steps = escape_steps(c, max_iter)
        color = palette(steps/max_iter)
        image[i, j, 0] = 255 * color.r
        image[i, j, 1] = 255 * color.g
        image[i, j, 2] = 255 * color.b

    return image

image = draw_MandelbrotSet(-0.74364 + 0.13183j, 8000)
iio.imwrite("Mandelbrot.jpg", image)
