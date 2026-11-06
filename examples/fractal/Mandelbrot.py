import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel
import matplotlib.pyplot as plt


default_pixels_per_value = 100

def escape_steps(c: complex, max_iter: int) -> float:
    zn = 0j
    abs_zn = abs(zn)
    n: int = 0
    while abs_zn <= 2 and n < max_iter:
        zn = zn**2 + c
        abs_zn = abs(zn)
        n += 1

    if n == max_iter:
        return float(max_iter)

    if abs_zn <= 1.0:
        return float(n)

    smooth = n + 1 - np.log2(np.log2(abs_zn))
    return smooth

@kernel
def draw_MandelbrotSet(image:NDArray[np.float32], view_center:complex, scale:float):
    value_per_pixel = 1 / (scale * default_pixels_per_value)
    left_top_complex = complex(
        view_center.real - 0.5*value_per_pixel*image.shape[1],
        view_center.imag + 0.5*value_per_pixel*image.shape[0]
    )
    max_iter: int = np.log10(1 + scale) + 500
    for i, j in np.ndindex(image.shape[0], image.shape[1]):
        c = complex(
            left_top_complex.real + j*value_per_pixel,
            left_top_complex.imag - i*value_per_pixel
        )
        steps = escape_steps(c, max_iter)
        image[i, j] = steps/max_iter

image = np.zeros((600, 800), dtype=np.float32)
draw_MandelbrotSet(image, -0.74355+0.1268j, 20)

plt.rcParams['toolbar'] = 'None'
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(image, cmap='viridis')
ax.axis('off')
plt.show()