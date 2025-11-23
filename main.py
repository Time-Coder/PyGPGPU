from pygpgpu.opencl import image2d_t, sampler_t, int2
from typing import Any

import numpy as np
from numpy.typing import NDArray
from test import flip_y, test


# src_image = image2d_t("test.png")
# dest_image = image2d_t(np.zeros_like(src_image.data))

# flip_y(src_image, dest_image, sampler_t())


# dest_image.save("dest.png")


a = np.stack([int2()]*100)
test(a, 100)
print(a)
