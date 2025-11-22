from pygpgpu.opencl import image2d_t, sampler_t

import numpy as np
from test import flip_y


src_image = image2d_t("test.png")
dest_image = image2d_t(np.zeros_like(src_image.data))

flip_y(src_image, dest_image, sampler_t())

dest_image.save("dest.png")
