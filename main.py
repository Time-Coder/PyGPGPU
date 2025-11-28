from pygpgpu.opencl import image2d_t, sampler_t, int2, float4
from typing import Any

import pygpgpu.numpy as np
from test import flipY


src_image = image2d_t("test.png")
dest_image = image2d_t(np.zeros_like(src_image.data))
s = sampler_t()
flipY["amd"](src_image, dest_image, s)
# dest_image.save("dest.png")