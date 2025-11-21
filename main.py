from pygpgpu.opencl.runtime import CL
from pygpgpu.opencl.oop import Platforms, Context
from pygpgpu.opencl import compile, int2, char2, image2d_t, sampler_t

import numpy as np
import imageio.v3 as iio
import sys

# CL.print_call = True

program = compile("test.cl", type_checked=False)
src_image = image2d_t("test.png")
dest_image = image2d_t(np.zeros_like(src_image.data))
program.gaussian_blur[Platforms[0].devices[0]](src_image, dest_image, sampler_t())
dest_image.save("dest.png")
