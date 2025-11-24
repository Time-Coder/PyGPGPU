from pygpgpu.opencl import image2d_t, sampler_t, int2
from typing import Any

import numpy as np
from numpy.typing import NDArray
from test import Point, compute_distance

output = np.array([0]*10, dtype=np.float32)
compute_distance(output, 10, np.array(Point(1, 1)))
print(output)