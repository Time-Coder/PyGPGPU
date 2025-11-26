from pygpgpu.opencl import image2d_t, sampler_t, int2, float4
from typing import Any

import numpy as np
from numpy.typing import NDArray
from test import update_particle, Particle, Point3D

particle = Particle()
particle.position = Point3D(1.0, 2.0, 3.0)
particle.velocity = Point3D(0.1, 0.2, 0.3)
particle.id = 42
particle.mass = 1.5

output = float4()

update_particle(output, particle)

print(output)