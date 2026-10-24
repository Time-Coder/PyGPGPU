import numpy as np

from pygpgpu.opencl import climport

climport.install()
climport.config()

from pygpgpu.opencl.driver import CL

CL.print_info = True

import test

N = 8
points = np.array([[x, 0.5] for x in range(N)], dtype=np.float32)
results = np.full(N, -999, dtype=np.int32)
circle = test.Circle(center=test.Point(x=0.5, y=0.5), radius=2.0)

# 'points' 只读 -> 只上传; 'results' 只写(静态分析得出) -> 只回读
test.check_points_in_circle(points, results, circle, np.int32(N))
print("results:", results)
