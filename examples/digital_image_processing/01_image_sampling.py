from pygpgpu import kernel
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import time

src_image: np.ndarray = data.coffee()
ratio: int = 20

dest_image = np.zeros(
    (int(src_image.shape[0]/ratio), int(src_image.shape[1]/ratio), src_image.shape[2]),
    dtype=np.int32
)

def mean(src_image: np.ndarray, i: int, j: int, k: int, ratio: int)->float:
    result: float = 0.0
    for sub_i in range(i*ratio, (i+1)*ratio):
        for sub_j in range(j*ratio, (j+1)*ratio):
            result += src_image[sub_i, sub_j, k]

    return result / ratio**2

@kernel
def down_sampling(src_image: np.ndarray, dest_image: np.ndarray, ratio: int):
    for i, j, k in np.ndindex(dest_image.shape):
        dest_image[i, j, k] = mean(src_image, i, j, k, ratio)

start_time = time.perf_counter()
down_sampling(src_image, dest_image, ratio)
stop_time = time.perf_counter()
time_elapsed = stop_time - start_time
print(f"{time_elapsed:.2} s")

plt.imshow(dest_image)
plt.show()