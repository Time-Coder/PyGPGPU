import numpy as np
from pygpgpu import (atomic_add, barrier, blockIdx, get_global_id, get_local_id, get_local_size,
                     kernel, zeros_per_group, work_size)


# Intensity histograms: one bin per possible uint8 value, so bin i counts the pixels whose
# intensity is exactly i. Fixed, because the pixel range is fixed.
HIST_BINS: int = 256
HIST_CHANNELS: int = 3


def mean(src_image: np.ndarray, i: int, j: int, k: int, ratio: int)->float:
    result: float = 0.0
    for sub_i in range(i*ratio, (i+1)*ratio):
        for sub_j in range(j*ratio, (j+1)*ratio):
            result += src_image[sub_i, sub_j, k]

    return result / ratio**2

@kernel
def down_sampling(src_image: np.ndarray, ratio:float)->np.ndarray:
    dest_image = np.zeros(
        (int(src_image.shape[0]/ratio), int(src_image.shape[1]/ratio), src_image.shape[2]),
        dtype=src_image.dtype
    )
    for i, j, k in np.ndindex(dest_image.shape):
        dest_image[i, j, k] = mean(src_image, i, j, k, ratio)

    return dest_image

@kernel
def quantify(src_image: np.ndarray, ratio: float):
    dest_image: np.ndarray = np.zeros_like(src_image)

    for i, j, k in np.ndindex(src_image.shape):
        dest_image[i, j, k] = int(src_image[i, j, k]/ratio)*ratio

    return dest_image

def exp_func(x:float, alpha:float)->float:
    if abs(alpha) < 1e-6:
        return x
    
    return (1 - np.exp(-alpha*x)) / (1 - np.exp(-alpha))

@kernel
def adjust_brightness(image:np.ndarray, alpha:float)->np.ndarray:
    dest_image:np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(image.shape):
        dest_image[i, j, k] = 255*exp_func(image[i, j, k]/255, alpha)

    return dest_image

@kernel
def scale(image:np.ndarray, alpha:float)->np.ndarray:
    dest_image:np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(image.shape):
        dest_image[i, j, k] = np.clip(alpha * image[i, j, k], 0, 255)

    return dest_image

@kernel
def swap_channels(image:np.ndarray, old_channel1:int, old_channel2:int, old_channel3:int)->np.ndarray:
    dest_image:np.ndarray = np.zeros_like(image)
    for i, j in np.ndindex(dest_image.shape[0], dest_image.shape[1]):
        dest_image[i, j, 0] = image[i, j, old_channel1]
        dest_image[i, j, 1] = image[i, j, old_channel2]
        dest_image[i, j, 2] = image[i, j, old_channel3]

    return dest_image

@kernel
def merge_images(image1:np.ndarray, image2:np.ndarray, weight1:float, weight2:float)->np.ndarray:
    dest_image:np.ndarray = np.zeros_like(image1)
    for i, j, k in np.ndindex(dest_image.shape):
        dest_image[i, j, k] = np.clip(weight1 * float(image1[i, j, k]) + weight2 * float(image2[i, j, k]), 0, 255)

    return dest_image

@kernel
def merge_gray_images(image1:np.ndarray, image2:np.ndarray, weight1:float, weight2:float)->np.ndarray:
    dest_image:np.ndarray = np.zeros_like(image1)
    for i, j in np.ndindex(dest_image.shape):
        dest_image[i, j] = np.clip(weight1 * float(image1[i, j]) + weight2 * float(image2[i, j]), 0, 255)

    return dest_image

@kernel
def gamma(image:np.ndarray, gamma_value:float):
    dest_image:np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(dest_image.shape):
        dest_image[i, j, k] = 255 * (image[i, j, k]/255) ** gamma_value

    return dest_image

@kernel
def hist(image:np.ndarray):
    """Intensity histogram of an RGB image: 256 bins per channel.

    Returns an ``int32`` array of shape ``(3, 256)``: ``result[channel, intensity]`` is the
    number of pixels of that channel whose value is exactly ``intensity`` (0..255), so
    ``result[k].sum() == image.shape[0] * image.shape[1]`` for every channel.

    The image is expected to be ``uint8`` with shape ``(height, width, 3)``. Counting goes
    through the work group's shared memory, the way a CUDA histogram does it: every group
    owns 768 bins (3 channels x 256), the pixels of the group land there, and each group
    merges its bins into the global histogram once instead of once per pixel. So the global
    array sees one atomic per bin per group rather than one per pixel.

    The strides are read from ``get_local_size`` instead of assuming a group width, so the
    kernel stays correct whatever group size the launch picks. The domain is 3-D, so the
    group spans three axes: ``lane`` flattens ``get_local_id`` into the one number that is
    unique inside the group (a CUDA block is 1-D there, which is why ``threadIdx.x`` alone
    is enough in the C version).
    """
    work_size(image)                       # one work item per channel sample
    size0 = get_local_size(0)
    size1 = get_local_size(1)
    lane = get_local_id(0) + size0 * (get_local_id(1) + size1 * get_local_id(2))
    group_total = size0 * size1 * get_local_size(2)
    bins = zeros_per_group(768, np.int32)            # __shared__ bins[3 * 256], per work group
    dest_hist:np.ndarray = np.zeros((3, 256), dtype=np.int32)

    # Clear the group's bins: each lane takes a strided slice of the 768 bins. The first
    # subscript names the group (`bins[blockIdx]`); on a device it selects storage the
    # hardware already gave this group, so no offset is emitted for it.
    b = lane
    while b < 768:
        bins[blockIdx][b] = 0
        b += group_total
    barrier()                              # __syncthreads()

    # Count in shared memory. k is the channel (the last axis of the image), so bin
    # (k, value) lives at k * HIST_BINS + value. int() because `k * HIST_BINS + uint8`
    # stays a uint8 under numpy's rules and overflows, while C promotes it to int.
    #
    # The launch rounds each extent up to a whole number of groups, so some work items fall
    # outside the image. They must not return early -- every thread of a group has to reach
    # both barriers -- so they simply contribute nothing.
    i = get_global_id(0)
    j = get_global_id(1)
    channel = get_global_id(2)
    if i < image.shape[0] and j < image.shape[1]:
        value = int(image[i, j, channel])
        atomic_add(bins[blockIdx], channel * HIST_BINS + value, 1)
    barrier()                              # __syncthreads()

    # Merge this group's bins into the global histogram, once per bin.
    b = lane
    while b < 768:
        atomic_add(dest_hist, b, bins[blockIdx][b])
        b += group_total

    return dest_hist