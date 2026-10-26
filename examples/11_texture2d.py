"""11_texture2d.py -- sample a 2D image in a CUDA kernel with tex2D.

An OpenCL kernel takes an ``image2d_t`` and a sampler; CUDA splits that into an
*array* (the storage the texture hardware reads) and a *texture object* (the
array plus its sampling rules). PyGPGPU exposes both through the context:

    array = ctx.create_array(width, height, np.float32)
    array.set_data(host_image)
    texture = ctx.create_texture(array, normalized=False)

The texture is then passed to a kernel as an opaque handle and sampled with
``tex2D<float>(tex, x, y)``. Coordinates are pixel indices when the texture is
created with ``normalized=False``, which keeps the kernel arithmetic the same as
for any other integer pixel loop.

Note: this example needs a driver whose cuTexObjectCreate works under the running
interpreter. On the machine this was developed on, CUDA 13.4 accepts roughly nine
out of ten texture creations under CPython 3.14 and none at all under CPython
3.7; the array APIs themselves work on both. The example reports a clear message
rather than a bare traceback when the driver refuses.

Run:
    python examples/11_texture2d.py
"""
import importlib
import os
import sys
import tempfile

import numpy as np

from pygpgpu import kernel
from pygpgpu import cuimport
from pygpgpu.cuda.driver import CUDA
from pygpgpu.cuda.oop.context import Context

CUDA.print_info = False

# A handwritten kernel: the texture analogue of the .cl examples.
SOURCE = """
typedef unsigned long long cudaTextureObject_t;

__global__ void sample_flat(cudaTextureObject_t tex, float *out, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) {
        int px = i % w;
        int py = i / w;
        out[i] = tex2D<float>(tex, px, py);
    }
}

__global__ void sample_2d(cudaTextureObject_t tex, float *out, int w, int h)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < w && py < h)
        out[py * w + px] = tex2D<float>(tex, px, py);
}
"""


@kernel
def warm(x, out, a):
    for i in range(out.shape[0]):
        out[i] = x[i] * a


def main():
    # A launch first, so the CUDA context exists.
    warmup = np.random.rand(1024).astype(np.float32)
    warm["cuda"](warmup, np.zeros_like(warmup), 2.0)

    context = Context.current()
    width, height = 16, 8

    ramp = np.arange(width * height, dtype=np.float32).reshape(height, width)
    scaled = ramp * 2.0 + 5.0

    # The array APIs work everywhere; the texture object depends on the driver
    # accepting cuTexObjectCreate under this interpreter.
    array_a = context.create_array(width, height, np.float32)
    array_a.set_data(ramp)
    print("array upload/read-back  :", bool(np.array_equal(array_a.to_numpy(), ramp)))

    try:
        texture_a = context.create_texture(array_a, normalized=False)
    except BaseException as error:
        print("this driver/interpreter combination refuses cuTexObjectCreate")
        print("(%s); the array APIs above still work" % type(error).__name__)
        return

    array_b = context.create_array(width, height, np.float32)
    array_b.set_data(scaled)
    texture_b = context.create_texture(array_b, normalized=False)

    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "texture2d.cu"), "w") as handle:
        handle.write(SOURCE)
    sys.path.insert(0, folder)
    cuimport.install()
    # Imported through importlib because the module is generated at run time, so a
    # static `import texture2d` would not resolve for a type checker.
    kernels = importlib.import_module("texture2d")

    out_a = np.zeros((height, width), dtype=np.float32)
    kernels.sample_flat[(width * height, 32)](texture_a, out_a,
                                              np.int32(width), np.int32(height))
    print("1D launch, pixel coords :", bool(np.array_equal(out_a, ramp)))

    out_b = np.zeros((height, width), dtype=np.float32)
    kernels.sample_2d[((width, height), (16, 8))](texture_b, out_b,
                                                  np.int32(width), np.int32(height))
    print("2D grid, block/thread   :", bool(np.array_equal(out_b, scaled)))

    print("sample of row 0         :", out_a[0][:6])


if __name__ == "__main__":
    main()

