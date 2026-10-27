"""11_texture2d.py -- read an image with a texture object, write one with a surface.

CUDA names the host objects after the device types, so a kernel signature and the
call site use the same spelling:

    texture = cudaTextureObject_t(image)      # kernel reads through tex2D
    surface = cudaSurfaceObject_t(shape=(width, height))

Both are built from what the host already has -- a file path, a numpy array, or just
a shape -- and take an image of 1, 2 or 3 spatial dimensions, optionally with 1, 2 or
4 channels:

    cudaTextureObject_t("photo.png")
    cudaTextureObject_t((8, 16))                  # 2D
    cudaTextureObject_t(np.zeros((3, 4, 8)))      # 3D
    cudaTextureObject_t(shape=(16, 8))

Pass either to a kernel like any other argument; the backend binds it to whichever
context the launch uses, so nothing here mentions a device.

A texture object is read-only by nature -- CUDA separates the two object kinds
precisely so a sampler cannot be stored through -- while a surface is readable and
writable, and the kernel's read/write analysis decides which copies happen.

Build every image before loading a module that samples or writes one: the driver
this was developed against stops accepting new texture and surface objects once such
a module is in the process.

Run:
    python examples/11_texture2d.py
"""
import importlib
import os
import sys
import tempfile

import numpy as np

from pygpgpu import kernel, cuimport
from pygpgpu.kernel import (surf1Dwrite, surf2Dread, surf2Dwrite,
                            surf3Dread, surf3Dwrite, tex2D)
from pygpgpu.cuda.driver import CUDA, cudaSurfaceObject_t, cudaTextureObject_t

CUDA.print_info = False

# A handwritten kernel: the texture analogue of the .cl examples.
SOURCE = """
typedef unsigned long long cudaSurfaceObject_t;

__global__ void sample_flat(cudaTextureObject_t tex, float *out, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) {
        int px = i % w;
        int py = i / w;
        float u = ((float)px + 0.5f) / (float)w;
        float v = ((float)py + 0.5f) / (float)h;
        out[i] = tex2D<float>(tex, u, v);
    }
}

__global__ void sample_volume(cudaTextureObject_t tex, float *out, int w, int h, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h * d) {
        int px = i % w;
        int py = (i / w) % h;
        int pz = i / (w * h);
        out[i] = tex3D<float>(tex, ((float)px + 0.5f) / (float)w,
                                   ((float)py + 0.5f) / (float)h,
                                   ((float)pz + 0.5f) / (float)d);
    }
}

__global__ void fill(cudaSurfaceObject_t surf, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) {
        int px = i % w;
        int py = i / w;
        surf2Dwrite<float>((float)(i) * 2.0f, surf, px * 4, py);
    }
}
"""


@kernel
def fill_surface1d(surf, width):
    """Write a ramp into a 1D surface."""
    for x in range(width):
        surf1Dwrite(surf, x * 3.0, x)


@kernel
def fill_surface3d(surf, width, height, depth):
    """Write a ramp into a 3D surface."""
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                surf3Dwrite(surf, ((z * height + y) * width + x) * 2.0, x, y, z)


@kernel
def read_surface3d(surf, out, width, height, depth):
    """Read a 3D surface back through surf3Dread."""
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                out[(z * height + y) * width + x] = surf3Dread(surf, x, y, z)


@kernel
def sample_python(tex, out, width, height):
    """The same sampling as ``sample_flat``, written as a Python kernel."""
    for y in range(height):
        for x in range(width):
            out[y * width + x] = tex2D(tex, (x + 0.5) / width, (y + 0.5) / height)


@kernel
def fill_python(surf, width, height):
    """Write a ramp into the surface from a Python kernel."""
    for y in range(height):
        for x in range(width):
            surf2Dwrite(surf, (y * width + x) * 2.0, x, y)


@kernel
def read_python(surf, out, width, height):
    """Read the surface back through surf2Dread."""
    for y in range(height):
        for x in range(width):
            out[y * width + x] = surf2Dread(surf, x, y)


def main():
    width, height = 16, 8
    ramp = np.arange(width * height, dtype=np.float32).reshape(height, width)
    trailing = ramp * 2.0 + 5.0

    depth = 3
    volume = np.arange(width * height * depth, dtype=np.float32).reshape(
        depth, height, width)

    # Every image is built before the module below is imported.
    texture_a = cudaTextureObject_t(ramp)
    texture_b = cudaTextureObject_t(trailing)
    texture_3d = cudaTextureObject_t(volume)

    # A surface has to be created before a module using tex2D/surf2Dwrite loads as
    # well, since both are "new device texture object" calls to the driver.
    surface = cudaSurfaceObject_t(shape=(width, height))
    surface1d = cudaSurfaceObject_t(shape=(64,))
    surface3d = cudaSurfaceObject_t(shape=(width, height, depth))

    print("textures                :", texture_a, texture_3d)
    print("surfaces                :", surface, surface1d, surface3d)

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
    print("2D texture, .cu         :", bool(np.array_equal(out_a, ramp)))

    out_b = np.zeros((height, width), dtype=np.float32)
    kernels.sample_flat[(width * height, 32)](texture_b, out_b,
                                              np.int32(width), np.int32(height))
    print("second texture, .cu     :", bool(np.array_equal(out_b, trailing)))

    out_v = np.zeros(width * height * depth, dtype=np.float32)
    kernels.sample_volume[(width * height * depth, 32)](
        texture_3d, out_v, np.int32(width), np.int32(height), np.int32(depth))
    print("3D texture, tex3D       :", bool(
        np.array_equal(out_v.reshape(depth, height, width), volume)))

    out_c = np.zeros(width * height, dtype=np.float32)
    sample_python["cuda"](texture_a, out_c, np.int32(width), np.int32(height))
    print("2D texture, from Python :", bool(
        np.array_equal(out_c.reshape(height, width), ramp)))

    fill_python["cuda"](surface, np.int32(width), np.int32(height))
    expected = (np.arange(width * height, dtype=np.float32) * 2.0).reshape(height, width)
    print("surface write, Python   :", bool(np.array_equal(surface.read(), expected)))

    out_s = np.zeros(width * height, dtype=np.float32)
    read_python["cuda"](surface, out_s, np.int32(width), np.int32(height))
    print("surface read, surf2Dread:", bool(
        np.array_equal(out_s.reshape(height, width), expected)))

    fill_surface1d["cuda"](surface1d, np.int32(64))
    expect1d = np.arange(64, dtype=np.float32) * 3.0
    print("1D surface, surf1Dwrite :", bool(
        np.array_equal(surface1d.read(), expect1d)))

    fill_surface3d["cuda"](surface3d, np.int32(width), np.int32(height),
                          np.int32(depth))
    expect3d = (np.arange(width * height * depth, dtype=np.float32) * 2.0).reshape(
        depth, height, width)
    print("3D surface, surf3Dwrite :", bool(
        np.array_equal(surface3d.read(), expect3d)))

    out_v3 = np.zeros(width * height * depth, dtype=np.float32)
    read_surface3d["cuda"](surface3d, out_v3, np.int32(width), np.int32(height),
                           np.int32(depth))
    print("3D surface, surf3Dread  :", bool(
        np.array_equal(out_v3.reshape(depth, height, width), expect3d)))


if __name__ == "__main__":
    main()
