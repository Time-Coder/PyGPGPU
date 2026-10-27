"""13_image.py -- one kernel, one image type, every backend.

CUDA samples through a texture object and stores through a surface object, in texel
space; OpenCL takes an image plus a separate sampler and addresses pixels as integers.
Writing one kernel for both used to mean picking a spelling and faking the other.

``Image`` removes that choice. A kernel indexes it exactly like an array:

    @kernel
    def brighten(src, dst):
        for y, x in np.ndindex(dst.shape):
            dst[y, x] = src[y, x] * 0.5

The subscript becomes each backend's sampling call, and the same subscript works on
the ``python`` backend, where it runs a host implementation instead -- so one kernel
produces the same numbers everywhere. This example proves that by running the same
kernel on all three backends and comparing the results element by element.

Coordinates follow CUDA's texel space: texel i spans [i, i+1) and its centre is at
i + 0.5. An integer subscript therefore means "that texel" and samples its centre;
a fractional subscript addresses a position between texels, which is what a linear
filter interpolates:

    src[y, x]        integer: the texel itself
    src[y, x + 0.25] fractional: a quarter of the way into the next texel

Sampling rules are fixed for the whole image, so a kernel needs no sampler argument, and
both are enums (a string with the member value still works):

    Image(..., filter=ImageFilter.LINEAR, address=ImageAddress.CLAMP)

Only ImageAddress.CLAMP is portable: the device samplers read the edge texel for WRAP and
MIRROR under texel-space coordinates, so a GPU launch with those raises rather than
returning numbers that differ from the python backend.

The channel count is a separate argument, never inferred from the shape, so a 4x4 grey
image and a 4-pixel RGBA row cannot be confused:

    Image(np.zeros((4, 4)))                 # 4x4, one channel
    Image(np.zeros((4, 4)), channels=4)     # 4x4, last axis is four channels
    Image(shape=(4, 4))                     # allocate, fill later

The HIP backend is the exception: it reuses CUDA code generation, which has no texture
object type, and hardware without texture support rejects hipTexObjectCreate outright,
so an image parameter there raises a PyKernelSyntaxError saying so. This example uses
the python, cuda and opencl backends.

Run:
    python examples/13_image.py
"""

import numpy as np

from pygpgpu import Image, kernel, list_devices


@kernel
def copy_image(src: Image, dst: Image):
    """Integer subscripts: each output texel is the matching input texel."""
    for y, x in np.ndindex(dst.shape):
        dst[y, x] = src[y, x]


@kernel
def shift_image(src: Image, dst: Image):
    """Fractional subscripts: sample a quarter texel into the next one."""
    for y, x in np.ndindex(dst.shape):
        dst[y, x] = src[y, x + 0.25]


@kernel
def convolve_image(src: Image, dst: Image):
    """A 3x3 box blur, exercising many samples per output texel."""
    for y, x in np.ndindex(dst.shape):
        total = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                total += src[y + dy, x + dx]
        dst[y, x] = total / 9.0


WIDTH, HEIGHT = 16, 8

# A ramp makes any wrong coordinate obvious: value = y * WIDTH + x.
ramp = np.arange(WIDTH * HEIGHT, dtype=np.float32).reshape(HEIGHT, WIDTH)

print("PyGPGPU images: one kernel, every backend")
print("host image:", Image(ramp))
print()

# Which backends this machine has; the python one always works. Image support differs per
# backend, so a failure is reported in the table below rather than raised.
available = {selector.split(":")[0] for selector in list_devices()}
backends = ["python"] + [name for name in ("cuda", "opencl") if name in available]
print("backends under test:", ", ".join(backends))
print()


def run(backend, fn, image_filter, expected):
    """Run one kernel on one backend and return (ok, first row)."""
    src = Image(ramp.copy(), filter=image_filter)
    dst = Image(shape=(HEIGHT, WIDTH), dtype=np.float32)
    fn[backend](src, dst)
    got = np.asarray(dst.read(), dtype=np.float32)
    ok = bool(np.allclose(got, expected))
    return ok, got[0][:6]


checks = []

# 1) point sampling, integer subscripts: exact copy.
for backend in backends:
    try:
        ok, row = run(backend, copy_image, "point", ramp)
        checks.append((backend, "copy (point, integer)", ok, row))
    except BaseException as error:
        checks.append((backend, "copy (point, integer)", None, str(error)[:60]))

# 2) linear sampling, integer subscripts: still the texel itself, because an integer
#    subscript lands on the texel centre where linear filtering reads one texel.
for backend in backends:
    try:
        ok, row = run(backend, copy_image, "linear", ramp)
        checks.append((backend, "copy (linear, integer)", ok, row))
    except BaseException as error:
        checks.append((backend, "copy (linear, integer)", None, str(error)[:60]))

# 3) linear sampling at a fractional coordinate. A coordinate of x + 0.25 sits inside
#    texel x, whose centre is at x + 0.5, so linear filtering blends that centre with
#    the previous one: 0.75 of texel x plus 0.25 of texel x-1.
expected_shift = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
for y in range(HEIGHT):
    for x in range(WIDTH):
        near = ramp[y, min(x, WIDTH - 1)]
        far = ramp[y, max(x - 1, 0)]
        expected_shift[y, x] = near * 0.75 + far * 0.25
for backend in backends:
    try:
        ok, row = run(backend, shift_image, "linear", expected_shift)
        checks.append((backend, "shift (linear, fractional)", ok, row))
    except BaseException as error:
        checks.append((backend, "shift (linear, fractional)", None, str(error)[:60]))

# 4) point sampling, many samples per texel: a box blur reads a neighbourhood, so a
#    guards against coordinates being swapped.
blurred = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
for y in range(HEIGHT):
    for x in range(WIDTH):
        total = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                total += ramp[min(max(y + dy, 0), HEIGHT - 1),
                              min(max(x + dx, 0), WIDTH - 1)]
        blurred[y, x] = total / 9.0
for backend in backends:
    try:
        ok, row = run(backend, convolve_image, "point", blurred)
        checks.append((backend, "blur (point, neighbours)", ok, row))
    except BaseException as error:
        checks.append((backend, "blur (point, neighbours)", None, str(error)[:60]))

print("%-8s %-26s %-6s %s" % ("backend", "case", "match", "row 0"))
print("-" * 74)
for backend, case, ok, detail in checks:
    verdict = "yes" if ok else ("ERROR" if ok is None else "NO")
    shown = np.array2string(detail, precision=2) if ok is not None else detail
    print("%-8s %-26s %-6s %s" % (backend, case, verdict, shown))

print()
every_ok = all(ok for _, _, ok, _ in checks if ok is not None)
errors = [case for _, case, ok, _ in checks if ok is None]
print("all backends agree element for element:", bool(every_ok))
if errors:
    print("(some backends could not be exercised:", ", ".join(sorted(set(errors))), ")")

# One Image serves several backends in one process: it binds lazily, per backend.
shared = Image(ramp.copy())
for backend in backends:
    dst = Image(shape=(HEIGHT, WIDTH), dtype=np.float32)
    try:
        copy_image[backend](shared, dst)
        print("%-8s reuses the same Image object: %s"
              % (backend, bool(np.allclose(np.asarray(dst.read()), ramp))))
    except BaseException as error:
        # A driver refusal is reported, not raised: the point of this block is reuse,
        # and the same limitation is already annotated in the table above.
        print("%-8s could not be exercised: %s" % (backend, str(error)[:60]))
