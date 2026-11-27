<div align="center">

<img src="https://raw.githubusercontent.com/Time-Coder/PyGPGPU/main/docs/logo/pygpgpu_256.png" alt="PyGPGPU logo" width="200"/>

# PyGPGPU

**English** | [Chinese](README.zh-CN.md)

</div>

Decorate a Python function with `@kernel` and it runs on the GPU — one codebase, four backends (**OpenCL / CUDA / HIP / Metal**), plus a serial interpreter backend for the CPU.

```python
import numpy as np
from pygpgpu import kernel

@kernel
def saxpy(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

    return out

x = np.arange(1_000_000, dtype=np.float32)

out = saxpy(x, x, 2.0)              # no backend given: probe in DEFAULT_BACKEND_ORDER (Metal first on macOS)
out = saxpy["cuda:0"](x, x, 2.0)    # the bracket switches backend per call
out = saxpy["hip:0"](x, x, 2.0)     # backend:device
out = saxpy["opencl:0"](x, x, 2.0)
out = saxpy["python"](x, x, 2.0)    # the same function, serially in the interpreter
```

No GPU code to write, no memory copies to manage, no work sizes to compute — call it like an ordinary function and the top-level `for` loop is parallelized on the GPU. The first call translates, compiles and caches; every later call is free.

A whole Mandelbrot renderer in plain Python ([examples/fractal/Mandelbrot.py](examples/fractal/Mandelbrot.py)): the helpers are ordinary functions, the palette returns a `float3` vector, `@kernel(float=np.float64)` runs it in double precision, and the image is the return value:

![Mandelbrot set rendered on the GPU by examples/fractal/Mandelbrot.py](https://raw.githubusercontent.com/Time-Coder/PyGPGPU/main/docs/Mandelbrot.jpg)

```python
import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, float3
import imageio.v3 as iio
import math


def escape_steps(c: complex, max_iter: int) -> int:
    zn = 0j
    abs_zn = abs(zn)
    n: int = 0
    while abs_zn <= 2 and n < max_iter:
        zn = zn**2 + c
        abs_zn = abs(zn)
        n += 1

    return n

def palette(x: float)->float3:
    return float3(
        math.exp( - 60 * abs(x-3/4)**3.2 ),
        math.exp( - 60 * abs(x-2/4)**3.2 ),
        math.exp( - 60 * abs(x-1/4)**3.2 )
    )

image_width, image_height = 600, 400
default_pixels_per_value = 150
max_iter = 890

@kernel(float=np.float64)
def draw_MandelbrotSet(view_center: complex, scale: float) -> NDArray[np.uint8]:
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    value_per_pixel = 1 / (scale * default_pixels_per_value)
    left_top_complex = complex(
        view_center.real - 0.5*value_per_pixel*image.shape[1],
        view_center.imag + 0.5*value_per_pixel*image.shape[0]
    )
    for i, j in np.ndindex(image.shape[0], image.shape[1]):
        c = complex(
            left_top_complex.real + j*value_per_pixel,
            left_top_complex.imag - i*value_per_pixel
        )
        steps = escape_steps(c, max_iter)
        color = palette(steps/max_iter)
        image[i, j, 0] = 255 * color.r
        image[i, j, 1] = 255 * color.g
        image[i, j, 2] = 255 * color.b

    return image

image = draw_MandelbrotSet(-0.74364 + 0.13183j, 8000)
iio.imwrite("Mandelbrot.jpg", image)
```

Already have handwritten kernels? PyGPGPU can also `import` `.cl`/`.cu`/`.hip`/`.metal` files like Python modules, then call the kernels inside them like ordinary Python functions.

## Installation

```bash
pip install pygpgpu
```

Python >= 3.7. Bring your own GPU runtime — any one of them is enough:

| Backend | What you need | Notes |
|---|---|---|
| OpenCL | GPU driver (usually already installed) | Nothing extra; works across NVIDIA/AMD/Intel |
| CUDA | NVIDIA driver (includes NVRTC) | The CUDA Toolkit is not required |
| HIP | AMD HIP SDK | Compiles through the SDK's clang; point `PYGPGPU_HIP_INCLUDES` at extra include dirs if needed |
| Metal | macOS on a real Apple GPU | Compiles through the framework's own runtime; a Mac VM has no Metal device |
| Python | nothing | Selected explicitly with `k["python"]`, never probed |

Without an explicit backend, `@kernel` probes **cuda -> hip -> metal -> opencl** and takes the first that enumerates a device — except on **macOS**, where **metal** leads (`metal -> cuda -> hip -> opencl`): Apple machines ship no CUDA or HIP runtime, and macOS's OpenCL is deprecated, so the native backend is what a Mac should find first.

Each backend reads its sources with a tree-sitter grammar, installed automatically as a dependency: `tree-sitter-opencl`, `tree-sitter-cuda37`, and [tree-sitter-metal](https://github.com/Time-Coder/tree-sitter-metal) (our own distribution — no MSL grammar was published on PyPI). All of them are built against the tree-sitter ABI that CPython 3.7 can load, so one install works from 3.7 through the current release.

## Basic usage

**One decorator for every kind of function.** `@kernel` decorates plain functions, instance methods and static methods; `self.xxx` attributes the body reads become kernel arguments automatically:

```python
class Layer:
    def __init__(self, weights, bias):
        self.weights = weights          # arrays and scalars both work
        self.bias = bias

    @kernel
    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)          # built on the host, above the loop
        for i in range(out.shape[0]):   # the loop bound IS the work size
            out[i] = x[i] * self.weights[i] + self.bias
        return out

    @staticmethod
    @kernel
    def double(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0
        return out

    @classmethod
    @kernel
    def scaled(cls, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for i in range(out.shape[0]):   # a cls member can be the loop extent
            out[i] = x[i] * cls.factor
        return out

x = np.ones(4, dtype=np.float32)
layer = Layer(np.arange(4, dtype=np.float32), 0.5)

layer.apply(x)           # -> array([0.5, 1.5, 2.5, 3.5])
layer.apply["cuda"](x)   # brackets work on methods too
Layer.double(x)          # -> array([2., 2., 2., 2.])
Layer.factor = 3.0
Layer.scaled(x)          # cls IS the receiver: class members are the arguments
```

Type annotations are optional (they serve editors and type checkers, not translation): the device types come from the actual arguments at call time, and the kernel is specialized per type signature.

**A classmethod kernel takes the class as the receiver.** `@kernel` decorates `@classmethod` too, in either order — `cls` is the receiver, so `cls.<attr>` reads (scalars, arrays, even a loop extent) and `cls.<helper>(...)` device calls work exactly as `self.<attr>` does on an instance method. Both `@kernel @classmethod` and `@classmethod @kernel` bind the same way: the call is `Layer.scaled(x)`, `cls.version = 9` rebinds the class attribute and reaches the caller after the launch (an alias holding the class sees it), and `submit()`/`async_call()` settle it the same way.

**Output arrays are return values.** A kernel creates its result with numpy and `return`s it — the shape and dtype come from numpy itself, and the caller receives an ordinary ndarray; there is no `out` parameter to pre-allocate. An object created above the magic loop is built once on the host and shared by every work item; one created inside the loop is device-private per work item. Either can be returned, every `return` statement may have its own shape, and the first work item to return wins.

**Kernel parameters accept plain Python objects.** An object's members become kernel arguments — scalars, arrays, even nested objects (each level is read at call time):

```python
class Params:
    def __init__(self, a, b):
        self.a = a
        self.b = b

@kernel
def scale(p, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(out.shape[0]):
        out[i] = x[i] * p.a + p.b
    return out

scale(Params(2.0, 10.0), x)      # -> x * 2.0 + 10.0
```

**Kernels can call methods and helpers.** A helper is plain Python — no decorator, no annotations: it is translated into a device function and specialized at the call site, recursively for helpers that call helpers. Besides a module-level function called by its bare name, a kernel can call another method of its own class (`self.other_method`), a static method of any class, a method of an object passed as an argument, and a function reached through a module:

```python
def clamp01(v):                     # module-level helper
    return min(max(v, 0.0), 1.0)

class Scene:
    def hit(self, v):               # a method of an argument object
        return v * v

@kernel
def shade(x: np.ndarray, scene: Scene) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(out.shape[0]):
        out[i] = clamp01(scene.hit(x[i]))
    return out
```

**A kernel needs an iteration domain** for the work size to be derived from. Either use a magic loop (`for i in range(x.shape[0])`, whose bound is the domain; `np.ndindex`/`np.ndenumerate` for 2-D/3-D) or, when dispatching on `get_global_id`, name the array: `@kernel(domain="x")`. Without either the call raises instead of guessing.

**One bracket syntax** names the backend, the device and the work size — the same on every backend, in any order:

```python
k(args)                          # fully automatic
k["cuda"](args)                  # backend (opencl/cuda/hip/metal/python; aliases ocl/cu/rocm/amd/mtl/py)
k["opencl:gfx1036"](args)        # backend:device
k[(4, 256)](args)                # work size: (blocks per grid, threads per block)
k["cuda"][(4, 256)](args)        # combined; None means automatic
```

The work size is CUDA's `(blocks_per_grid, threads_per_block)` everywhere — OpenCL and Metal included. A block count never has to divide anything: the total is blocks * threads by construction.

**Every backend in one process.** Each backend/device pair is specialized and cached separately, so alternating between them is an extra dictionary lookup, not a recompile — one process can mix vendors, and `k["python"]` is always available as the interpreter fallback.

**Process-wide settings** live on one object, `pygpgpu.config` — there is no `set_...()` to look up:

```python
import pygpgpu

pygpgpu.config.print_info = False                    # progress logging
pygpgpu.config.type_map = {float: np.float64}        # the width a python float travels as
pygpgpu.config.default_device = ["cuda:0", "hip:0"]  # what a plain k(...) tries, in order
```

Every commonly used name is importable from the top level — `kernel`, `funcs`, vector types (`float4`, ...), device builtins (`get_global_id`, `atomic_add`, `barrier`, ...), `list_devices`/`get_device`, `config`, `kernel_import`, `inline` — so an editor completes `pygpgpu.` and finds them.

## Writing a kernel where it is launched

A **kernel block** is a `@kernel` body written at its call site, with no `def`. It is the same body — the same magic loop, the same host regions, the same copy policy, the same generated file and cache — so there is nothing new to learn about what a body means. What changes is where it is written and where its data comes from:

```python
from pygpgpu import inline, kernel

a = np.arange(1024, dtype=np.float32)
b = np.arange(1024, dtype=np.float32) * 2.0
out = np.zeros(1024, dtype=np.float32)

with inline, kernel(default_device="cuda"):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0 + b[i]

# `out` holds the result; not one line of the block ran in this frame
```

**The data comes from the scope, not from a call.** A `@kernel` body reads its values from its parameters, a closure cell or a module global. A block has no parameter list, so the names it reads from the enclosing function are what travel to the device — in the order the body first reads them — and a module global needs no capture at all, because the rebuilt body shares the module's own globals. The caller writes no argument list, which is the point: the block is one-shot, launched where it stands.

```python
factor = 2.5
with inline, kernel():
    for i in range(out.shape[0]):
        out[i] = a[i] * factor          # `factor` and `a` are captured; no parameters written

class Layer:
    def __init__(self, bias):
        self.bias = bias

    def apply(self, x, out):
        with inline, kernel():
            for i in range(out.shape[0]):
                out[i] = x[i] + self.bias   # `self` travels as the receiver structure
```

**The head carries options, in the decorator's own spelling.** `default_device=` (or its alias `device=`), `int=` and `float=` mean exactly what they mean on `@kernel`, and `blocks_per_grid=` / `threads_per_block=` state the work size from the head, in the same `(blocks, threads)` spelling a bracket uses:

```python
with inline, kernel(blocks_per_grid=8, threads_per_block=128):
    for i in range(out.shape[0]):
        out[i] = a[i]
```

**The body states its own domain, with a magic loop.** This is the one place a block is deliberately narrower than `@kernel`: it does not take `domain=`, and a gid-style body (`get_global_id`) is refused by name. The reason is that a block is written where it is launched, so the array whose shape gives the work size is already in scope — `for i in range(out.shape[0])` says the domain in the body itself, and there is nothing left for `domain=` to add. `@kernel(domain="x")` exists because a decorated body is written away from the arrays it runs over.

```python
# the block's domain: the loop bound IS the work size
with inline, kernel():
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0
```

**A `return` answers through the object `inline` binds,** because a `with` statement has no value of its own:

```python
with inline as result, kernel():
    for i in range(out.shape[0]):
        out[i] = a[i]
    return 42                 # the same return a @kernel body writes

print(result.value)           # 42
```

**`inline` must come first.** The block is suppressed by raising from the head and catching that signal in `inline.__exit__`, so `with kernel(...), inline:` enters the head before the catcher exists — that order is refused with a message saying so. `with inline:` on its own is refused too, since there would be nothing to launch.

**A block is one kernel per site.** Executing one block inside a loop builds its translation once and launches the same `PyKernel` each time, exactly as a decorated kernel does.

**What the spelling cannot mean is refused before translation.** A gid-style body (`get_global_id`) or a body with no magic loop at all, a `domain=` option, a `nonlocal` (the block is translated as a kernel body, so there is no enclosing scope to write — capture a one-element array instead), a top-level `break`/`continue` (they would target the caller's loop, which the block never enters), a one-line body, and an `await`/`yield` all raise a `SyntaxError` naming the fix. A reported error inside the body points at the block's own file and line, with the real source text under the caret.

## Random numbers

A kernel body draws the way it would on the host — `random.random()`, `random.uniform(0.0, 1.0)`, `np.random.normal(0.0, 1.0)` — and the **launch owns the seed and the work item**, so the body never passes either. That is the same contract `taichi.random` offers, and it is what a model function needs: a helper that draws takes no `tid` and no `seed`.

```python
import random
from pygpgpu import kernel

def jitter() -> float:                  # a helper draws with no seed and no work item
    return random.uniform(-0.1, 0.1)

@kernel
def sample(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = random.random() + jitter()
```

**Each work item gets its own stream.** The stream is seeded from the launch seed and the work item's own position, so two work items never share a sequence, and a work item's k-th draw is the same number whatever order the work items ran in. The draws are PCG32, the generator `taichi.random` uses.

**The seed is a process-wide setting** — `pygpgpu.config.random.seed`. Left alone, each process draws its own; pinned, the sequence replays exactly:

```python
pygpgpu.config.random.seed = 1234       # reproducible from here on
```

**The interpreter backend runs the body as written**, so a draw there is python's own `random`/`numpy.random` call — the same distribution, but python's sequence rather than the device's. That follows from the interpreter having no work items (a magic-loop body runs once, standing for the whole launch).

**What a kernel cannot do is refused by name.** A draw answers one number, so `np.random.random(3)` and a `size=` argument cannot be represented in a body; `np.random.seed(0)` is the launch's business, not a work item's; and a keyword argument (`np.random.uniform(0.0, high=1.0)`) is refused because a device call takes positional arguments only.

## Importing handwritten kernels

`kernel_import` is the import hook that turns existing kernel files into Python modules: kernels and struct types become module attributes, and a `.pyi` stub is generated beside the source so IDE completion works. Write `vecadd.cl`:

```c
__kernel void vecadd(__global const float* a, __global const float* b,
                     __global float* out, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;         // the launch size may be rounded up, so clamp
    out[i] = a[i] + b[i];
}
```

```python
from pygpgpu import kernel_import

kernel_import.install()
import vecadd                      # one .cl file = one module

out = np.empty_like(a)
vecadd.vecadd(a, a, out, np.int32(len(a)))                    # work size inferred
vecadd.vecadd["opencl"][(3907, 256)](a, a, out, np.int32(len(a)))   # backend + work size
```

A struct `typedef` in the source becomes a ctypes type (a `Pt` above would pass by value, or as an array of structures). Same-named `.cl`/`.cu`/`.hip` files in one package merge into a single module: the same parameter list is one kernel with one implementation per backend (the bracket picks it), different parameter lists become overloads chosen by the argument types, and a bracket naming a backend without an implementation raises `KeyError` at the bracket. ([06](examples/06_handwritten_kernels.py) imports one file; [15](examples/15_kernel_import.py) merges three.)

## Features

- **One decorator for every function** — plain functions, instance methods, static methods and classmethods all take `@kernel`; the `self.xxx`/`cls.xxx` attributes the body reads become kernel arguments automatically, with the class as the receiver for a classmethod. ([01](examples/01_basic_kernel.py), [03](examples/03_nd_loop_and_class_method.py))
- **Return values** — the result array is created in the body with numpy and `return`ed; the caller receives an ordinary ndarray. Scalars, vectors, structures and arrays all return, every `return` may have its own shape, and the first work item to return wins. ([11_returns_and_scope.py](examples/11_returns_and_scope.py))
- **Python objects as arguments** — pass a plain object and its members (scalars, arrays, nested objects) become kernel arguments, read at call time. ([14](examples/14_heterogeneous_data.py))
- **Python objects inside the kernel** — read nested members and call methods of an argument object (`scene.hit(x)`); together with `self.method(...)`, static methods of any class and module-level functions, the call spellings are exactly ordinary Python's. ([03](examples/03_nd_loop_and_class_method.py), [14](examples/14_heterogeneous_data.py))
- **Objects created inside the kernel** — `np.zeros`/`np.zeros_like` written above the magic loop are built once on the host and shared by every work item; written below the loop they are device-private per work item. Both can be `return`ed. ([14](examples/14_heterogeneous_data.py))
- **Helpers** — a helper called from a kernel needs no decorator and no annotations; it is specialized at the call site, recursively for helpers that call helpers. ([03](examples/03_nd_loop_and_class_method.py))
- **Recursive device functions** — a helper or method that calls itself is compiled into a loop automatically, on every backend: the tail call, a result the caller consumes (`r = f(...)` with statements after it, or `return g(f(...))` with the recursive call at any argument position), and structure arguments the walk replaces per level (the bounce ray of a ray tracer). A shape the loop cannot order — a wrapper reading a value that differs per level, or two self-calls in one expression — is refused with its reason; CUDA and HIP answer those shapes through native recursion. ([ray tracing](examples/ray_tarcing_in_one_weekend/render.py))
- **Outer-scope variables** — variables the body reads from outside become kernel parameters, so a changed value is visible on the next call; a scalar declared `nonlocal`/`global` is written back to its scope after the call. ([11_returns_and_scope.py](examples/11_returns_and_scope.py))
- **Non-blocking launches** — `submit()` (a `concurrent.futures.Future`) and `await k.async_call(...)` (an `asyncio.Future`) on the same decorated function; the brackets apply here too. ([10](examples/10_async_launch.py))
- **Random numbers** — a kernel body draws with the spellings it already knows (`random.random()`, `np.random.normal(...)`); the launch owns the seed and the work item, so a helper draws with no `tid` and no `seed`. One stream per work item, and `pygpgpu.config.random.seed` replays the sequence. ([21](examples/21_random_numbers.py))

## How it compares

The same computation in this library and three other GPU-from-Python projects ([Taichi](https://www.taichi-lang.org/), [Numba](https://numba.pydata.org/), [NVIDIA Warp](https://github.com/NVIDIA/warp)). The API gap is widest around the ways you write things: helpers stay ordinary Python, arguments can be plain Python objects, instance/static methods decorate directly, and output arrays are return values.

**PyGPGPU** — arrays in, an array returned; no launch geometry, no buffer management, and helpers need no decorator at all:

```python
from pygpgpu import kernel

def clamp01(v):                         # plain Python: no decorator, no annotations
    return min(max(v, 0.0), 1.0)

def weight(v):                          # helpers may call other helpers
    return clamp01(v * v) * 0.5 + 0.25

@kernel
def apply_all(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)              # output arrays are return values
    for i in range(out.shape[0]):       # loop bound = work size
        out[i] = weight(x[i])
    return out

out = apply_all(x)                      # numpy arrays as-is
out = apply_all["hip"](x)               # the same function, another backend
```

**Taichi** — declare fields first, then copy data in and out by hand; a helper needs `@ti.func` and every argument must be annotated:

```python
import taichi as ti
ti.init(arch=ti.gpu)

x = ti.field(ti.f32, n)
y = ti.field(ti.f32, n)
out = ti.field(ti.f32, n)
x.from_numpy(nx)
y.from_numpy(ny)

@ti.func                                  # helpers need their own decorator
def weight(v: ti.f32) -> ti.f32:
    return ti.max(0.0, ti.min(1.0, v * v)) * 0.5 + 0.25

@ti.kernel
def saxpy(a: ti.f32):
    for i in range(n):
        out[i] = x[i] * a + y[i] + weight(x[i])

saxpy(2.0)
res = out.to_numpy()
```

**Numba** — compute blocks/threads yourself, guard inside, manage device memory; callable helpers are compiled separately with `@cuda.jit(device=True)`:

```python
from numba import cuda

@cuda.jit(device=True)                    # helpers need device=True
def weight(v):
    return min(1.0, max(0.0, v * v)) * 0.5 + 0.25

@cuda.jit
def saxpy(x, y, out, a):
    i = cuda.grid(1)
    if i < out.size:                      # bounds guard is your job
        out[i] = x[i] * a + y[i] + weight(x[i])

d_x = cuda.to_device(nx)
d_y = cuda.to_device(ny)
d_out = cuda.device_array_like(nx)
threads = 256
saxpy[(n + threads - 1) // threads, threads](d_x, d_y, d_out, 2.0)
res = d_out.copy_to_host()
```

**Warp** — typed array parameters, the launch size is passed explicitly, and helpers need `@wp.func`:

```python
import warp as wp
wp.init()

@wp.func                                  # helpers need their own decorator
def weight(v: float) -> float:
    return wp.min(1.0, wp.max(0.0, v * v)) * 0.5 + 0.25

@wp.kernel
def saxpy(x: wp.array(dtype=float), y: wp.array(dtype=float),
          out: wp.array(dtype=float), a: float):
    i = wp.tid()
    out[i] = x[i] * a + y[i] + weight(x[i])

wp.launch(saxpy, dim=n, inputs=[x, y, out, 2.0])
```

| | PyGPGPU | Taichi | Numba | Warp |
|---|---|---|---|---|
| Input data | numpy arrays directly | wrap in `ti.field`, explicit copies | `cuda.to_device` + `copy_to_host` | wrap in `wp.array` |
| Result | **the kernel returns an ndarray** | `out.to_numpy()` by hand | `copy_to_host()` by hand | written into a `wp.array` you manage |
| Helper functions | **plain Python — no decorator, annotations optional** | `@ti.func` + annotated args | `@cuda.jit(device=True)` | `@wp.func` + annotated args |
| Python objects as arguments | **yes — members become kernel arguments** | no (arrays must be wrapped in `ti.field`) | no (scalars and device arrays only) | no (needs `wp.array`) |
| Accessing Python objects in a kernel | **yes — read nested members, call their methods** | no | no | no |
| Instance methods | **yes — `self.xxx` becomes arguments automatically** | instance methods in a `@ti.data_oriented` class | no | no |
| Static methods | **yes** | no (only those instance methods) | no | no |
| Backends in one process | **all, mixed freely via brackets** | one, fixed by `ti.init` | CUDA only | CUDA only |
| Backends | OpenCL, CUDA, HIP, Metal | CPU, CUDA, Vulkan, ... | CPU, CUDA | CPU, CUDA |
| Copy optimization | **uploads/read-backs skipped per argument** | no | no | no |

Where the others are ahead, worth knowing before you pick: automatic differentiation (Taichi, Warp), sparse data structures (Taichi), and throughput-oriented CPU execution — the `python` backend here is serial and meant for testing. This table was written from each project's documentation; their APIs move fast, so confirm against current docs before relying on a cell.

## More documentation

The numbered scripts in [examples/](examples/) are standalone end-to-end tours; [examples/README.md](examples/README.md) lists them all with each one's dependencies:

| Topic | Examples |
|---|---|
| `@kernel` basics, brackets, device selection | [01](examples/01_basic_kernel.py), [02](examples/02_backend_device_selection.py) |
| 2-D/3-D magic loops, class methods, helpers by name | [03](examples/03_nd_loop_and_class_method.py) |
| Every kind of kernel value: arguments, built objects, returns | [14](examples/14_heterogeneous_data.py) |
| Vector types and the dual-sided `funcs` | [04](examples/04_vector_types.py) |
| Device-resident `gnp` arrays, zero copy, atomics | [05](examples/05_gnp_zero_copy_and_atomics.py), [19](examples/19_device_resident_arrays.py) |
| Handwritten kernels: one file; a multi-backend module | [06](examples/06_handwritten_kernels.py), [15](examples/15_kernel_import.py) |
| Low-level OOP API | [07](examples/07_low_level_oop.py) |
| Comparing backends on one machine | [08](examples/08_compare_backends.py) |
| Atomics and shared-memory reduction | [09](examples/09_reduction.py) |
| Async launches; the pure-Python backend; returns and scope | [10](examples/10_async_launch.py), [11](examples/11_python_backend.py), [11_returns_and_scope.py](examples/11_returns_and_scope.py) |
| `print()` in a kernel; images | [12](examples/12_print_in_kernel.py), [13](examples/13_image.py) |
| Scientific math and `device_function` | [16](examples/16_scientific_math.py) |
| The Metal backend | [17](examples/17_metal_backend.py), [18](examples/18_metal_device_check.py) |

## License

MIT
