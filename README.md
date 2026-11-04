# PyGPGPU

**English** | [Chinese](README.zh-CN.md)

Decorate a Python function and it runs on the GPU — one codebase, three backends (**OpenCL / CUDA / HIP**).

```python
import numpy as np
from pygpgpu import kernel

@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

x = np.arange(1_000_000, dtype=np.float32)
out = np.zeros_like(x)

saxpy(x, x, out, 2.0)              # no backend given: probe cuda -> hip -> opencl, take the first available
saxpy["cuda"](x, x, out, 2.0)      # bracket switches the backend
saxpy["hip:gfx1036"](x, x, out, 2.0)   # backend:device
saxpy["python"](x, x, out, 2.0)    # the same function, serially in the interpreter (no GPU)
```

No C code to write, no build system to configure; the first call translates, compiles and caches, every later call is free. Existing handwritten kernels are fine too — `.cl`/`.cu`/`.hip` files can be `import`ed like Python modules.

## Features

- **`@kernel` decorator** — decorate a Python function directly (class methods and static methods included); at call time it is specialized per argument type, translated to backend source and compiled (monomorphization).
- **Plain-Python helper functions** — a helper called from a kernel needs no decorator, and its annotations are optional: it is translated into a device function and monomorphized from the call site, recursively for helpers that call helpers. The same holds for helpers reached through a name: `module.func(...)`, `SomeClass.static_method(...)`, `self.static_method(...)`, and `self.other_method(...)` between methods of one class.
- **All three backends in one process** — the bracket chooses the backend per call, so a single process (and even a single kernel object) can mix OpenCL, CUDA and HIP; each backend/device pair is specialized and cached separately.
- **Pure-Python backend** — `k["python"]` runs the same `@kernel` body in the interpreter, serially on the CPU: no GPU, no compiler, no runtime. Useful for testing, for small inputs and as a reference for what the GPU semantics mean.
- **Handwritten kernel file import** — `.cl` / `.cu` / `.hip` files import like Python modules: kernels and struct types become module attributes, and a `.pyi` type stub is generated automatically.
- **Low-level OOP API** — Device/Context/Buffer/Program/Kernel, for when you need full control.
- **One bracket syntax** for backend/device/work size (identical to handwritten kernels, order does not matter):
  ```python
  k(args)                          # fully automatic
  k["cuda"](args)                  # backend (opencl/cuda/hip/python, aliases ocl/cu/rocm/amd/py)
  k["opencl:gfx1036"](args)        # backend:device
  k[(4, 256)](args)                # work size: (blocks per grid, threads per block)
  k["cuda"][(4, 256)](args)        # combined
  ```
- **CUDA work-size brackets on every backend**: (blocks_per_grid, threads_per_block), the spelling a CUDA or HIP reader already writes. k[(4, 256)] launches 4 blocks of 256 threads (1024 work items); k[(n,)] is a bare item count with an automatic block size, and None means automatic. The same bracket works on every backend, in @kernel and in the low-level oop API, and a block count never has to divide anything: the total is locks * threads by construction.
- **Static read/write analysis**: the kernel body is parsed to infer whether each pointer argument is read-only, write-only or read-write — write-only arguments skip the upload, read-only arguments skip the read-back, and data stays on the device across calls (zero intermediate copies for gnp arrays).
- **Vector types**: 50 vector types (float4/int3/...) with `.x/.y/.z/.w` members, swizzles, construction, arithmetic (scalar broadcast), comparisons and component-wise math.
- **Dual-sided function library** `pygpgpu.funcs` (aka `pygpgpu.vec_types.funcs`): `dot/cross/normalize/clamp/...` run as-is in host Python and are translated to device builtins inside `@kernel`.
- **Device builtins, exported at the top level**: `get_global_id`, `barrier`, `atomic_add`, `dot`, ... are plain functions imported from `pygpgpu`. Inside a `@kernel` body the translator recognizes each one by the object it resolves to and maps it to the call of every backend, so the same kernel body runs on OpenCL, CUDA, HIP and in the interpreter.
- **Multi-dimensional iteration**: `np.ndindex(shape)` / `np.ndenumerate(arr)` magic loops (2D/3D), `enumerate(arr)` for a 1D index plus value snapshot.
- **Three cache layers**: parsed metadata / compiled artifacts / generated sources, all invalidated by mtime — unchanged sources mean zero recompilation.
- **Host-side CUDA libraries bound with ctypes** — cuBLAS, cuBLASLt, cuSOLVER, cuFFT, cuSPARSE and the CUDA runtime, in the same shape as the driver (lazy DLL, signature table, the library's own error messages), with a numpy-friendly layer over them (`gemm`, `axpy`, `scal`, `dot`, `rfft`, `csrmv`). See "Host-side CUDA libraries" below.
- **Host-side HIP libraries bound the same way** — hipBLAS and hipBLASLt, sharing the binding machinery with the CUDA side and deriving their signature tables from it. The *values* are hipBLAS's own, though (its `HIPBLAS_OP_N` is 111, not cuBLAS's 0), so the enums live here. See "Host-side HIP libraries" below.

## Installation

```bash
pip install pygpgpu
```

Python ≥ 3.7. Bring your own GPU runtime (any one of them is enough):

| Backend | What you need | Notes |
|---|---|---|
| OpenCL | GPU driver (usually already installed) | Nothing extra to install; works across NVIDIA/AMD/Intel |
| CUDA | NVIDIA driver (includes NVRTC) | The Toolkit is not required — the driver-provided nvrtc is enough |
| HIP | AMD HIP SDK | On Windows compilation runs through the SDK's clang; if MSVC/Windows SDK headers are not on the default paths, point `PYGPGPU_HIP_INCLUDES` at the include directories (os.pathsep-separated) |
| Python | nothing | Runs the `@kernel` body serially in the interpreter; selected explicitly with `k["python"]` and never probed for |

When `@kernel` has no explicit `backend`, backends are probed in **cuda -> hip -> opencl** order and the first one that enumerates a device wins.

## Quick start

Every commonly used API is importable straight from the `pygpgpu` top level — no need to remember submodule names:

```python
from pygpgpu import kernel, funcs, float4             # @kernel / dual-sided funcs / vector types
from pygpgpu import list_devices, get_device
from pygpgpu import kernel_import                    # the import hook for handwritten .cl/.cu/.hip files
```

The submodule paths (`pygpgpu.kernel` / `pygpgpu.vec_types` / `pygpgpu.opencl` / ...) still work. Note that top-level `pygpgpu.kernel` is the `@kernel` decorator; to reach things inside the submodule write `from pygpgpu.kernel import ...`.

`kernel_import` needs no GPU runtime to be imported or installed, and a merged kernel loads only its own backend's runtime when it is first called.

### The @kernel decorator

```python
import math
import numpy as np
from pygpgpu import kernel, funcs, float4

def norm2(v: float4) -> float:
    return funcs.dot(v, v)

@kernel
def normalize_all(vecs: np.ndarray, out: np.ndarray) -> None:
    for i, v in enumerate(vecs):          # 1D: index + float4 value snapshot
        out[i] = funcs.normalize(v)       # expanded component-wise on device

@kernel
def matrix_scale(m: np.ndarray, out: np.ndarray, k: float) -> None:
    for y, x in np.ndindex(out.shape):    # 2D NDRange
        out[y, x] = m[y, x] * k

@kernel(backend="cuda", domain="data")
def histogram(data: np.ndarray, counts: np.ndarray) -> None:
    i = get_global_id(0)
    atomic_add(counts, data[i], 1)    # atomics, results read back automatically
```

Supported: scalar/array/vector arguments, if/while/for-range, arithmetic/comparison/bit ops, `math.*` and `np.*` math functions, class methods (`self.xxx` attributes become arguments automatically), static methods, and calling helpers through a name — a module-level function, `module.func(...)`, `SomeClass.static_method(...)`, and `self.method(...)` for another method of the same class. Unsupported syntax raises a `SyntaxError` carrying the line number.

### `+=` into an array element

`arr[i] op= value` follows Python's semantics even when several work items meet at the same element: `bins[image[i, j, k]] += 1` accumulates every pixel, without writing `atomic_add` by hand. The translator emits a device atomic whenever it cannot prove the element belongs to one work item — the proof being that the subscript is built from exactly the magic loop's own index variables, so `out[i] += x[i]` stays a plain (faster) read-modify-write.

```python
@kernel
def histogram(image: np.ndarray):
    bins: np.ndarray = np.zeros((3, 256), dtype=np.int32)   # created on the host
    for i, j, k in np.ndindex(image.shape):
        bins[k, image[i, j, k]] += 1    # one atomic add per pixel: no lost counts
    return bins
```

Only operators with a device atomic counterpart are automated (`+=`, `-=`, `&=`, `|=`, `^=`). A racy `*=`/`/=`/`**=`, or a racy accumulation into a float element, raises a `SyntaxError` instead of losing data: there is no portable float atomic on the OpenCL backend, so use an integer buffer or accumulate per work item and reduce afterwards (`examples/09_reduction.py`).

### Accumulating into a single scalar

A handwritten kernel takes a pointer to one element (`__global int* total`) and passes it to `atomicAdd`. There are three equivalent shapes here:

```python
from pygpgpu import atomic_add, kernel

# 1) a 1-element array IS that pointer
@kernel
def count(x: np.ndarray, total: np.ndarray):
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            atomic_add(total, 0, 1)     # or simply: total[0] += 1

# 2) an outer scalar declared nonlocal/global becomes a device pointer,
#    and the result is written back to the enclosing scope after the call
@kernel
def count_global(x: np.ndarray):
    global TOTAL
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            TOTAL += 1                  # atomic on the device, native in the interpreter

# 3) a counter created in the body can be returned as the result
@kernel
def count_and_return(x: np.ndarray):
    total: np.ndarray = np.zeros(1, dtype=np.int32)
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            total[0] += 1
    return total                        # a 1-element array: the caller reads [0]
```

All three are exact on every backend — thousands of work items incrementing one location lose nothing. An outer scalar is always a *shared* location, so `TOTAL += 1` there is always atomic; `atomic_add(scalar, 0, 1)` is refused with a message pointing at that spelling, since an atomic needs an array to index.



Type annotations are optional: the snippets in this README carry them for readability, but the device types come from the arguments at call time, so the same code works unannotated (`self` is not a kernel argument, so annotating it changes nothing).

A kernel needs an iteration domain for its work size to be derived. Either use a magic loop (`for i in range(x.shape[0])`, whose bound is the domain) or, when dispatching on `get_global_id`, name the array in the decorator: `@kernel(domain="x")`. The domain contributes all of `x`'s dimensions, and a `k[(blocks, threads)]` bracket also works. Without any of these the call raises instead of guessing.

### Printing from a kernel

`print()` works inside a kernel on the OpenCL and CUDA backends: it is translated into device `printf` calls that write the same line Python would, so debugging a kernel needs no handwritten C.

```python
@kernel
def inspect(a: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        print("i =", i, "a[i] =", a[i], "ok =", a[i] > 0, "v =", float2(a[i], a[i] * 2))
        out[i] = a[i]
```

Arguments are separated by one space and the line ends with a newline, exactly as `print` does. Values render the way `str()` renders them: `True`/`False` for bools, plain decimals for integers, Python's shortest round-trip form for floats (`2.0`, `0.1`, `1e+20`), and `float4(1.0, 2.0, 3.0, 4.0)` for vectors. `None` prints as `None`, and literal strings become fixed text.

Three things to know:

- a line is emitted as **one `printf` per branch leaf**, selected by a computed index, because each device printf call site has its own output buffer — devices drain those buffers call site by call site, so a line split across several printf calls would come back interleaved with other work items. A bool argument therefore doubles the number of printfs for its line (one for `True`, one for `False`) and a float multiplies it by its digit budget; an unusually busy line hits a guard that asks you to split it;
- the order of the *lines* is the device's scheduling order, not the loop order, since work items run concurrently. A line as a whole stays intact; work items interleave between lines;
- **the HIP backend is excluded**: launching a kernel that carries the generated printf call sites makes the AMD driver drop the device on the ROCm 7.2 machine this was tested on (a Windows display-driver reset follows). `@kernel` raises for a `print()` in a kernel body when it is being compiled for HIP, so run the kernel with the `opencl` or `cuda` backend to read its output, or write results into an array instead.

The pure-Python backend (`k["python"]`) needs no translation: its `print` is the ordinary builtin, so keyword arguments and array printing work there too.

### Non-blocking launches

A decorated kernel takes the same `submit` / `async_call` methods as a handwritten one, so a `@kernel` function can be launched and awaited without writing any C:

```python
import asyncio
from pygpgpu import kernel

@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

# concurrent.futures
future = saxpy.submit(x, y, out, 2.0)
future.result()                       # resolves once the read-back has landed

# asyncio
await saxpy.async_call(x, y, out, 2.0)

# the brackets apply here too
saxpy["cuda"].submit(x, y, out, 2.0)
saxpy["opencl"][(len(x), 256)].async_call(x, y, out, 2.0)
```

The result is only safe to read from the future: the output arrays are written back by the completion callback, so touching them before the future resolves races with the transfer. `submit` and `async_call` differ only in the future type (`concurrent.futures.Future` vs `asyncio.Future`).

A launch needs an execution channel, and every backend pools it rather than creating one per call. On CUDA and HIP that channel is a stream: Blocking calls on the main thread ride the backend's default (NULL) stream and allocate nothing; blocking calls on other threads each cache a stream, held in a least-recently-used table capped at 16 per context, so spawning and joining many threads cannot accumulate streams. Asynchronous launches draw from a separate pool of at most 16 non-blocking streams: when all of them are in flight, `submit`/`async_call` returns immediately and finishes on a worker thread once a stream frees up, while a blocking call simply waits. A stream cached by a thread whose context has since gone away is dropped rather than reused, because a CUDA/HIP stream belongs to the context of the thread that created it.  OpenCL pools command queues the same way, with one difference: it has no default queue, so the main thread caches one too instead of riding a shared NULL channel. Queues are held in a least-recently-used table capped at 16 per context, the same take-over / re-key / cross-pool-borrow strategies apply, and asynchronous launches use a separate pool of at most 16. A hundred blocking launches now create no queue at all beyond the one already cached. Streams are also recycled across threads and between the two pools: a new thread takes over an idle cached stream instead of creating one, a full cache re-keys its least-recently-used entry onto the current thread instead of destroying it, and one pool drains the other's idle streams before anything new is created, so the two pools share a single budget of live streams.

### The pure-Python backend

`k["python"]` (aliases `"py"` and `"host"`) runs the decorated function in the interpreter instead of compiling it for a device. Nothing is translated, so the body does not have to stay inside what the GPU translator accepts, and nothing has to be installed: no driver, no cache directory, no device.

```python
@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

saxpy["python"](x, y, out, 2.0)      # ordinary Python, one call, serial
saxpy["py"].submit(x, y, out, 2.0)   # same launch on a thread pool
```

It is a correctness and convenience backend, not a speed one: the whole kernel runs on the CPU, serially. A launch can also be started with `submit` (a `concurrent.futures.Future`) or `async_call` (an `asyncio.Future`), both backed by a `ThreadPoolExecutor`; one launch is still serial, so concurrent launches must write to different arrays.

The two kernel styles keep their meanings, which is where the interpreter differs from a plain call:

| Kernel style | What `k["python"](...)` does |
|---|---|
| magic loop (`for i in range(x.shape[0])`, `enumerate(x)`, `np.ndindex`/`np.ndenumerate`) | runs the function once: the loop already covers every element. A `(blocks, threads)` bracket changes nothing here, because the loop, not the launch, decides the extent |
| gid-style (`get_global_id`), domain from `@kernel(domain="x")` | runs the body once per work item of the declared domain (`x`'s full shape), with `get_global_id` reporting the current index — one serial step per element, in `np.ndindex` order |
| gid-style, domain from a `k[(blocks, threads)]` bracket | same, one step per work item of the bracket, so the body must clamp itself against the real element count exactly as it does on a GPU |

Inside such a body the same functions resolve to their host implementations: the indexing builtins, the atomics (plain read-modify-writes — serial execution makes them exact), `barrier()` (a no-op) and the vector math family, which uses `pygpgpu.funcs`' host implementation. Two things cannot be emulated and raise instead: `local` (group-local memory is shared by parallel work items) and a `get_global_id` call inside a magic-loop body, where there is no current work item — use the loop variable there.

Vector-typed arrays work unchanged: reading `v = points[i]` yields a vector instance, and `out[i] = v.xyzy * 2.0` writes its components.

### One body, two behaviours: `KERNEL_TRANSLATING`

The interpreter runs the body as written, so a kernel that needs *different* code on each side has nowhere to put the difference — and a plain `if` on a value the device cannot see would be refused. `KERNEL_TRANSLATING` is the escape hatch, borrowed from `typing.TYPE_CHECKING`: it is `False` in Python and `True` while the body is being translated, so the condition is decided at **translation time** and exactly one branch is emitted:

```python
@kernel(domain="out")
def measure(data: np.ndarray, out: np.ndarray):
    i = get_global_id(0)
    if KERNEL_TRANSLATING:
        out[i] = data[i] * 2.0          # device: the real computation
    else:
        out[i] = len(str(i))            # python: a host-only expression the device could not run
```

The branch is inlined as a macro — no `if` survives into the generated source — and the dropped branch is never translated, so Python-only work costs the device nothing. Only the name is recognized, so a kernel may declare its own `KERNEL_TRANSLATING` instead of importing it.

### Handwritten kernel files

Already have C-style kernels? `.cl` / `.cu` / `.hip` files (native syntax of the matching backend) import like Python modules. Write a `vecadd.cl`:

```c
__kernel void vecadd(__global const float* a, __global const float* b,
                     __global float* out, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = a[i] + b[i];
}
```

```python
from pygpgpu import kernel_import
import numpy as np

kernel_import.install()
import vecadd                          # one .cl file = a module with one backend

a = np.random.rand(1_000_000).astype(np.float32)
out = np.zeros_like(a)
vecadd.vecadd(a, a, out, np.int32(len(a)))        # work size inferred
vecadd.vecadd[(3907, 256)](a, a, out, np.int32(len(a)))   # (blocks, threads)
```

Note the `n` argument: a handwritten kernel must clamp against the real element count, because the
launch size is rounded up to a multiple of the local size. `get_global_size(0)` reports that rounded
size, not the buffer length, so it is not a usable bound.

Struct definitions map to ctypes types; a `.pyi` stub is generated next to the kernel so IDE completion works.

### One kernel, several backends

Same-named `.cl` / `.cu` / `.hip` files are ONE module. Put `vecops.cl`, `vecops.cu` and
`vecops.hip` in a package directory (with an `__init__.py`), then:

```python
from pygpgpu import kernel_import

kernel_import.install()
from kernel_import import vecops        # the package that holds the three sources

vecops.add_offset(a, b, out, offset, n)      # default device, like a @kernel call
vecops.add_offset["cuda"](a, b, out, ...)    # a bracket names the backend
vecops.add_offset["opencl:1"][(blocks, threads)](...)   # device + work size, either order
```

What makes one module out of three files:

* a kernel declared with the **same parameter list** in several files is one function with one
  implementation per backend -- the bracket picks the implementation, the call keeps the kernel's
  own signature;
* declared with **different parameter lists** it is a set of overloads -- the stub declares one
  `@overload` per list, and the argument types pick the implementation at call time (a bare call
  walks the same `cuda -> hip -> opencl` probe a `@kernel` uses);
* declared in only one file it still exists, and a bracket naming a backend without an
  implementation raises `KeyError` at the bracket.

A `.pyi` stub is generated for the merged module: one `class Kernel_<name>` per kernel, one
`__call__` per distinct parameter list, and the structs from the first file that declares each.

### The scientific-computing vocabulary

The device-side math functions are plain names, like every other builtin:

```python
from pygpgpu import erf, tgamma, fast_exp, normal_cdf

@kernel
def integrate(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = erf(x[i]) * tgamma(x[i] + 1.0)
```

Three groups, and the difference matters:

* the functions **all three backends have** -- `erf`, `erfc`, `tgamma`, `lgamma`, `rint`,
  `remainder`, `fmin`, `fmax`, `fdim`, `copysign`, `nextafter`, `ldexp`, `fract`, `sign`,
  and the `native_*` family. These run on OpenCL, CUDA and HIP alike.
* the functions **CUDA and HIP have** and OpenCL does not -- the fast approximations
  (`fast_exp`, `fast_log`, `fast_sin`, ...), the correctly-rounded forms (`exact_recip`,
  `exact_divide`, `exact_sqrt`, ...), the special functions (`erf_inverse`, `normal_cdf`,
  `reciprocal_hypot`, `sin_pi`, `cos_pi`, the Bessel functions) and the conversions
  (`float_to_int_nearest`, `int_to_float`, ...). A kernel that uses one of these is a
  **CUDA or HIP kernel**: on OpenCL the translator refuses it *by name*, so the message
  says which function has no OpenCL counterpart rather than leaving you with an
  "undeclared identifier" from the compiler.
* the `mad` / `fma` family and the vector operations, which pygf already publishes and
  which are pygf's own objects here (see `pygpgpu.funcs`).

CUDA and HIP answer the same thing by the same name, but they are not the same device and
the difference is worth knowing:

* HIP has no `__exp2f`, so `native_exp2` is spelled `exp2f` there -- the ordinary name,
  which HIP does have.
* HIP has **none** of the round-toward-zero intrinsics (`__frcp_rz`, `__fdiv_rz`,
  `__fsqrt_rz`), so `exact_recip_rz`, `exact_divide_rz` and `exact_sqrt_rz` are computed
  on HIP by a device helper the package emits: it takes the correctly rounded value, uses
  the fused multiply-add to learn which side of it the exact result is on, and steps one
  unit toward zero when it is the other way. The result agrees with CUDA's own intrinsic
  bit for bit (a division is the one exception, because the two hardware divide units
  round differently in the last bit -- the rounding *direction* is the same either way).

Two names are worth a warning, because the device's documentation is the authority and
the spelling does not tell you:

* CUDA's `rhypotf` is the **reciprocal** of the hypotenuse (`1/sqrt(x²+y²)`), so the name
  here is `reciprocal_hypot`; the plain hypotenuse is `hypot`.
* CUDA's `sinpif` is `sin(πx)`, **not** the sinc function -- so the name here is `sin_pi`.

### Naming a device function that already exists

A device function that is not one of the builtins -- a `__device__` function in a header
you ship, or a compiler intrinsic with a spelling of your own -- is registered:

```python
from pygpgpu import device_function

@device_function(cuda="my_fast_rcp", opencl="native_recip",
                 headers={"cuda": "myclib.h"})
def my_fast_rcp(x: float) -> float:
    """The host's implementation; the device gets the symbol the registry names."""
    return 1.0 / x

@kernel
def use_it(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = my_fast_rcp(x[i])
```

The device calls the symbol the table names; the header is copied into the cache and
included in the generated source, so you do not have to know where the cache is (a header
given as a full `#include` line is carried into the source verbatim instead, which is what
a system header such as `#include <cuda_fp16.h>` needs). A backend with no symbol for the
function is refused by name, the same way the CUDA-only builtins are.

The body is the **host's** implementation -- the `python` backend runs it -- so the two
sides have to agree, exactly as they do for the builtins.

### Class methods

```python
class Model:
    def __init__(self) -> None:
        self.weights = np.random.rand(1024).astype(np.float32)
        self.lr = 0.01

    @kernel
    def apply(self, x: np.ndarray, out: np.ndarray) -> None:
        for i in range(out.shape[0]):
            out[i] = x[i] * self.weights[i] + self.lr

m = Model()
m.apply(x, out)                     # self.weights/self.lr are passed as kernel arguments
m.apply["cuda"](x, out)             # brackets still available
```

### Calling helpers by name

A kernel can call a helper written as ordinary Python. Besides a module-level function called by its bare name, a helper can be reached through a name:

```python
import mylib

def soften(v: float) -> float:            # module-level helper
    return v * 0.5

class Model:
    def __init__(self) -> None:
        self.level = 1.0            # read by gain(), so it becomes a kernel argument

    @staticmethod
    def scale(v: float) -> float:         # static method: no receiver
        return v * 3.0

    def gain(self) -> float:              # instance method: reads self.level
        return self.level + 1.0

    @kernel
    def apply(self, x: np.ndarray, out: np.ndarray) -> None:
        for i in range(out.shape[0]):
            out[i] = soften(x[i]) * self.scale(x[i]) + self.gain() + mylib.bias(x[i])
```

- `self.method(...)` calls another method of the same class. It is translated like any helper, except that the attributes *it* reads (`self.level` above) become kernel arguments too — recursively, through nested helper calls.
- `SomeClass.static_method(...)` and `self.static_method(...)` work for static methods of any class; they need no receiver.
- `module.func(...)` works for a plain function reached through a module object.
- Not supported from device code: calling a `@kernel`-decorated method, a `classmethod` (there is no `cls` on a device), or a method of *another instance* — a device function has no object model, so pass that object's arrays as explicit kernel arguments instead. Attributes cannot be written back either (`self.x = ...` is rejected; write into an array element).

### Copy optimization

Each pointer argument goes through function-body-level static analysis (`points[i] = ...` is a write, `x = a[i]` is a read, plus pointer-alias propagation and interprocedural analysis across functions):

```python
@kernel
def k(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i] * 2.0   # a read-only, b read-only, out write-only
# at runtime: a/b skip the read-back after upload, out skips the upload and only reads back
```

Combined with `pygpgpu.numpy` (gnp) arrays, data stays entirely on the device between consecutive kernel calls and is synchronized only when the host touches it.

## How it compares

The same computation, written in this library and in three other GPU-from-Python projects
([Taichi](https://www.taichi-lang.org/), [Numba](https://numba.pydata.org/),
[NVIDIA Warp](https://github.com/NVIDIA/warp)). Snippets are cut down to the parts that
differ — check each project's docs for exact syntax. The examples also show a shared helper
function, which is where the API gap is widest: here it stays ordinary Python, while the
others need a second decorator and typed signatures.

**PyGPGPU** — arrays in, arrays out; no launch geometry, no buffer management, and helper
functions need no decorator at all:

```python
from pygpgpu import kernel

def square(v):                          # plain Python: no decorator, no annotations needed
    return v * v

def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def weight(v):                          # helpers may call other helpers
    return clamp01(square(v)) * 0.5 + 0.25

@kernel
def apply_all(x: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):       # loop bound = work size
        out[i] = weight(x[i])

apply_all(x, out)                       # numpy arrays as-is
apply_all["hip"](x, out)                # the same function, another backend
```

**Taichi** — declare fields first, then copy data in and out by hand; a helper needs `@ti.func`
and every argument must be annotated:

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

**Numba** — compute blocks/threads yourself, guard inside, manage device memory; callable helpers
are compiled separately with `@cuda.jit(device=True)`:

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
| Input data | numpy arrays directly | wrap in `ti.field`, explicit `from_numpy`/`to_numpy` | `cuda.to_device` + `copy_to_host` by hand | wrap in `wp.array` |
| Helper functions | **plain Python — no decorator, annotations optional** | `@ti.func` + annotated args | `@cuda.jit(device=True)` | `@wp.func` + annotated args |
| Backends in one process | **all three, mixed freely** | one, fixed by `ti.init(arch=...)` | CUDA only | CUDA only |
| Launch size | inferred from the loop bound (or `domain="x"` / a bracket) | inferred from `range` | blocks × threads by hand, plus a guard | pass `dim=` |
| Bounds guard | emitted for you | emitted for you | you write it | you write it |
| Backends | OpenCL, CUDA, HIP | CPU, CUDA, Vulkan and others | CPU, CUDA | CPU, CUDA |
| AMD GPU | HIP (native AMD codegen) and OpenCL | via Vulkan | no | no |
| Switching backend | bracket `k["hip"]`, same source | `ti.init(arch=...)`, same source | not applicable | not applicable |
| Install footprint | Python + ctypes only: no LLVM, no CUDA Toolkit | ships its own runtime | needs LLVM | needs the CUDA Toolkit |
| Handwritten `.cl`/`.cu`/`.hip` | imported as Python modules, `.pyi` generated | no | via the `cuda` module's PTX/cubin path | no |
| Static read/write analysis | yes — uploads/read-backs skipped per argument | no | no | no |

### Several backends at the same time

Taichi fixes its backend once, at `ti.init(arch=...)`; Numba and Warp only target CUDA. Here the
bracket picks the backend per call, so one process — and even one kernel object — can use every
backend the machine has:

```python
@kernel
def apply_all(x: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = square(x[i]) + 1.0

apply_all["cuda"](x, out)      # NVIDIA
apply_all["hip"](x, out)       # AMD
apply_all["opencl"](x, out)    # whichever device OpenCL enumerates
apply_all["python"](x, out)    # the interpreter, on the CPU (no device needed)
apply_all(x, out)              # unqualified: probe cuda -> hip -> opencl
```

Each backend/device pair is specialized and cached separately, so alternating between them is just
an extra dictionary lookup — no recompilation after the first call. This is useful for comparing a
kernel across vendors on one machine (`examples/08_compare_backends.py`), or for falling back to a
second GPU when the first is busy.

Where the others are ahead — worth knowing before you pick:
- **Automatic differentiation**: Taichi and Warp can differentiate kernels; this library cannot.
- **Sparse and irregular data structures**: Taichi has sparse grids and SNode trees; here you pass dense numpy arrays (or handle sparsity yourself).
- **CPU fallback**: all three can run on the CPU; this library runs on the CPU only through the explicit `k["python"]` interpreter backend, which is serial and meant for testing rather than throughput.
- **Minimum interpreter version**: this library runs on Python 3.7 and newer, on both 64- and 32-bit Windows builds; the others require a newer interpreter.

This table was written from each project's documentation and the snippets above; their APIs move
fast, so confirm against current docs before relying on a specific cell.

### Return values and outer variables

A kernel runs once per work item, so there is no single C return value to hand back. The
translator gives every returned position a device parameter of its own — a one-element
buffer for a scalar or vector, the array object itself for an array — and the first work
item to reach a `return` claims the result with an atomic compare-and-swap. The host reads
the slots back after the launch and returns exactly what the winning work item returned:

```python
@kernel
def first_peak(x: np.ndarray):
    for i in range(x.shape[0]):
        if x[i] > 10.0:
            return x[i], i          # a tuple
    return None                     # a launch where nothing returns is None

value, index = first_peak[backend](x)
```

The supported return types are exactly the supported parameter types, every `return`
statement may have its own shape (`return x[i]` from one branch, `return x[i], i` from
another), and a bare `return` gives `None`. `submit()` and `async_call()` deliver the same
value once their future resolves, after read-back. A value-returning `return` is refused
below the magic loop: there it would return one work item's local, not the function's
result.

Variables the body reads from outside become parameters too, so a changed outer value is
visible on the next call — inlining it would leave the kernel stale:

```python
SCALE = 2.0

@kernel
def scaled(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * SCALE

scaled[backend](x, out)     # SCALE is passed as an argument
SCALE = 100.0               # the next call uses 100.0
scaled[backend](x, out)
```

A `nonlocal`/`global` declaration makes the outer scalar writable: it travels as a pointer,
every use inside is a dereference, and the value is copied back into its scope after the
call — a closure cell for `nonlocal`, the module globals for `global`. Outer arrays are
pointer parameters as well, and are read back when the kernel writes them. Rebinding an
outer array (as opposed to writing its elements) is refused, because the object is shared
with the caller. A helper function gets the same treatment as the kernel that calls it.

Statements above the magic loop may also *create* data with numpy — `np.zeros`, `np.ones`,
`np.empty`, `np.full`, `np.arange`, `np.linspace`, `*_like`, `np.float32(...)` and the rest
of the allocating constructors. The interpreter runs those statements, so the array has
exactly the shape and dtype numpy gives it, and it reaches the kernel as a parameter:

```python
@kernel
def normalized(x: np.ndarray, out: np.ndarray):
    total = np.zeros(1, dtype=np.float32)      # built on the host
    for i in range(x.shape[0]):
        total[0] = total[0] + x[i]
    for i in range(out.shape[0]):
        out[i] = x[i] / total[0]

@kernel
def doubled(x: np.ndarray):
    result = np.zeros_like(x)                  # created, then returned as itself
    for i in range(x.shape[0]):
        result[i] = x[i] * 2.0
    return result
```

A created object the body returns reuses the parameter it already travels through, so the
caller receives the host's own object rather than a copy. Creation below the magic loop is
refused: it would build one object per work item instead of one.

### Images

`Image` is one image type for every backend. A kernel indexes it like an array, and the
subscript becomes `tex2D`/`surf2Dwrite` on CUDA, `read_imagef`/`write_imagef` on OpenCL,
or a host implementation on the `python` backend, so one kernel gives the same numbers
everywhere:

```python
from pygpgpu import Image, kernel

@kernel
def brighten(src, dst):
    for y, x in np.ndindex(dst.shape):
        dst[y, x] = src[y, x] * 0.5

src = Image("photo.png", filter="linear")
dst = Image(shape=(height, width), dtype=np.float32)

brighten["cuda"](src, dst)      # or "opencl", or "python"
out = dst.read()
```

Construct it from a path, an array or a shape. The channel count is a separate argument
and is never inferred from the shape, so `(4, 4)` is a 4x4 grey image rather than a
4-pixel RGBA row:

```python
Image(np.zeros((4, 4)))                    # 4x4, 1 channel
Image(np.zeros((4, 4)), channels=4)        # 4x4, last axis is 4 channels
Image(shape=(4, 4))                        # allocate 4x4, fill later
Image(shape=(4, 4, 3), channels=2)         # 4x4x3 volume, 2 channels
```

Channels must be 1, 2 or 4 -- the widths a device sample can carry. `channels=3` is
refused with a message asking for padding to 4, because RGB is not a device image format.

Coordinates follow CUDA's texel space: texel i spans [i, i+1), so an integer subscript
samples that texel's centre and a fractional one addresses a position between texels,
which a linear filter blends. Both rules are fixed for the whole image, so no kernel
takes a sampler, and both are given as enums (a plain string with the member's value
still works):

```python
from pygpgpu import ImageFilter, ImageAddress

Image(arr, filter=ImageFilter.LINEAR, address=ImageAddress.CLAMP)
Image(arr, filter="linear")                       # same thing
```

`ImageFilter` is `POINT` or `LINEAR`. `ImageAddress` is `CLAMP`, `WRAP` or `MIRROR`, but
only `CLAMP` is portable: launching an image with `WRAP` or `MIRROR` on the cuda or
opencl backend raises, because the device samplers read the edge texel for those modes
under texel-space coordinates (measured on every GPU available while developing this) and
the result would differ from the `python` backend. The `python` backend implements all
three, so a kernel that needs wrap or mirror can run there.

The HIP backend refuses images: it reuses CUDA code generation, which has no texture
object type, and the driver rejects `hipTexObjectCreate` on hardware without texture
support.

## Host-side CUDA libraries

The CUDA *driver* (`pygpgpu/cuda/driver`) is the low-level API: contexts, streams,
modules, memory. The CUDA *math libraries* ship as separate DLLs next to it and do
the actual mathematics. They are bound with ctypes under `pygpgpu/cuda/libraries`, in
the same shape as the driver — a lazily loaded DLL, a signature table per library, and
an error code turned into a `RuntimeError` carrying the library's own message:

```python
from pygpgpu.cuda.libraries import driver_for

CUBLAS = driver_for("cublas")          # the cuBLAS entry points
CUFFT  = driver_for("cufft")           # the cuFFT entry points
```

| Library | Bound operations |
|---|---|
| **cuBLAS** | `gemm` (float32/float64), `axpy`, `scal`, `dot`, `GemmEx`, handle and stream management |
| **cuBLASLt** | the generic `matmul` with a heuristically chosen algorithm, matrix layouts, epilogues |
| **cuFFT** | `plan1d`, `exec` (R2C / C2R / C2C), plan destruction, stream binding |
| **cuSPARSE** | `csrmv` (createCsr / createDnVec / SpMV), dense and sparse descriptors |
| **cuSOLVER** | `potrf` (Cholesky), `gesvd` (SVD), `getrf` (LU), with their workspace queries |
| **CUDA runtime** | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, the device count and properties |

A thin, numpy-friendly layer sits on top of the raw entry points, so a call takes
arrays rather than device pointers:

```python
import numpy as np
from pygpgpu.cuda_libraries import gemm, axpy, dot, rfft, csrmv

a = np.random.rand(3, 4).astype(np.float32)
b = np.random.rand(4, 2).astype(np.float32)

c = gemm(a, b)                  # a @ b, on the GPU
d = axpy(2.0, a[0], b[0])       # elementwise
e = dot(a[0], b[0])             # a scalar
f = rfft(a[0])                  # n/2+1 complex numbers, like np.fft.rfft
g = csrmv(row_offsets, column_indices, values, x)   # a sparse product
```

The same names are reachable as attributes of the module, so both spellings work:

```python
from pygpgpu import cuda_libraries
cuda_libraries.gemm(a, b)

from pygpgpu.cuda_libraries import gemm
gemm(a, b)
```

The wrappers take numpy's **row-major** layout and return the same layout; cuBLAS
reads column-major, and the difference is handled inside, so `gemm(a, b)` is just
`a @ b`. The 1-D cases follow numpy's own rules for `@`, so a matrix and a vector
multiply without the caller reshaping anything.

**A CUDA context has to be current.** A library handle belongs to the context that
was current when it was created, so the wrappers make one current if none is — but
if you use the raw entry points, make one current yourself first:

```python
from pygpgpu import get_device

device = get_device("cuda:0")
device.default_context.make_current()      # before the first library call
```

Without it the call fails with `CUDA_ERROR_INVALID_CONTEXT`, and the message says
so.

**These are host-side operations, not device builtins.** A `gemm` moves whole arrays
to the device and runs one library call; a `@kernel` body is per-thread arithmetic.
That is why a `@kernel` cannot call `cuda_libraries.gemm` — the two are different
layers, and mixing them would mean a kernel launching a library call per work item.
The device-side counterparts (a per-element multiply, a per-element dot) are the
device builtins the top level already exports.

The libraries are **loaded lazily**: `import pygpgpu` loads no CUDA library at all,
and each library's DLL is loaded on the first call that needs it. A machine without
the CUDA toolkit still imports the package, and the failure surfaces at the call with
the loader's own error.

**cuSOLVER needs kernels the consumer cards do not have.** NVIDIA does not ship
cuSOLVER's dense factorization kernels for every architecture, and a card without
them fails at execution rather than at setup: the call returns a failure code and the
context reports an illegal memory access. Measured on the machine this was developed
on (an RTX 4070 Laptop, compute capability 8.9, CUDA 12.6), `potrf`, `getrf`,
`gesvd`, `gesvdj` and `syevj` all fail that way, with `compute-sanitizer` putting the
fault inside cuSOLVER's own DLL. The bindings are correct and the test skips with
that reason instead of failing.

## Host-side HIP libraries

hipBLAS and hipBLASLt are AMD's ports of cuBLAS and cuBLASLt, so they are bound the
same way and reached through the same kind of layer:

```python
from pygpgpu import hip_libraries

c = hip_libraries.gemm(a, b)          # a @ b, on the GPU
d = hip_libraries.axpy(2.0, x, y)     # y = 2*x + y
e = hip_libraries.dot(x, y)           # a scalar
```

The two natural spellings work, exactly as they do for CUDA:

```python
from pygpgpu import hip_libraries
hip_libraries.gemm(a, b)

from pygpgpu.hip_libraries import gemm
gemm(a, b)
```

`gemm`, `lt_gemm`, `axpy`, `scal` and `dot` take numpy arrays in numpy's row-major
layout and return the same layout — hipBLAS reads column-major, and the difference is
handled inside, so `gemm(a, b)` is just `a @ b`, including numpy's 1-D rules for `@`
and the `transpose_a`/`transpose_b` flags. `ensure_context`, `allocate`, `free`,
`to_device`, `to_host`, `synchronize`, `create_handle`, `destroy_handle` and
`set_stream` are there too, so the raw entry points are usable without a second
allocator: `allocate` goes through the HIP runtime the backend already uses.

**The API is cuBLAS's, but the values are not — and that is the trap.** hipBLAS's
`HIPBLAS_OP_N` is **111** where `CUBLAS_OP_N` is 0, its status codes are numbered
separately (`HIPBLAS_STATUS_ALLOC_FAILED` is 2 where cuBLAS's is 3), and its compute
types are a dense 0..10 where cuBLAS's are 64..77. So the enums and the status
messages here are hipBLAS's own (`pygpgpu/hip/libraries/_enums.py`), read from ROCm's
headers. Passing a cuBLAS value to a hipBLAS call makes it answer
`HIPBLAS_STATUS_INVALID_ENUM` for an ordinary gemm.

The **signature tables**, on the other hand, really are cuBLAS's: the argument lists
are identical, so they are derived from the CUDA ones rather than written out again,
and each entry is copied rather than shared (a shared entry would carry one library's
bound function into the other). Only the names change, and not uniformly: hipBLAS
drops the ABI version suffix CUDA 12 carries (`cublasSgemm_v2` is `hipblasSgemm`), and
hipBLASLt spells its entry points `hipblasLt*` even though its DLL is `hipblaslt.dll`.

**hipBLASLt needs a kernel library your device may not have.** rocBLASLt loads its
kernels from a Tensile file named after the architecture
(`TensileLibrary_lazy_<arch>.dat`), and ROCm ships them for the discrete GPUs only —
on the machine this was developed on (an integrated Radeon 610M, `gfx1036`) that file
does not exist, so every `lt_gemm` fails inside rocBLASLt while the binding itself is
fine. `gemm` does not depend on it. The message says so, because the status code is
`INVALID_VALUE` and points at the arguments otherwise.

**hipFFT, hipSPARSE and hipSOLVER are not bound.** Their entry points exist and load,
but their signatures are not cuFFT's / cuSPARSE's / cuSOLVER's in every case —
hipFFT's plan handle and its exec call in particular — so a table copied from the CUDA
side binds symbols it cannot then call. They will come with their own tables; the
binding machinery under `pygpgpu/libraries` is already shared between the two
backends, so only the tables are missing.

Like the CUDA ones, these are **host-side operations**: a `gemm` moves whole arrays to
the device and runs one library call, while a `@kernel` body is per-thread arithmetic.
A kernel body cannot call any of this, and the libraries are loaded lazily — `import
pygpgpu` loads no HIP library.

## Device selection

Three forms of device identifier inside the brackets (same on every backend):

| Form | Meaning |
|---|---|
| `"0"`, `"1"` | index in the backend's device list (OpenCL uses flat numbering across platforms) |
| `"gfx1036"`, `"geforce"` | case-insensitive substring match on name/vendor/**architecture name** |
| `"amd"`, `"nvidia"`, `"intel"` | vendor shorthand: matched against the vendor string the driver reports, so `opencl:amd` finds a device whose vendor is spelled out as `Advanced Micro Devices, Inc.` |
| `"geforce:1"` | when a name matches several devices, take the n-th one |

Helpers:

```python
from pygpgpu import list_devices, get_device

list_devices()          # every available "backend:device" string, e.g. ['cuda:0', 'hip:0', 'opencl:0']
dev = get_device("cuda:0")   # the Device object of that backend, directly
```

### Choosing the default device

A call without a bracket runs on the first usable backend in `cuda -> hip -> opencl`.
`set_default_device()` replaces that order:

```python
from pygpgpu import kernel, set_default_device

set_default_device("opencl:cpu", "cuda")     # try the CPU device, fall back to cuda
k(x, out)                                    # no bracket: walks that list

set_default_device("gfx1036")                # a bare device name: found on any backend
set_default_device("cuda", "opencl:0")
set_default_device()                         # no arguments: restore the built-in order
```

The same selector strings the brackets accept are used, and each entry is tried until one
resolves; an unusable entry is skipped. The list is read at call time, so it also applies
to kernels decorated before the call, and `get_default_device()` reports what it currently
is. A kernel decorated with an explicit `@kernel(backend=...)` keeps that backend.

## Caching

| Backend | Directory | Contents |
|---|---|---|
| OpenCL | `__clcache__/` | parsed meta + per-device binaries |
| CUDA | `__cucache__/` | parsed meta + per-device cubin/ptx |
| HIP | `__hipcache__/` | parsed meta + per-device code objects |
| Python | none | nothing is generated or compiled; the body runs as-is |

Caches are invalidated by the mtime of the source file (and every `#include`); sources generated by `@kernel` are keyed by a hash of the Python source. These directories should be in `.gitignore`.

## Progress output

Every backend logs what it does — translation, compilation, buffer copies and each kernel launch, prefixed with the backend it ran on:

```
launch OpenCL Kernel('pyk_saxpy_fb5e5907') on Device('NVIDIA GeForce RTX 4070 Laptop GPU')
       with global_work_size=(1024,), local_work_size=(256,)
launch CUDA Kernel('pyk_saxpy_e6a555c8') on Device('NVIDIA GeForce RTX 4070 Laptop GPU')
       with grid_dim=(3907, 1, 1), block_dim=(256, 1, 1)
```

The logging is on by default and is controlled by one flag per backend (`CL.print_info`, `CUDA.print_info`, `NVRTC.print_info`, `HIP.print_info`, `HipCompiler.print_info`). To switch them all at once:

```python
from pygpgpu import set_print_info

set_print_info(False)      # quiet from here on
set_print_info(True)       # verbose again (the default)
```

`set_print_info` only touches backends whose module is already imported, so it never loads a GPU runtime just to silence output. A backend imported *after* the call starts out verbose again — call it once your backends are in use, or set a single flag directly (`pygpgpu.opencl.driver.CL.print_info = False`).

## Known limitations

- `@kernel` does not support: strings/lists/dicts, recursion, dynamic multi-dimensional shape indexing. Closures and module globals *are* supported: an outer variable becomes a kernel parameter, and one declared `nonlocal`/`global` is written back after the call (see "Return values and outer variables" above);
- `print()` inside a kernel prints one `True`/`False` pair, integer, real or vector at a time. `sep` and `end` are translated, but must be string literals (the format string is fixed in the device code); `file` and `flush` are refused; it also cannot print an array (print an element instead). A line containing many floats expands into a `switch` over their digit counts, so an extremely long line can hit the code-size guard and should be split into several calls. The HIP backend rejects `print()` outright: its driver drops the device on such kernels on the tested runtime;
- vector comparison, writing through a swizzle, and nesting beyond three operands are restricted in favour of cross-backend consistency (the error message suggests a workaround);
- `Image` is unsupported on the HIP backend: HIP reuses CUDA code generation, which has no texture object type, and `hipTexObjectCreate` is rejected with "operation not supported" on hardware without texture support (an integrated Radeon under ROCm 7.2 Windows, for instance). The translator raises a `SyntaxError` naming the backend; use cuda, opencl or python for image kernels;
- type annotations on a `@kernel` function are accepted and ignored: the device type comes from the actual argument at call time, since one kernel is specialized per argument-type signature. Annotations are there for editors and type checkers, not for translation;
- a `@kernel` signature may carry default values, and a call may pass arguments by keyword: the call is bound to positional order before launch, with omitted defaults filled in. `*args`, `**kwargs` and keyword-only parameters remain unsupported;
- fp16 (`np.float16`) arguments are unavailable on the CUDA/HIP backends (use float32);
- the HIP backend compiles with the clang shipped by the AMD HIP SDK; hipRTC on this machine (ROCm 7.2 Windows) fails on any input, so it is not used;
- the pure-Python backend (`k["python"]`) does not emulate `local` (group-shared memory): it runs serially and has no parallel work items to share it.

On CUDA, the array APIs (`Array` upload/read-back through `cuMemcpy2D`) work on every supported interpreter. `cuSurfObjectCreate` and `cuTexObjectCreate` occasionally return `CUDA_ERROR_INVALID_VALUE` on a descriptor that is byte-for-byte valid; the failures cluster in the first few calls of a process and then stop. Measured on the machine this was developed on (driver 13.4, RTX 4070): about one in a hundred creations fails, retrying an already-failed descriptor does not recover it, and a fresh descriptor for a newly created array usually does. Because an `Image` binds one surface and one texture per launch, a scenario that launches several image kernels in one process can therefore lose a few launches. The `python` and OpenCL backends are unaffected: OpenCL images are created per launch on a code path that has not shown the failure, and the `python` backend touches no device at all. `examples/11_texture2d.py` reports a driver refusal instead of raising.
## License

MIT
