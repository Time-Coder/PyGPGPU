# PyGPGPU

**English** | [Chinese](README.zh-CN.md)

Decorate a Python function and it runs on the GPU — one codebase, three backends (**OpenCL / CUDA / HIP**).

```python
import numpy as np
from pygpgpu import kernel

@kernel
def saxpy(x, y, out, alpha):
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
- **Plain-Python helper functions** — a helper called from a kernel needs no decorator and no type annotations: it is translated into a device function and monomorphized from the call site, recursively for helpers that call helpers. The same holds for helpers reached through a name: `module.func(...)`, `SomeClass.static_method(...)`, `self.static_method(...)`, and `self.other_method(...)` between methods of one class.
- **All three backends in one process** — the bracket chooses the backend per call, so a single process (and even a single kernel object) can mix OpenCL, CUDA and HIP; each backend/device pair is specialized and cached separately.
- **Pure-Python backend** — `k["python"]` runs the same `@kernel` body in the interpreter, serially on the CPU: no GPU, no compiler, no runtime. Useful for testing, for small inputs and as a reference for what the GPU semantics mean.
- **Handwritten kernel file import** — `.cl` / `.cu` / `.hip` files import like Python modules: kernels and struct types become module attributes, and a `.pyi` type stub is generated automatically.
- **Low-level OOP API** — Device/Context/Buffer/Program/Kernel, for when you need full control.
- **One bracket syntax** for backend/device/work size (identical to handwritten kernels, order does not matter):
  ```python
  k(args)                          # fully automatic
  k["cuda"](args)                  # backend (opencl/cuda/hip/python, aliases ocl/cu/rocm/amd/py)
  k["opencl:gfx1036"](args)        # backend:device
  k[(1024, 256)](args)             # work size: (total, per group), None = automatic
  k["cuda"][(1024, 256)](args)     # combined
  ```
- **Total-work semantics** (OpenCL/SYCL style): in `(global, local)`, `global` is the **total number of work items**; CUDA/HIP grid dimensions are derived as `global / local`, so all three backends behave exactly the same.
- **Static read/write analysis**: the kernel body is parsed to infer whether each pointer argument is read-only, write-only or read-write — write-only arguments skip the upload, read-only arguments skip the read-back, and data stays on the device across calls (zero intermediate copies for gnp arrays).
- **Vector types**: 50 vector types (float4/int3/...) with `.x/.y/.z/.w` members, swizzles, construction, arithmetic (scalar broadcast), comparisons and component-wise math.
- **Dual-sided function library** `pygpgpu.funcs` (aka `pygpgpu.vec_types.funcs`): `dot/cross/normalize/clamp/...` run as-is in host Python and are translated to device builtins inside `@kernel`.
- **Device builtin namespace** `gpu.*`: `get_global_id/barrier/atomic_add/dot/...` covers GPU builtins that have no Python counterpart, mapped per backend.
- **Multi-dimensional iteration**: `np.ndindex(shape)` / `np.ndenumerate(arr)` magic loops (2D/3D), `enumerate(arr)` for a 1D index plus value snapshot.
- **Three cache layers**: parsed metadata / compiled artifacts / generated sources, all invalidated by mtime — unchanged sources mean zero recompilation.

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
from pygpgpu import kernel, gpu, funcs, float4       # @kernel / device builtins / dual-sided funcs / vector types
from pygpgpu import list_devices, get_device
from pygpgpu import climport, cuimport, hipimport    # import hooks for handwritten .cl/.cu/.hip files
```

The submodule paths (`pygpgpu.kernel` / `pygpgpu.vec_types` / `pygpgpu.opencl` / ...) still work. Note that top-level `pygpgpu.kernel` is the `@kernel` decorator; to reach things inside the submodule write `from pygpgpu.kernel import ...`.

The three import hooks load lazily: the corresponding backend is only loaded when you actually use the hook, so `import pygpgpu` itself needs no GPU runtime at all.

### The @kernel decorator

```python
import math
import numpy as np
from pygpgpu import kernel, gpu, funcs, float4

def norm2(v):
    return funcs.dot(v, v)

@kernel
def normalize_all(vecs, out):
    for i, v in enumerate(vecs):          # 1D: index + float4 value snapshot
        out[i] = funcs.normalize(v)       # expanded component-wise on device

@kernel
def matrix_scale(m, out, k):
    for y, x in np.ndindex(out.shape):    # 2D NDRange
        out[y, x] = m[y, x] * k

@kernel(backend="cuda")
def histogram(data, counts):
    gpu.work_size(data)                   # iteration domain for a gid-style kernel
    i = gpu.get_global_id(0)
    gpu.atomic_add(counts, data[i], 1)    # atomics, results read back automatically
```

Supported: scalar/array/vector arguments, if/while/for-range, arithmetic/comparison/bit ops, `math.*` and `np.*` math functions, class methods (`self.xxx` attributes become arguments automatically), static methods, and calling helpers through a name — a module-level function, `module.func(...)`, `SomeClass.static_method(...)`, and `self.method(...)` for another method of the same class. Unsupported syntax raises a `SyntaxError` carrying the line number.

A kernel needs an iteration domain for its work size to be derived. Either use a magic loop (`for i in range(x.shape[0])`, whose bound is the domain) or, when dispatching on `gpu.get_global_id`, declare `gpu.work_size(x)` — that statement is host-side only, erased by the translator, and contributes all of `x`'s dimensions. A `k[(total, local)]` bracket also works. Without any of these the call raises instead of guessing.

### Printing from a kernel

`print()` works inside a kernel on the OpenCL and CUDA backends: it is translated into device `printf` calls that write the same line Python would, so debugging a kernel needs no handwritten C.

```python
@kernel
def inspect(a, out):
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
def saxpy(x, y, out, alpha):
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
def saxpy(x, y, out, alpha):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

saxpy["python"](x, y, out, 2.0)      # ordinary Python, one call, serial
saxpy["py"].submit(x, y, out, 2.0)   # same launch on a thread pool
```

It is a correctness and convenience backend, not a speed one: the whole kernel runs on the CPU, serially. A launch can also be started with `submit` (a `concurrent.futures.Future`) or `async_call` (an `asyncio.Future`), both backed by a `ThreadPoolExecutor`; one launch is still serial, so concurrent launches must write to different arrays.

The two kernel styles keep their meanings, which is where the interpreter differs from a plain call:

| Kernel style | What `k["python"](...)` does |
|---|---|
| magic loop (`for i in range(x.shape[0])`, `enumerate(x)`, `np.ndindex`/`np.ndenumerate`) | runs the function once: the loop already covers every element. A `(total, local)` bracket changes nothing here, because the loop, not the launch, decides the extent |
| gid-style (`gpu.get_global_id`), domain from `gpu.work_size(x)` | runs the body once per work item of the declared domain (`x`'s full shape), with `gpu.get_global_id` reporting the current index — one serial step per element, in `np.ndindex` order |
| gid-style, domain from a `k[(total, local)]` bracket | same, one step per work item of the bracket, so the body must clamp itself against the real element count exactly as it does on a GPU |

Inside such a body `gpu.*` resolves to host implementations: the indexing builtins, the atomics (plain read-modify-writes — serial execution makes them exact), `gpu.barrier()` (a no-op) and the vector math family, which uses `pygpgpu.funcs`' host implementation. Two things cannot be emulated and raise instead: `gpu.local` (group-local memory is shared by parallel work items) and a `gpu.get_global_id` call inside a magic-loop body, where there is no current work item — use the loop variable there.

Vector-typed arrays work unchanged: reading `v = points[i]` yields a vector instance, and `out[i] = v.xyzy * 2.0` writes its components.

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
from pygpgpu import climport             # cuimport / hipimport work the same way
import numpy as np
import vecadd                          # just like importing a Python module

a = np.random.rand(1_000_000).astype(np.float32)
out = np.zeros_like(a)
vecadd.vecadd(a, a, out, np.int32(len(a)))    # work size inferred
vecadd.vecadd[(1_000_192, 256)](a, a, out, np.int32(len(a)))   # explicit (total, per group)
```

Note the `n` argument: a handwritten kernel must clamp against the real element count, because the
launch size is rounded up to a multiple of the local size. `get_global_size(0)` reports that rounded
size, not the buffer length, so it is not a usable bound.

Struct definitions map to ctypes types; a `.pyi` stub is generated next to the kernel so IDE completion works.

### Class methods

```python
class Model:
    def __init__(self):
        self.weights = np.random.rand(1024).astype(np.float32)
        self.lr = 0.01

    @kernel
    def apply(self, x, out):
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

def soften(v):                      # module-level helper
    return v * 0.5

class Model:
    def __init__(self):
        self.level = 1.0            # read by gain(), so it becomes a kernel argument

    @staticmethod
    def scale(v):                   # static method: no receiver
        return v * 3.0

    def gain(self):                 # instance method: reads self.level
        return self.level + 1.0

    @kernel
    def apply(self, x, out):
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
def k(a, b, out):
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

def square(v):                          # plain Python: no decorator, no annotations
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
def apply_all(x, out):
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
| Helper functions | **plain Python — no decorator, no annotations** | `@ti.func` + annotated args | `@cuda.jit(device=True)` | `@wp.func` + annotated args |
| Backends in one process | **all three, mixed freely** | one, fixed by `ti.init(arch=...)` | CUDA only | CUDA only |
| Launch size | inferred from the loop bound (or `gpu.work_size(x)` / a bracket) | inferred from `range` | blocks × threads by hand, plus a guard | pass `dim=` |
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
def apply_all(x, out):
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

- `@kernel` does not support: strings/lists/dicts, closures, recursion, kernel return values, dynamic multi-dimensional shape indexing;
- `print()` inside a kernel prints one `True`/`False` pair, integer, real or vector at a time and takes no `sep`/`end`/`file` keywords; it also cannot print an array (print an element instead). A line containing many floats expands into a `switch` over their digit counts, so an extremely long line can hit the code-size guard and should be split into several calls. The HIP backend rejects `print()` outright: its driver drops the device on such kernels on the tested runtime;
- vector comparison, writing through a swizzle, and nesting beyond three operands are restricted in favour of cross-backend consistency (the error message suggests a workaround);
- `Image` is unsupported on the HIP backend: HIP reuses CUDA code generation, which has no texture object type, and `hipTexObjectCreate` is rejected with "operation not supported" on hardware without texture support (an integrated Radeon under ROCm 7.2 Windows, for instance). The translator raises a `PyKernelSyntaxError` naming the backend; use cuda, opencl or python for image kernels;
- type annotations on a `@kernel` function are accepted and ignored: the device type comes from the actual argument at call time, since one kernel is specialized per argument-type signature. Annotations are there for editors and type checkers, not for translation;
- fp16 (`np.float16`) arguments are unavailable on the CUDA/HIP backends (use float32);
- the HIP backend compiles with the clang shipped by the AMD HIP SDK; hipRTC on this machine (ROCm 7.2 Windows) fails on any input, so it is not used;
- the pure-Python backend (`k["python"]`) does not emulate `gpu.local` (group-shared memory): it runs serially and has no parallel work items to share it.

On CUDA, the array APIs (`Array` upload/read-back through `cuMemcpy2D`) work on every supported interpreter. `cuSurfObjectCreate` and `cuTexObjectCreate` occasionally return `CUDA_ERROR_INVALID_VALUE` on a descriptor that is byte-for-byte valid; the failures cluster in the first few calls of a process and then stop. Measured on the machine this was developed on (driver 13.4, RTX 4070): about one in a hundred creations fails, retrying an already-failed descriptor does not recover it, and a fresh descriptor for a newly created array usually does. Because an `Image` binds one surface and one texture per launch, a scenario that launches several image kernels in one process can therefore lose a few launches. The `python` and OpenCL backends are unaffected: OpenCL images are created per launch on a code path that has not shown the failure, and the `python` backend touches no device at all. `examples/11_texture2d.py` reports a driver refusal instead of raising.
## License

MIT
