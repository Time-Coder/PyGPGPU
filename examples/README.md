# Examples

**English** | [Chinese](README.zh-CN.md)

Runnable scripts covering the main PyGPGPU usage patterns. Each file is a standalone script,
run from the repository root with `python examples/xxx.py` (do not run them through
`exec`/stdin in an environment that breaks `inspect.getsource`). All comments are in English,
matching the rest of the repository.

## Dependencies

- Any single GPU backend is enough to run `01`–`05` (with no backend specified, they probe `cuda -> hip -> opencl`);
- `06` needs the runtime for the handwritten-kernel import hook (OpenCL by default, available on nearly every platform);
- `07` uses the low-level OOP API, on the default OpenCL device;
- `08` compares backends only across the backends actually present on your machine — a missing one is skipped, so it always produces results;
- `09` uses only `@kernel` plus atomics, so any single backend works;
- `10` uses `submit`/`async_call` on a decorated kernel (any backend);
- `11` needs no GPU at all: it runs entirely on the pure-Python backend.

## Contents

| File | What it covers |
|---|---|
| [01_basic_kernel.py](01_basic_kernel.py) | The basics of `@kernel`: SAXPY, automatic work size, switching backend/device with brackets |
| [02_backend_device_selection.py](02_backend_device_selection.py) | `list_devices()` / `get_device()` and the device-selection syntax |
| [03_nd_loop_and_class_method.py](03_nd_loop_and_class_method.py) | Multi-dimensional `np.ndindex`/`np.ndenumerate`, plus class methods and static methods; helpers reached through a name: another method of the same class, a static method of any class, a module function |
| [04_vector_types.py](04_vector_types.py) | Vector types (float4/...), swizzles, and the dual-sided function library `pygpgpu.funcs` |
| [05_gnp_zero_copy_and_atomics.py](05_gnp_zero_copy_and_atomics.py) | Zero-copy with device-resident `gnp` arrays, atomic histograms, and local-memory reduction |
| [06_handwritten_kernels.py](06_handwritten_kernels.py) | Importing handwritten `.cl`/`.cu`/`.hip` files like Python modules |
| [07_low_level_oop.py](07_low_level_oop.py) | Low-level OOP API: Platforms/Device/Context/Buffer/Program/Kernel |
| [08_compare_backends.py](08_compare_backends.py) | The same kernel benchmarked on OpenCL / CUDA / HIP |
| [09_reduction.py](09_reduction.py) | Array-sum reduction: per-element atomics, strided private sums, and a local-memory tree reduction |
| [10_async_launch.py](10_async_launch.py) | Non-blocking launches of a `@kernel` function: `submit` / `async_call` |
| [11_python_backend.py](11_python_backend.py) | The pure-Python backend: `k["python"]`, magic loop vs gid-style, plus `submit`/`async_call` |
| [11_returns_and_scope.py](11_returns_and_scope.py) | Return values (as parameters, first return wins) and outer-scope variables (parameterized, `nonlocal` written back) |
| [12_print_in_kernel.py](12_print_in_kernel.py) | `print()` inside a kernel: translated to device `printf` with python's exact rendering |
| [13_image.py](13_image.py) | One backend-neutral `Image` type: a kernel indexes it like an array (`src[y, x]`), translated into `tex2D`/`surf2Dwrite` on CUDA and `read_imagef`/`write_imagef` on OpenCL, and executed by a host implementation on the `python` backend so all three agree element for element |
| [14_heterogeneous_data.py](14_heterogeneous_data.py) | Every kind of value a kernel carries, in all three roles (argument, object built in the body, return value): scalars, 1/2/3-D scalar arrays, vectors by value, 1/2/3-D vector arrays, structures with every kind of member, and 1/2/3-D structure arrays |

`@kernel` function bodies accept a docstring (the leading string literal is ignored by the
translator and never reaches the generated GPU source), so writing explanations inside a
decorated function is safe.

## About automatic work sizes

There are two ways for a kernel to state its iteration domain, and the automatic work size needs
one of them:

- a **magic loop**, `for i in range(x.shape[0]): ...` (or `np.ndindex`/`np.ndenumerate` for 2-D/3-D),
  whose loop bound is the domain;
- for a **gid-style** kernel that dispatches on `get_global_id`, a declaration at the top of the
  body: `work_size(x)`. That statement is host-side only — the translator erases it and emits no
  device code — and it contributes all of `x`'s dimensions, matching what `np.ndindex(x.shape)` would
  give. Without either, the call raises instead of silently guessing.

With a domain, `@kernel` derives the local size from those extents and may round the global size up
to a multiple of it (this keeps work groups efficient for awkward extents, e.g. a prime length,
instead of collapsing them to a single work item). The generated kernel guards every work item
against the real bound, so this is safe for `@kernel`.

Handwritten kernels must clamp themselves, and they cannot use `get_global_size(0)` for that: it
reports the rounded launch size, not the element count. Pass the real count as a kernel argument
and compare against it, as `vecadd.cl` does.

The pure-Python backend (`k["python"]`) keeps the same meaning for both styles: a magic-loop kernel
runs once as written (the loop already covers the elements), while a gid-style kernel runs once per
work item of its iteration domain. It cannot emulate `local`, since group-shared memory needs
parallel work items.

## About backend initialization

Importing the `pygpgpu` top level needs no GPU runtime at all (`climport`/`cuimport`/`hipimport`
are lazy). But **importing a backend subpackage** (`pygpgpu.cuda` / `pygpgpu.hip`) initializes
that runtime immediately: a missing `nvcuda.dll` / `amdhip64.dll` raises right away. So on a
machine without a given runtime, only run the examples that use the default backend or OpenCL;
`02` and `08` are the exceptions — they probe through `list_devices()` and adapt to
whatever runtimes exist, so they never fail because a backend is missing.
