# Comparing PyGPGPU with Taichi, Numba CUDA and Numba CPU

Four scripts, one per framework, each a standalone process running the same two tasks. They
are deliberately separate: importing a framework costs a few hundred milliseconds, and in one
shared process every import would be billed to whichever framework happened to run last.

```bash
python run_compare.py                 # all four, both tasks
python run_compare.py --only pygpgpu  # one of them
python bench_taichi.py --task laplacian --arch cuda   # one script by hand
```

## What the three numbers mean

| column | how it is measured | what it tells you |
| --- | --- | --- |
| **cold script** | wall time of the whole process, with the framework's caches deleted first | what a first ever run costs, compilation and JIT included |
| **warm script** | wall time of the next runs, cache in place, best of three | what starting an already-built program costs |
| **call 2** | reported by the script itself, the second invocation of the same function | what one call costs once the program is up |

The last one is the number to compare for kernel speed; the first two are dominated by
interpreter start-up and by how each framework stores its compiled output.

Every script prints a checksum and a verdict, so a fast wrong answer cannot pass.

## The tasks

* **mandelbrot** -- 1024x512 pixels, up to 200 iterations of `z = z*z + c` per pixel. Compiled
  loops and data-dependent branches; compute-bound.
* **laplacian** -- 2048x2048, a 5-point stencil. Five reads and one write per element, no
  branching; memory-bound.

Both compute in **float32**, which is what a GPU kernel normally uses. Numba would otherwise
infer float64 from the literals, and a float64 build of the same loop is a different and much
slower program.

The arrays live on the device for all four (`pygpgpu.numpy`, `cuda.to_device`, `ti.field`), and
the transfer back to the host for the check happens outside the timed region. A framework that
uploaded and downloaded on every call would be measured against the PCIe bus rather than
against its compiler.

## Reproducing the numbers

`run_compare.py` clears, before each cold run:

* `__cucache__` / `__clcache__` / `__hipcache__` / `__metalcache__` (PyGPGPU's generated
  sources and compiled modules, written beside the script);
* `__pycache__` and `.numba-cache` (Numba's `.nbi`/`.nbc` files, redirected here through
  `NUMBA_CACHE_DIR` so they can be removed by name);
* `~/.cache/taichi`, if it exists.

Taichi 1.7 keeps its offline cache in the frontend rather than on disk, so its cold and warm
columns come out equal. That is the measurement, not a missing step.

## Requirements

`taichi`, `numba` and `numpy` are needed for the corresponding scripts. On a machine whose
driver is newer than the ABI Numba's own ctypes bindings assume, Numba's CUDA target needs the
official bindings:

```bash
pip install "cuda-python<13"
```

`bench_numba_cuda.py` sets `NUMBA_CUDA_USE_NVIDIA_BINDING=1` for exactly that reason; without
it the first context query faults inside `cuCtxGetDevice`.

PyGPGPU needs its own dependency (`pygf`) importable, and the repository on `sys.path` -- the
script inserts `../..` itself, so running it from this folder is enough.

## A note on the PyGPGPU result

The first version of this comparison found PyGPGPU roughly **100x** behind Numba's CUDA
target on both tasks. Profiling one launch showed almost none of that time was on the device:

* the kernel source was re-parsed several times per call, and the compatibility shim that
  rewrites pre-3.9 AST shapes ran on every node of every parse, where `isinstance` against
  `ast.Num`/`ast.Str` raises a `DeprecationWarning` on Python 3.12 -- thousands of
  `warnings.warn` calls per launch;
* `inspect.getsource` was called five times per launch, re-reading and re-tokenizing the file
  each time;
* every array argument went through a vector-type lookup that asked all forty-odd vector
  classes for their numpy dtype, a ctypes round trip each, for an array that was plain float32.

Three caches and one short-circuit later the per-call time is within a small factor of the
other two GPU frameworks. The fixes are in the library, not in the benchmark: the numbers this
folder reports today are what a user gets.
