# Comparing PyGPGPU with Taichi, NVIDIA Warp and Numba

One script per framework, each a standalone process running the same two tasks. They are
deliberately separate: importing a framework costs a few hundred milliseconds, and in one
shared process every import would be billed to whichever framework happened to run last.

The matrix is per **framework and backend**, because the backend is most of what a GPU
framework's number means: PyGPGPU's CUDA and OpenCL paths are the same source through two
different compilers, and Taichi's CPU backend is a different program from its CUDA one.

| entry | script | backend |
| --- | --- | --- |
| `pygpgpu-cuda` / `-opencl` / `-hip` | `bench_pygpgpu.py --arch ...` | CUDA, OpenCL, HIP |
| `taichi-cuda` / `-cpu` | `bench_taichi.py --arch ...` | CUDA, x64 |
| `quadrants-cuda` / `-vulkan` / `-cpu` | `bench_quadrants.py --arch ...` | CUDA, Vulkan, x64 |
| `warp-cuda` | `bench_warp.py` | CUDA |
| `numba-cuda` / `numba-cpu` | `bench_numba_cuda.py`, `bench_numba_cpu.py` | CUDA, CPU |

**Quadrants is a Taichi derivative, and its script mirrors `bench_taichi.py` call for call** on
purpose: a difference between the two numbers is then a difference in the compiler and
runtime, not in how the task was written. Its Vulkan backend *does* work here, where Taichi's
aborts -- so the two are not interchangeable even though the API is.

Taichi's Vulkan backend is **not in the list**: `ti.init(arch=ti.vulkan)` aborts on this
machine with a VulkanMemoryAllocator assertion (`vk_mem_alloc.h`, line 16039), so there is no
number to report rather than a slow one.

```bash
python run_compare.py                 # every entry, both tasks
python run_compare.py --only pygpgpu-cuda warp-cuda
python bench_taichi.py --task laplacian --arch cpu    # one script by hand
```

## What the four numbers mean

| column | how it is measured | what it tells you |
| --- | --- | --- |
| **start-up** | a run with `--startup-only`: import the framework, start its runtime, stop | what a process pays before it can run anything |
| **cold (full)** | wall time of the whole process, caches deleted first | a first ever run, compilation and JIT included |
| **warm (full)** | the next runs, caches in place, best of several | starting an already-built program |
| **call 2 / best call** | reported by the script itself | one call once the program is up |

`start-up` is the honest process-level number; `cold (full)` also contains the timed calls
and the reference computation every script does to check its answer, so the difference
between the two columns is not purely "compile time". What a *compilation* costs is
`CALL1_MS` inside a script, which each one prints.

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

`taichi`, `numba`, `warp-lang` (`import warp`) and `numpy` are needed for the corresponding
scripts; PyGPGPU's OpenCL backend also needs `tree-sitter-opencl`. On a machine whose driver
is newer than the ABI Numba's own ctypes bindings assume, Numba's CUDA target needs the
official bindings:

```bash
pip install "cuda-python<13"
```

`bench_numba_cuda.py` sets `NUMBA_CUDA_USE_NVIDIA_BINDING=1` for exactly that reason; without
it the first context query faults inside `cuCtxGetDevice`.

PyGPGPU needs its own dependency (`pygf`) importable, and the repository on `sys.path` -- the
script inserts `../..` itself, so running it from this folder is enough.

## Where a process's time actually goes

Measured one step at a time, each in a fresh interpreter (best of three):

| step | cumulative | what it added |
| --- | ---: | ---: |
| interpreter only | 135 ms | 135 ms |
| `+ numpy` | 279 ms | 144 ms |
| `+ pygf` | 312 ms | 33 ms |
| `+ pygpgpu` | 493 ms | 181 ms |
| `+ pygpgpu.cuda` | 602 ms | 109 ms |
| `import taichi` | 565 ms | 435 ms over the interpreter |
| `taichi.init(arch=cuda)` | 895 ms | **330 ms for the runtime alone** |
| `import warp; warp.init()` | 671 ms | 541 ms over the interpreter |
| `import numba, numba.cuda` | 715 ms | 585 ms over the interpreter |

Two things follow. **PyGPGPU starts cheapest** and Taichi's CUDA entry is the most expensive
purely because `ti.init` costs a third of a second by itself. And numpy's 144 ms is a floor
every framework pays, which is why the start-up column is compared as a whole rather than
taken apart.

The remaining room in PyGPGPU's own 290 ms of import (the package plus its CUDA backend) is
in *what it imports up front*: the backends it is not going to use and the compiler plumbing
are loaded whether or not the call needs them. Making that lazy is a real, measurable target
of a few hundred milliseconds -- and it would disturb the import graph of every backend, so
it is a change of its own rather than a tweak.

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

Caching those three and short-circuiting the fourth brought the call from 16.7 ms to about
1 ms. A second pass at the profile then found two more:

* the gate that decides whether a body constructs a class was walking the AST **from a fresh
  `deepcopy` on every launch**, which was the largest single python cost once compiling was
  out of the way -- it only reads the tree, so it takes the cached one now (`_readonly_parse`);
* that gate and the "does this body build a value from a class" answer are facts about the
  source, so they are remembered per function rather than re-derived per call.

Together the two rounds took the steady-state call to **0.29 ms (mandelbrot) and 0.31 ms
(stencil)** against Numba CUDA's 0.12/0.19 and Taichi's 0.17/0.13 -- a factor of 2.5 and 1.6,
where it started at ~100. What is left is dominated by the six driver calls a launch makes,
not by python.

None of the fixes are in the benchmark: the numbers this folder reports are what a user gets.
