# AGENTS.md

PyGPGPU runs Python functions as GPU kernels on OpenCL (default), CUDA, and HIP, and can also run them serially in the interpreter through the `python` backend. Pure Python + ctypes, Python >= 3.7, packaged with setuptools. `README.md` is the English version (shown first on GitHub); `README.zh-CN.md` holds the Chinese version and the two link to each other at the top — when editing user-facing docs, keep both in sync.

## Language policy: English only (hard rule)

- **All comments, docstrings, error messages, log/UI text, and every other string in code must be written in English. Never generate Chinese (CJK) characters** in `.py` / `.cl` / `.cu` / `.hip` sources, generated stubs and templates (`*.pyi`, `*.pyi.in`), `examples/`, or any documentation file.
- The **only** permitted Chinese files are `README.zh-CN.md` mirrors of an English `README.md`, in any directory (currently two: repo root and `examples/`). When user-facing docs change, keep the matching mirror in sync. Nothing else may be added in Chinese, and no code, script, or tool may emit Chinese text.
- One narrow exception, and do not extend it: the native-form author name in `pyproject.toml`'s `authors` table, kept alongside the ASCII `Binghui Wang` entry used everywhere else. Machine-readable packaging metadata is the only place CJK is allowed; the ASCII form is canonical for all prose, code, and display.
- If you find Chinese in a file you are touching, translate it to English as part of that edit rather than copying the style.
- Spot check before finishing a task: search the tree for CJK codepoints (`rg -n '[\p{Han}]' --glob '!README.zh-CN.md'`) — the only acceptable hits are the two `README.zh-CN.md` mirrors and the native-form author name in `pyproject.toml`. Language bars link with the word `[Chinese]`, never CJK, so no other file needs an exemption.
- This rule is self-referential: AGENTS.md itself contains no CJK, and neither should any file you add.

## Setup and verification

- `pip install -e ".[dev]"` (dev dependencies: pytest, ruff, ty). No CI, pre-commit or committed test suite exists. The whole-repo check is `python check.py` (runs `ruff check --fix` then `ty check` against the config in `pyproject.toml`). Do not imply `pytest` has tests to run.
- Fast checks without a GPU: `python -m compileall -q pygpgpu` and `python -c "from pygpgpu.kernel import kernel"`.
- The end-to-end smoke tests are the numbered scripts in `examples/` (each is standalone: `python examples/01_basic_kernel.py`; no backend specified means probe `cuda -> hip -> opencl`). `examples/06_handwritten_kernels.py` covers the `.cl` import hook, struct parameters and static read/write analysis; `examples/09_reduction.py` covers atomics, local memory and tree reductions.
- Backend packages initialize their runtime eagerly: `import pygpgpu.cuda` needs `nvcuda.dll`; `import pygpgpu.hip` needs `amdhip64.dll` and raises when it is missing. Keep smoke scripts backend-specific instead of importing all three.
- HIP compiles by spawning ROCm's `clang.exe` (hipRTC is unusable on the tested ROCm 7.2 Windows). Set `PYGPGPU_HIP_INCLUDES` (`os.pathsep`-separated) when MSVC/Windows SDK headers are not auto-discovered.

## Layout and ownership

- `pygpgpu/kernel/`: `@kernel` decorator (`decorator.py`), Python-to-device AST translator (`translator.py`), and `gpu.*` builtin mapping (`builtins.py`). Backend-neutral except that HIP reuses CUDA generation. `python_backend.py` is the interpreter backend: it translates nothing and executes the body serially, so it is not a fourth code generator — keep it that way (no AST translation there, only the source scan that tells a magic loop from a gid-style body).
- `pygpgpu/{opencl,cuda,hip}/driver/`: ctypes bindings. Vendor headers are in `driver/include/`; function/enum tables are in `*info.py` / `*types.py`.
- `pygpgpu/{opencl,cuda,hip}/oop/`: parallel Device/Context/Buffer/Program/Kernel APIs. The three stacks are intentionally similar but not shared; a semantic change usually needs mirrored edits in all three.
- `pygpgpu/cparser/`: the active parse path is `cpreprocessor.py` + `ts_parser.py` (tree-sitter mixin used by all backends). `cparser.py`, `csyntaxtree.py`, `ctokens.py`, and `ShaderBuiltins.py` are legacy and not imported at runtime. `pcpp/` is vendored; avoid patching it.
- `climport` / `cuimport` / `hipimport`: meta-path finders that let `.cl` / `.cu` / `.hip` files be imported like modules. Importing a backend package installs its hook; resolution walks `sys.path`.
- `pygpgpu/numpy/` is `gnp`: a device-resident `np.ndarray` subclass shared by all backends through the context/stream interfaces.

## Repo-specific behavior to preserve

- `(global, local)` work sizes use total-work-item semantics on every backend; CUDA/HIP grid dimensions are derived as `global // local`. Do not reinterpret `global` as a block count.
- Automatic work sizes are chosen in `utils.py` (`detect_work_size` / `local_size_for_global`): the local size comes from the extents (or their divisors) and is clamped to the device's real limits, preferring warp-aligned groups. The derived `global` is intentionally **not** required to equal the array length: when a bounds guard is guaranteed (`@kernel` always emits one) the global may be rounded up to a multiple of the local, which keeps work groups efficient for awkward extents such as prime lengths. Do not "fix" this rounding back to exact equality.
- Handwritten kernels must clamp themselves against the real element count passed as an argument. `get_global_size(dim)` reports the (rounded) launch size, not the buffer length, so it is not a valid bound; see `examples/vecadd.cl`.
- A gid-style kernel (`gpu.get_global_id`) has no loop for the automatic work size to read a domain from, so it must declare one: `gpu.work_size(x)` at the top of the body (host-side only, erased by the translator, contributes all of `x`'s dimensions) or a `k[(total, local)]` bracket. Without either, the launch raises instead of falling back to a buffer-shape guess: that guess used to silently pick a 1-work-item grid for a 100000-element reduction and produce a wrong answer.
- `@kernel` specialization is per backend + device + argument type signature, compiled at first call rather than decoration. Kernels need retrievable Python source: define them in real `.py` files, not stdin/`exec` strings (`inspect.getsource`).
- Helper functions called from a kernel need no decorator and no annotations: `_emit_device_call` translates them into device functions and monomorphizes per call-site argument types, recursively. Emission order must be dependency order (`_dependency_order` in `decorator.py`) because C requires a callee before its caller and the cache fills depth-first. A helper defined in `python -c`/stdin has a `<string>` filename and yields an invalid C identifier, so keep helpers in real `.py` files.
- All three backends can be used in one process, and the same `PyKernel` object caches one specialization per backend + device + signature. Keep that working: do not add process-wide backend state (Taichi-style `init(arch=...)` global config), and keep backend package imports lazy so importing `pygpgpu` still needs no GPU runtime. The `python` backend takes part in the same bracket dispatch but caches nothing (there is nothing to specialize) and is never probed as a default, so a machine without a GPU keeps raising instead of silently falling back to the interpreter.
- The `python` backend's launch semantics must stay in step with the GPU ones: a magic-loop body runs once as written, a gid-style body runs once per work item of its `gpu.work_size(...)` domain or `(total, local)` bracket, and `gpu.*` resolves through the thread-local frame in `builtins.py` so concurrent launches never share a work item.
- Static read/write analysis (`KernelFlags` in `constants.py`, implemented in `ts_parser.py`) decides which arguments are uploaded/downloaded. Do not defeat it when adding argument handling.
- Importing `pygpgpu` monkey-patches `numpy.array` process-wide to support nested `ctypes.Structure` values. Keep that behavior unless the change is deliberate.
- Caches live beside the source (`__clcache__`, `__cucache__`, `__hipcache__`; generated `@kernel` sources under `.../pykernels`) and invalidate by mtime/source hash. Delete the relevant cache when debugging stale generated code.
- Handwritten-kernel `.pyi` stubs are regenerated next to the source on import; `examples/vecadd.pyi` comes from `examples/vecadd.cl`. Edit the kernel/parser, not the generated stub.
- Regenerate the vector-type stubs with `python -m pygpgpu.vec_types.helper`; templates are `genVecN.pyi.in`.

## Known scope limits

- `@kernel` does not support strings/lists/dicts, closures, recursion, return values, or dynamic multi-dimensional indexing; unsupported syntax raises `PyKernelSyntaxError` with a line number. These limits belong to the GPU translator: the `python` backend runs the body as-is and accepts such code, except `gpu.local` (no serial equivalent) and `gpu.get_global_id` inside a magic-loop body (no current work item).
- `print()` inside a kernel is translated into device `printf` (`FunctionTranslator._emit_print`) on OpenCL and CUDA. A whole line must be **one** printf call site: every device call site owns an output buffer and the buffers are drained per call site, so a line split over several printf calls comes back interleaved with other work items. That is why a bool argument branches the line (one printf per `True`/`False`) and a real argument expands it per digit budget into a `switch` of literal formats — OpenCL rejects a computed format string, so `%.*f`/`%.*e` cannot be emitted. Keep the leaf count bounded by `PRINT_MAX_LINES`.
- `print()` is **deliberately rejected on HIP**. Do not try to "make it work" by experimenting with HIP printf kernels: on the tested ROCm 7.2 Windows machine, launching kernels that carry those printf call sites makes the AMD driver drop the device and Windows performs a display-driver reset (the machine has to be recovered). The OpenCL and CUDA paths are verified; treat the HIP rejection as the final behaviour.
- Floats print in python's shortest round-trip form, and the digit count is chosen on the device (`_print_real_selector`) because it is a per-value property. `PRINT_REAL_DIGITS`, `PRINT_REAL_SCI_HI`/`_LO` and `PRINT_REAL_PLAIN_MAX` encode numpy `str()` (float32) and python `repr()` (double) notation; do not collapse this back to one fixed `%g` precision.
- fp16 arguments are unsupported on CUDA/HIP; use float32.
