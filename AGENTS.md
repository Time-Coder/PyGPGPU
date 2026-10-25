# AGENTS.md

PyGPGPU runs Python functions as GPU kernels on OpenCL (default), CUDA, and HIP. Pure Python + ctypes, Python >= 3.12, packaged with setuptools. `README.md` is the English version (shown first on GitHub); `README.zh-CN.md` holds the Chinese version and the two link to each other at the top — when editing user-facing docs, keep both in sync.

## Language policy: English only (hard rule)

- **All comments, docstrings, error messages, log/UI text, and every other string in code must be written in English. Never generate Chinese (CJK) characters** in `.py` / `.cl` / `.cu` / `.hip` sources, generated stubs and templates (`*.pyi`, `*.pyi.in`), `examples/`, `pyproject.toml`, or any documentation file.
- The **only** permitted Chinese files are `README.zh-CN.md` mirrors of an English `README.md`, in any directory (currently two: repo root and `examples/`). When user-facing docs change, keep the matching mirror in sync. Nothing else may be added in Chinese, and no code, script, or tool may emit Chinese text.
- If you find Chinese in a file you are touching, translate it to English as part of that edit rather than copying the style.
- Spot check before finishing a task: search the tree for CJK codepoints (`rg -n '[\p{Han}]' --glob '!README.zh-CN.md'`) — it must come back empty. Language bars link with the word `[Chinese]`, never CJK, so no file outside a mirror needs an exemption.
- This rule is self-referential: AGENTS.md itself contains no CJK, and neither should any file you add.

## Setup and verification

- `pip install -e ".[dev]"` (dev dependencies: pytest, ruff, ty). No CI, pre-commit or committed test suite exists. The whole-repo check is `python check.py` (runs `ruff check --fix` then `ty check` against the config in `pyproject.toml`). Do not imply `pytest` has tests to run.
- Fast checks without a GPU: `python -m compileall -q pygpgpu` and `python -c "from pygpgpu.kernel import kernel"`.
- The end-to-end smoke tests are the numbered scripts in `examples/` (each is standalone: `python examples/01_basic_kernel.py`; no backend specified means probe `cuda -> hip -> opencl`). `examples/06_handwritten_kernels.py` covers the `.cl` import hook, struct parameters and static read/write analysis; `examples/09_reduction.py` covers atomics, local memory and tree reductions.
- Backend packages initialize their runtime eagerly: `import pygpgpu.cuda` needs `nvcuda.dll`; `import pygpgpu.hip` needs `amdhip64.dll` and raises when it is missing. Keep smoke scripts backend-specific instead of importing all three.
- HIP compiles by spawning ROCm's `clang.exe` (hipRTC is unusable on the tested ROCm 7.2 Windows). Set `PYGPGPU_HIP_INCLUDES` (`os.pathsep`-separated) when MSVC/Windows SDK headers are not auto-discovered.

## Layout and ownership

- `pygpgpu/kernel/`: `@kernel` decorator (`decorator.py`), Python-to-device AST translator (`translator.py`), and `gpu.*` builtin mapping (`builtins.py`). Backend-neutral except that HIP reuses CUDA code generation.
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
- All three backends can be used in one process, and the same `PyKernel` object caches one specialization per backend + device + signature. Keep that working: do not add process-wide backend state (Taichi-style `init(arch=...)` global config), and keep backend package imports lazy so importing `pygpgpu` still needs no GPU runtime.
- Static read/write analysis (`KernelFlags` in `constants.py`, implemented in `ts_parser.py`) decides which arguments are uploaded/downloaded. Do not defeat it when adding argument handling.
- Importing `pygpgpu` monkey-patches `numpy.array` process-wide to support nested `ctypes.Structure` values. Keep that behavior unless the change is deliberate.
- Caches live beside the source (`__clcache__`, `__cucache__`, `__hipcache__`; generated `@kernel` sources under `.../pykernels`) and invalidate by mtime/source hash. Delete the relevant cache when debugging stale generated code.
- Handwritten-kernel `.pyi` stubs are regenerated next to the source on import; `examples/vecadd.pyi` comes from `examples/vecadd.cl`. Edit the kernel/parser, not the generated stub.
- Regenerate the vector-type stubs with `python -m pygpgpu.vec_types.helper`; templates are `genVecN.pyi.in`.

## Known scope limits

- `@kernel` does not support strings/lists/dicts, closures, recursion, return values, or dynamic multi-dimensional indexing; unsupported syntax raises `PyKernelSyntaxError` with a line number.
- fp16 arguments are unsupported on CUDA/HIP; use float32.
