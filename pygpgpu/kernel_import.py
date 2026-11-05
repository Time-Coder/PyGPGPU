"""kernel_import -- import one module name from same-named `.cl`/`.cu`/`.hip` sources.

A hand-written kernel is written once per backend, so three backends of one kernel would
otherwise need three file names and three imports. This hook instead looks for ALL of
`<stem>.cl`, `<stem>.cu` and `<stem>.hip` in the same place and publishes ONE module named
`<stem>`:

    vecops.cl    -> opencl
    vecops.cu    -> cuda
    vecops.hip   -> hip
    from kernel_import import vecops

Every kernel in the sources becomes a module attribute used exactly like a `@kernel`
function: no bracket runs on the default device, `["cuda"]` / `["opencl:1"]` names a
backend or device, `[(blocks, threads)]` states the work size -- in either order. The same
kernel name declared with the same parameter list in several sources is ONE function whose
implementations differ per backend; declared with different parameter lists it is a set of
overloads. A bracket naming a backend that has no implementation for the name raises
`KeyError` at the bracket, before any argument is looked at.

Nothing here goes through a backend's import hook or its bracket-style wrapper: each
backend is reached through its own `ProgramParser` (metadata) and `Program`/`Kernel`
objects, which is the layer a launch is built on anyway.

A machine that lacks one of the runtimes is not a problem for the import: the driver layer
records a failed runtime load instead of failing its own import, so every source is parsed
and the failure surfaces -- as the driver's own error -- only when a kernel on that backend
is actually selected.
"""
from __future__ import annotations

import importlib
import sys
import os
import types
import threading
from typing import Dict, List, Optional, Tuple, Union


# The extension each backend's sources carry, in the probe order `@kernel` uses.
_BACKEND_EXTENSIONS:Tuple[Tuple[str, str], ...] = (
    ("cuda", ".cu"),
    ("hip", ".hip"),
    ("metal", ".metal"),
    ("opencl", ".cl"),
)

# Where each backend keeps the parser that reads its sources. Both are stated once, so the
# four backends differ only by name.
_PARSER_MODULES:Dict[str, str] = {
    "cuda": ".cuda.oop.program_parser",
    "hip": ".hip.oop.program_parser",
    "metal": ".metal.oop.program_parser",
    "opencl": ".opencl.oop.program_parser",
}

# The qualifier words a declaration may carry; dropped before two backends' parameter lists
# are compared, so OpenCL's `__global const float*` and CUDA's `const float*` agree.
_QUALIFIER_WORDS:Tuple[str, ...] = ("const", "volatile", "restrict", "__global",
                                    "__constant", "__local", "__private", "__restrict")


def _source_paths_for(fullname:str, path:Optional[List[str]])->Dict[str, str]:
    """The backend sources one module name resolves to: {"cuda": path, ...}.

    `fullname` may be dotted (`kernel_import.vecops`); the file path is the module path
    joined with each backend's extension. When `path` is given (a parent package's
    `__path__`, which is how `from kernel_import import vecops` arrives) only the module's
    own name is appended to each of those directories; otherwise the full dotted path is
    spelled under each `sys.path` entry. The first directory holding any of the extensions
    wins as a whole, so a module's backends always come from one place.
    """
    parts:list = fullname.split(".")
    if path is None:
        # A top-level name: walk sys.path the way the backend hooks do.
        search:List[str] = list(sys.path)
    else:
        # A parent package's `__path__`: python has already descended into the package.
        search:List[str] = list(path)
        parts = parts[-1:]

    for directory in search:
        if not directory:
            continue
        file_path:str = directory.replace("\\", "/")
        for part in parts:
            file_path += f"/{part}"

        found:Dict[str, str] = {}
        for backend, extension in _BACKEND_EXTENSIONS:
            candidate:str = file_path + extension
            if os.path.isfile(candidate):
                found[backend] = candidate
        if found:
            return found
    return {}


def normalized_signature(args:List[Tuple[str, str]])->Tuple[Tuple[str, str], ...]:
    """A backend-independent parameter list: names + normalized pointer/type shapes.

    Address-space qualifiers are dropped (they are OpenCL spellings of "a pointer") and so
    are `const`/`volatile` -- they describe what a kernel does with the argument, and every
    backend accepts the same host values either way.
    """
    normalized:List[Tuple[str, str]] = []
    for name, type_str in args:
        words:str = " ".join(type_str.split())
        for qualifier in _QUALIFIER_WORDS:
            words = words.replace(qualifier, " ")
        pointer:bool = words.strip().endswith("*")
        base:str = words.strip()[:-1].strip() if pointer else words.strip()
        normalized.append((name, base + ("*" if pointer else "")))
    return tuple(normalized)


def _scalar_tables(backend:str)->Dict[str, Dict[str, type]]:
    """The scalar/vector ctypes tables one backend parses and checks types against."""
    if backend == "opencl":
        from .opencl.driver import CLInfo
        return {"basic": CLInfo.basic_types, "vec": CLInfo.vec_types}
    if backend == "cuda":
        from .cuda.driver import CUInfo
        return {"basic": CUInfo.basic_types, "vec": CUInfo.vec_types}
    if backend == "hip":
        from .hip.driver import HIPInfo
        return {"basic": HIPInfo.basic_types, "vec": HIPInfo.vec_types}
    if backend == "metal":
        return _metal_scalar_tables()
    return {}


# MSL's vector widths are 2, 3 and 4 -- there is no `float8`/`float16`, unlike
# OpenCL -- so `_metal_scalar_tables` keeps only those names. The width filter is
# what makes this safe to derive from OpenCL's table: the *types* are the very ones
# the other backends use (a vector is a layout of its components, and does not
# depend on which device reads it); only the names and the widths are MSL's.
_METAL_TABLES: Dict[str, Dict[str, type]] = {}

# The vector suffixes MSL declares. `2`/`3`/`4` only.
_METAL_VECTOR_WIDTHS: Tuple[str, ...] = ("2", "3", "4")

# The scalar spellings MSL declares, without the `f`-suffixed CUDA ones. `half` is
# left out on purpose: it is a storage type, and `kernel_import`'s type check binds
# a host array to a parameter by matching numpy dtypes, which a `half` parameter in
# a handwritten source cannot express.
_METAL_SCALAR_NAMES: Tuple[str, ...] = (
    "bool", "char", "uchar", "short", "ushort", "int", "uint",
    "long", "ulong", "float", "double",
)


def _metal_scalar_tables()->Dict[str, Dict[str, type]]:
    """Metal's scalar and vector tables, derived from the OpenCL ones.

    Built lazily on first use: the tables reach `pygpgpu/opencl/driver`, and
    `kernel_import` is imported on machines that only ever parse sources.

    The derived tables are what makes a `.metal` module type-checked like any other
    instead of silently skipping the check (`_declaration_accepts` answers `None`
    when a name is absent, which the caller reads as "cannot tell").
    """
    if _METAL_TABLES:
        return _METAL_TABLES

    from .opencl.driver import CLInfo

    basic: Dict[str, type] = {}
    for name in _METAL_SCALAR_NAMES:
        scalar = CLInfo.basic_types.get(name)
        if scalar is not None:
            basic[name] = scalar

    vec: Dict[str, type] = {}
    for name, vector in CLInfo.vec_types.items():
        # Keep only what MSL declares: a vector of a known scalar, 2/3/4 wide.
        for suffix in _METAL_VECTOR_WIDTHS:
            if not name.endswith(suffix):
                continue
            base: str = name[: -len(suffix)]
            if base in _METAL_SCALAR_NAMES:
                vec[name] = vector
                basic[name] = vector
            break

    _METAL_TABLES.update({"basic": basic, "vec": vec})
    return _METAL_TABLES


class _BackendSource:
    """One backend's source: its parsed metadata and the programs built from it.

    Parsing happens when the merged module is imported (cheap, cache-backed) so structures
    and signatures are known without a device; a `Program` is built on first use per
    device, because that is where the backend's compiler and context belong.
    """

    def __init__(self, backend:str, file_name:str, options:object, type_checked:bool):
        self.backend:str = backend
        self.file_name:str = file_name
        self.options:object = options
        self.type_checked:bool = type_checked
        self.parser:object = self._parse()
        self._lock = threading.Lock()
        self._programs:Dict[str, object] = {}
        self._kernels:Dict[str, object] = {}
        # The per-source compiler object each backend needs: NVRTC for CUDA, the clang
        # wrapper for HIP, and nothing extra for OpenCL (a context compiles directly).
        self._compiler:object = None

    def _parse(self)->object:
        # Importing `pygpgpu.<backend>` initializes that backend's driver, whose init()
        # records -- but does not raise -- a runtime the machine lacks (see the driver
        # layer). Nothing special is needed here: a parse is pure tree-sitter over a
        # file, and the driver's own failure surfaces when a call is actually made.
        module:object = importlib.import_module(_PARSER_MODULES[self.backend], "pygpgpu")
        parser:object = module.ProgramParser()
        parser.parse(self.file_name, options=self.options)
        return parser

    def kernel_infos(self)->Dict[str, object]:
        return self.parser.kernel_infos

    def struct_types(self)->Dict[str, type]:
        return self.parser._struct_types

    # ---------- kernels ----------

    def kernel(self, name:str, device_sel:Optional[str],
               bracket:Optional[Union[int, str, Tuple[int, ...]]])->object:
        """The backend's own `Kernel`, bound to a device and to the bracket's work size.

        The device is resolved through the decorator's resolver, so every spelling the
        `@kernel` bracket accepts works here too. ``bracket`` is what the caller wrote:
        a CUDA ``(blocks, threads)`` pair, a bare item count, or None for automatic.
        """
        device:object = self._device(device_sel)
        device_key:str = getattr(device, "unique_key", None) or str(device)
        key:str = f"{device_key}|{name}"
        with self._lock:
            kernel:object = self._kernels.get(key)
            if kernel is None:
                kernel = self._program(device)[name]
                self._kernels[key] = kernel
        if self.backend == "opencl":
            # A program belongs to a platform, and a kernel handle to the whole context;
            # the device it runs on is per-call state on the Kernel itself. Every other
            # backend gets its device from the context its Program was built in.
            kernel = kernel[device]
        if bracket is not None:
            kernel = kernel[_work_size_item(bracket)]
        return kernel

    def _device(self, device_sel:Optional[str])->object:
        from .kernel.decorator import _resolve_device
        device:object
        device, _key = _resolve_device(self.backend, device_sel)
        return device

    def _program(self, device:object)->object:
        """This backend's `Program` for one device, compiled on first use."""
        backend:str = self.backend
        if backend == "opencl":
            from .opencl.oop.program import Program
            platform_key:str = str(id(device.platform))
            program:Optional[object] = self._programs.get(platform_key)
            if program is None:
                program = Program(device.platform.default_context,
                                  type_checked=self.type_checked,
                                  program_parser=self.parser)
                self._programs[platform_key] = program
            return program
        if backend == "cuda":
            from .cuda.oop.program import Program
            device_key:str = getattr(device, "unique_key", None) or str(device)
            program = self._programs.get(device_key)
            if program is None:
                device.default_context.make_current()
                program = Program(self.file_name, options=self.options,
                                  type_checked=self.type_checked,
                                  nvrtc_program=self._nvrtc(device))
                self._programs[device_key] = program
            return program
        if backend == "hip":
            from .hip.oop.program import Program
            device_key = getattr(device, "unique_key", None) or str(device)
            program = self._programs.get(device_key)
            if program is None:
                device.default_context.make_current()
                program = Program(self.file_name, options=self.options,
                                  type_checked=self.type_checked,
                                  compiler=self._hip_compiler())
                self._programs[device_key] = program
            return program
        if backend == "metal":
            from .metal.oop.program import Program
            device_key = getattr(device, "unique_key", None) or str(device)
            program = self._programs.get(device_key)
            if program is None:
                program = Program(self.file_name, options=self.options,
                                  type_checked=self.type_checked, device=device)
                self._programs[device_key] = program
            return program
        raise ValueError(f"unknown backend '{backend}'")

    def _nvrtc(self, device:object)->object:
        """The NVRTC program for this source, parsed once and compiled per device."""
        from .cuda.oop.nvrtc_program import NVRTCProgram
        if self._compiler is None:
            self._compiler = NVRTCProgram(self.file_name, options=self.options,
                                          program_parser=self.parser)
        return self._compiler

    def _hip_compiler(self)->object:
        """The HIP compiler for this source, parsed once and compiled per architecture."""
        from .hip.oop.hip_compiler import HipCompiler
        if self._compiler is None:
            self._compiler = HipCompiler(self.file_name, options=self.options,
                                         program_parser=self.parser)
        return self._compiler


def _work_size_item(bracket:Union[int, str, Tuple[int, ...]])->Tuple[Optional[int], Optional[int]]:
    """The bracket as the backend `Kernel.__getitem__` wants it: a ``(blocks, threads)`` pair.

    A bare count (`k[n]`, `k[(n,)]`) is a total with an automatic group size, which every
    backend spells ``(count, None)``. Passing a one-entry tuple through unchanged would be
    read as a *device* selector by OpenCL's `Kernel`, and rejected by CUDA's.
    """
    if isinstance(bracket, tuple):
        if len(bracket) == 1:
            return (bracket[0], None)
        return (bracket[0], bracket[1])
    return (bracket, None)


class KernelImportWrapper:
    """A merged kernel: one object, one implementation per backend, `@kernel` usage.

    The bracket language is the decorator's own (see pygpgpu.kernel.decorator): a string
    selects a backend ("cuda", "opencl:gfx1036", or a bare device substring), a tuple of
    one or two entries states the work size in CUDA's `(blocks, threads)` terms, and
    brackets stack in any order. `submit`/`async_call` launch without blocking. A call
    without a bracket walks the documented probe order and takes the first backend whose
    declaration accepts the arguments.
    """

    def __init__(self, name:str, sources:Dict[str, "_BackendSource"],
                 declarations:Dict[str, Tuple[Tuple[str, str], ...]]):
        self._name:str = name
        self._sources:Dict[str, "_BackendSource"] = sources
        # backend -> its one declaration for this name (a C file cannot declare a name
        # twice, so a backend never owns two overloads of the same kernel).
        self._declarations:Dict[str, Tuple[Tuple[str, str], ...]] = dict(declarations)
        self._bracket_backend:Optional[str] = None
        self._bracket_device:Optional[str] = None
        self._bracket:object = None

    # ---------- introspection ----------

    @property
    def name(self)->str:
        return self._name

    @property
    def backends(self)->List[str]:
        """The backends that implement this kernel, in the documented probe order."""
        return [backend for backend, _ in _BACKEND_EXTENSIONS
                if backend in self._declarations]

    def sources(self)->Dict[str, str]:
        """The source file each backend implements this kernel from."""
        return {backend: self._sources[backend].file_name for backend in self.backends}

    def declarations(self)->Dict[str, Tuple[Tuple[str, str], ...]]:
        """backend -> the parameter list it declares for this kernel."""
        return dict(self._declarations)

    def __repr__(self)->str:
        return (f"KernelImportWrapper({self._name}, "
                f"backends=[{', '.join(self.backends)}])")

    # ---------- bracket handling ----------

    def _clone(self)->"KernelImportWrapper":
        """A fresh wrapper carrying this one's state; a bracket never mutates the original.

        A module attribute is shared by every caller, so the brackets build copies and the
        attribute itself stays bracket-free.
        """
        wrapper:KernelImportWrapper = KernelImportWrapper.__new__(KernelImportWrapper)
        wrapper.__dict__.update(self.__dict__)
        return wrapper

    def __getitem__(self, item:Union[int, str, Tuple[int, ...]])->"KernelImportWrapper":
        from .kernel.decorator import _parse_selector, _parse_work_item

        if isinstance(item, str):
            backend, device = _parse_selector(item)
            if backend is None and device is None:
                raise ValueError("empty kernel bracket selector")
            if backend is not None and backend not in self._declarations:
                # The failure the bracket is for: this backend has no implementation of
                # the name, so say so before any argument is looked at.
                raise KeyError(
                    f"kernel '{self._name}' has no {backend} implementation "
                    f"(implemented by: {', '.join(self.backends) or 'none'})")
            if backend is None:
                # A bare device substring: keep the first implementing backend whose
                # device list answers to it, so `k["geforce"]` still picks a backend.
                chosen:Optional[str] = None
                for candidate in self.backends:
                    try:
                        self._sources[candidate]._device(device)
                    except Exception:
                        continue
                    chosen = candidate
                    break
                if chosen is None:
                    raise KeyError(
                        f"no device matches '{item}' on any backend implementing "
                        f"'{self._name}' ({', '.join(self.backends) or 'none'})")
                backend = chosen
            wrapper:KernelImportWrapper = self._clone()
            wrapper._bracket_backend = backend
            wrapper._bracket_device = device
            return wrapper

        if isinstance(item, tuple) and len(item) == 2 and all(isinstance(part, str)
                                                              for part in item):
            # Two strings are two selectors, not a work size.
            raise TypeError(
                f"kernel bracket must be a device selector string or a work size, "
                f"got ({item[0]!r}, {item[1]!r})")
        # Validate with the decorator's own parser so a malformed bracket is refused in
        # the same words `@kernel` uses; what travels onward is the caller's own spelling,
        # which is the `(blocks, threads)` form every backend `Kernel` takes.
        _parse_work_item(item)
        wrapper = self._clone()
        wrapper._bracket = item
        return wrapper

    # ---------- declaration matching ----------

    def _bind_arguments(self, signature:Tuple[Tuple[str, str], ...],
                        args:Tuple[object, ...],
                        kwargs:Dict[str, object])->Optional[List[object]]:
        """The arguments in declaration order, or None when they do not fit the signature."""
        names:List[str] = [name for name, _ in signature]
        if len(args) > len(names):
            return None
        values:Dict[str, object] = dict(zip(names, args))
        for key, value in kwargs.items():
            if key not in names or key in values:
                return None
            values[key] = value
        if len(values) != len(names):
            return None
        return [values[name] for name in names]

    def _declaration_accepts(self, backend:str, signature:Tuple[Tuple[str, str], ...],
                             args:Tuple[object, ...],
                             kwargs:Dict[str, object])->Optional[bool]:
        """Whether the declaration can take these arguments; None when it cannot be told.

        A pointer parameter accepts an array whose dtype the parsed element type names; a
        scalar accepts a number. A structure or a vector passed by value is accepted as it
        is -- the shape of the value is the backend's own business, not this dispatch's.
        """
        import numpy as _np
        bound:Optional[List[object]] = self._bind_arguments(signature, args, kwargs)
        if bound is None:
            return False
        tables:Dict[str, Dict[str, type]] = _scalar_tables(backend)
        scalar_types:Dict[str, type] = tables.get("basic", {})
        vec_types:Dict[str, type] = tables.get("vec", {})
        uncertain:bool = False
        for (_name, type_str), value in zip(signature, bound):
            pointer:bool = type_str.strip().endswith("*")
            base:str = (type_str.strip()[:-1].strip() if pointer else type_str.strip())
            if pointer:
                if not isinstance(value, _np.ndarray):
                    return False
                expected:object = (scalar_types.get(base) or vec_types.get(base))
                if expected is None:
                    uncertain = True
                    continue
                try:
                    if _np.dtype(expected) != value.dtype:
                        return False
                except TypeError:
                    uncertain = True
            else:
                expected = scalar_types.get(base)
                if expected is None:
                    uncertain = True
                    continue
                try:
                    _np.dtype(expected).type(value)
                except (TypeError, ValueError):
                    return False
        return None if uncertain else True

    def _select_backend(self, args:Tuple[object, ...],
                        kwargs:Dict[str, object])->Tuple[str, Optional[bool]]:
        """The backend this call runs on, and whether its declaration definitely fits.

        With a bracket the backend is named, so it is used even when the check cannot
        confirm the types (the check is a dispatch aid, not a gate); without one the
        documented probe order supplies the first backend whose declaration accepts the
        arguments.
        """
        if self._bracket_backend is not None:
            backend:str = self._bracket_backend
            if backend not in self._declarations:
                raise KeyError(
                    f"kernel '{self._name}' has no {backend} implementation "
                    f"(implemented by: {', '.join(self.backends) or 'none'})")
            return backend, self._declaration_accepts(
                backend, self._declarations[backend], args, kwargs)

        from .kernel.decorator import DEFAULT_BACKEND_ORDER
        from .kernel.decorator import _probe_backend
        untried:List[str] = []
        for backend in DEFAULT_BACKEND_ORDER:
            signature:Optional[Tuple[Tuple[str, str], ...]] = self._declarations.get(backend)
            if signature is None:
                continue
            if not args and not kwargs:
                return backend, None
            fits:Optional[bool] = self._declaration_accepts(backend, signature, args, kwargs)
            if fits is None or fits:
                # The declaration fits, but the backend still has to exist on this
                # machine: a bare call walks the probe order to find a backend it can
                # actually run on, so a matching declaration whose runtime is missing
                # (no ROCm here, say) must fall through rather than raise.
                if _probe_backend(backend):
                    return backend, fits
                untried.append(backend)
                continue
            untried.append(backend)
        if untried:
            raise TypeError(
                f"{self._name}() was given arguments no declaration accepts; declared as: "
                + "; ".join(f"{backend} {_signature_text(self._declarations[backend])}"
                            for backend in untried))
        raise KeyError(
            f"kernel '{self._name}' has no implementation on any backend "
            f"({', '.join(self._declarations) or 'none'})")

    # ---------- calls ----------

    def _ready_kernel(self, args:Tuple[object, ...], kwargs:Dict[str, object])->object:
        backend:str = self._select_backend(args, kwargs)[0]
        source:_BackendSource = self._sources[backend]
        return source.kernel(self._name, self._bracket_device, self._bracket)

    def __call__(self, *args:object, **kwargs:object)->object:
        return self._ready_kernel(args, kwargs)(*args, **kwargs)

    def submit(self, *args:object, **kwargs:object)->object:
        """Non-blocking launch; the bracket language matches the call."""
        return self._ready_kernel(args, kwargs).submit(*args, **kwargs)

    def async_call(self, *args:object, **kwargs:object)->object:
        """Non-blocking asyncio launch; the bracket language matches the call."""
        return self._ready_kernel(args, kwargs).async_call(*args, **kwargs)


def _signature_text(signature:Tuple[Tuple[str, str], ...])->str:
    return "(" + ", ".join(f"{name}: {type_str}" for name, type_str in signature) + ")"


class _MergedModuleLoader:
    """Parses every backend's source at import and publishes the merged attributes."""

    def __init__(self, fullname:str, sources:Dict[str, str],
                 options:Dict[str, object], type_checked:bool, generate_pyi:bool):
        self.fullname:str = fullname
        self.sources:Dict[str, str] = sources
        self.options:Dict[str, object] = options
        self.type_checked:bool = type_checked
        self.generate_pyi:bool = generate_pyi
        self.backends:Dict[str, _BackendSource] = {}
        self.kernels:Dict[str, KernelImportWrapper] = {}
        self.structs:Dict[str, type] = {}

    def create_module(self, spec:object)->None:
        # The default module creation is what a merged module wants (a plain ModuleType);
        # returning None asks the import machinery to do exactly that.
        return None

    def exec_module(self, module:types.ModuleType)->None:
        # Parse each backend's source (cache-backed). A source that fails to parse is an
        # error the user must see, so no backend is silently dropped.
        for backend, _extension in _BACKEND_EXTENSIONS:
            file_name:Optional[str] = self.sources.get(backend)
            if file_name is None:
                continue
            self.backends[backend] = _BackendSource(
                backend, file_name, self.options.get(backend), self.type_checked)

        # Structures: the first declaration wins. Two backends describe the same layouts,
        # and a type only one of them declares is still published.
        for backend, _extension in _BACKEND_EXTENSIONS:
            source:Optional[_BackendSource] = self.backends.get(backend)
            if source is None:
                continue
            for struct_name, struct_type in source.struct_types().items():
                if struct_name not in self.structs:
                    self.structs[struct_name] = struct_type

        # Kernels: one attribute per name. Same name in several files = one function with
        # several implementations; different parameter lists = the stub's overloads.
        declarations:Dict[str, Dict[str, Tuple[Tuple[str, str], ...]]] = {}
        for backend, _extension in _BACKEND_EXTENSIONS:
            source = self.backends.get(backend)
            if source is None:
                continue
            for kernel_name, kernel_info in source.kernel_infos().items():
                signature:Tuple[Tuple[str, str], ...] = tuple(
                    (arg.name, arg.type_str) for arg in kernel_info.args.values())
                declarations.setdefault(kernel_name, {})[backend] = signature

        for kernel_name, by_backend in declarations.items():
            self.kernels[kernel_name] = KernelImportWrapper(
                kernel_name, self.backends, by_backend)

        for struct_name, struct_type in self.structs.items():
            setattr(module, struct_name, struct_type)
        for kernel_name, wrapper in self.kernels.items():
            setattr(module, kernel_name, wrapper)

        if self.generate_pyi:
            from .kernel_import_stub import generate_pyi_for
            generate_pyi_for(self.sources, self.backends, self.kernels)


class kernel_import:
    """The import hook: `kernel_import.install()`, then import the module."""

    _lock = threading.Lock()
    _installed:bool = False
    _config:Dict[str, object] = {
        "type_checked": False,
        "generate_pyi": True,
        # Per-backend build options; None keeps each backend's own default.
        "options": {"cuda": None, "hip": None, "opencl": None},
    }

    @staticmethod
    def install()->None:
        """Install the finder. Idempotent, and it goes in front of the backend hooks."""
        with kernel_import._lock:
            if kernel_import._installed:
                return
            sys.meta_path.insert(0, _KernelImportFinder())
            kernel_import._installed = True

    @staticmethod
    def uninstall()->None:
        with kernel_import._lock:
            for i in range(len(sys.meta_path) - 1, -1, -1):
                if isinstance(sys.meta_path[i], _KernelImportFinder):
                    del sys.meta_path[i]
            kernel_import._installed = False

    @staticmethod
    def config(type_checked:bool=False, generate_pyi:bool=True,
               options:Optional[Dict[str, object]]=None)->None:
        """Configure the hook. `options` maps backend -> BuildOptions (None = default)."""
        with kernel_import._lock:
            kernel_import._config["type_checked"] = type_checked
            kernel_import._config["generate_pyi"] = generate_pyi
            if options is not None:
                kernel_import._config["options"].update(options)


class _KernelImportFinder:
    """The meta-path finder: resolve a name to same-named .cl/.cu/.hip sources."""

    def find_spec(self, fullname:str, path:Optional[List[str]] = None,
                  target:object = None)->Optional[object]:
        import importlib.util
        sources:Dict[str, str] = _source_paths_for(fullname, path)
        if not sources:
            return None
        config:Dict[str, object] = kernel_import._config
        loader = _MergedModuleLoader(fullname, sources, config["options"],
                                     config["type_checked"], config["generate_pyi"])
        # The module's file location is whichever source comes first in the probe order;
        # a merged module is a leaf, so it carries no `__path__`.
        first_source:str = next(sources[backend] for backend, _ in _BACKEND_EXTENSIONS
                                if backend in sources)
        return importlib.util.spec_from_file_location(fullname, first_source, loader=loader)
