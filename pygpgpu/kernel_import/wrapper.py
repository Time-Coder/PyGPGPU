# The module object a handwritten kernel file imports as.
from __future__ import annotations
from .source import _BackendSource
from .tables import _signature_text
from .tables import (
    _BACKEND_EXTENSIONS,
    _scalar_tables,
)
import types
from typing import Dict, List, Optional, Tuple, Union



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
        from ..kernel.decorator import _parse_selector, _parse_work_item

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
                # An integer parameter is range-checked against the width the source
                # declares, which is the same rule a @kernel scalar follows. The old test
                # asked numpy to *convert* the value, and numpy 2 raises `OverflowError`
                # for an out-of-range one -- an exception neither this check nor a caller
                # was prepared for, so a handwritten `uchar` parameter accepted `256` by
                # accident on one numpy and raised an unhandled error on another.
                from ..kernel.translator import (  # noqa: PLC0415 (import cycle)
                    check_integer_range,
                    ctypes_device_type_name,
                )

                declared:Optional[str] = ctypes_device_type_name(expected)
                if declared is not None and isinstance(value, int) and not isinstance(value, bool):
                    try:
                        check_integer_range(int(value), declared)
                    except ValueError:
                        return False
                    continue
                try:
                    _np.dtype(expected).type(value)
                except (TypeError, ValueError, OverflowError):
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

        from ..kernel.decorator import DEFAULT_BACKEND_ORDER
        from ..kernel.decorator import _probe_backend
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

    def _check_scalar_widths(self, backend:str, args:Tuple[object, ...],
                             kwargs:Dict[str, object])->None:
        """Refuse a scalar the declaration's own width cannot hold.

        `_declaration_accepts` decides which backend a bare call goes to, and a bracket names
        the backend outright, so neither path is a gate: `k["opencl"](256, ...)` on a
        parameter declared `uchar` reached the device and wrapped to `0`. The declared width
        is the caller's contract with that source, so a value outside it is refused here,
        once, before any backend launches.
        """
        signature:Optional[Tuple[Tuple[str, str], ...]] = self._declarations.get(backend)
        if signature is None:
            return
        bound:Optional[List[object]] = self._bind_arguments(signature, args, kwargs)
        if bound is None:
            return
        tables:Dict[str, Dict[str, type]] = _scalar_tables(backend)
        scalar_types:Dict[str, type] = tables.get("basic", {})
        from ..kernel.translator import (  # noqa: PLC0415 (import cycle)
            check_integer_range,
            ctypes_device_type_name,
        )

        for (name, type_str), value in zip(signature, bound):
            if type_str.strip().endswith("*"):
                continue                 # a buffer's dtype is checked by the dispatch
            expected:Optional[type] = scalar_types.get(type_str.strip())
            if expected is None:
                continue
            declared:Optional[str] = ctypes_device_type_name(expected)
            if declared is None or not isinstance(value, int) or isinstance(value, bool):
                continue
            check_integer_range(int(value), declared, name)

    def _ready_kernel(self, args:Tuple[object, ...], kwargs:Dict[str, object])->object:
        backend:str = self._select_backend(args, kwargs)[0]
        self._check_scalar_widths(backend, args, kwargs)
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
            from ..kernel_import_stub import generate_pyi_for
            generate_pyi_for(self.sources, self.backends, self.kernels)
