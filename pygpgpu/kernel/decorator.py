from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..utils import local_size_for_global, detect_work_size, device_limits
from .python_backend import PythonKernel
from .translator import (
    FunctionTranslator,
    PyKernelSyntaxError,
    infer_ctype,
    scan_self_attrs,
    short_hash,
    type_sig,
)


# ---------- bracket selector parsing ----------

_BACKEND_ALIASES:Dict[str, str] = {
    "opencl": "opencl", "ocl": "opencl", "cl": "opencl",
    "cuda": "cuda", "cu": "cuda",
    "hip": "hip", "rocm": "hip", "amd": "hip",
    # The pure-python backend: no translation, no device, serial execution on the host.
    "python": "python", "py": "python", "host": "python",
}

# Probe order when no backend is specified: take the first backend that enumerates at least one device
DEFAULT_BACKEND_ORDER:Tuple[str, ...] = ("cuda", "hip", "opencl")

_probed_backend:Optional[str] = None


def _backend_devices(backend:str)->List[Any]:
    """Enumerate the backend's device list (order = numbering used by plain-numeric selectors); raises when the runtime is missing."""
    if backend == "opencl":
        from ..opencl.oop.platforms import Platforms
        return [d for platform in Platforms for d in platform.devices]
    if backend == "hip":
        from ..hip.oop.devices import Devices
        return list(Devices)
    if backend == "cuda":
        from ..cuda.oop.devices import Devices
        return list(Devices)
    raise ValueError(f"unknown backend '{backend}'")


def _probe_backend(backend:str)->Optional[str]:
    """Probe whether a backend is usable: enumerating at least one device counts as usable.

    Backend packages load their runtime (nvcuda/amdhip64/OpenCL) only at import time; failures are treated as unavailable.
    """
    try:
        if _backend_devices(backend):
            return backend
    except Exception:
        pass
    return None


def _default_backend()->str:
    """Probe the first usable backend per DEFAULT_BACKEND_ORDER; the result is cached in-process."""
    global _probed_backend
    if _probed_backend is None:
        for backend in DEFAULT_BACKEND_ORDER:
            if _probe_backend(backend) is not None:
                _probed_backend = backend
                break
        else:
            _probed_backend = "opencl"
    return _probed_backend


def _parse_selector(sel:str)->Tuple[Optional[str], Optional[str]]:
    """Parse selectors of the form "backend" / "backend:device" / "device".

    Returns (backend, device); unspecified parts are None (filled in by defaults / earlier brackets).
    """
    if not isinstance(sel, str):
        raise TypeError(f"kernel bracket selector must be a string like 'opencl:gfx1036', got {type(sel).__name__}")
    sel = sel.strip()
    if not sel:
        raise ValueError("empty kernel bracket selector")
    if ":" in sel:
        head, rest = sel.split(":", 1)
        backend:Optional[str] = _BACKEND_ALIASES.get(head.strip().lower())
        if backend is not None:
            return backend, (rest.strip() or None)
        # the part after the colon is not a backend name -> the whole string is a device identifier (e.g. "geforce:0")
        return None, sel
    backend = _BACKEND_ALIASES.get(sel.lower())
    if backend is not None:
        return backend, None
    return None, sel


def _parse_work_item(item:Any)->Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Parse a work-size bracket: int / (global,) / (global, local); None means automatic."""
    def norm(v:Any)->Optional[Tuple[int, ...]]:
        if v is None:
            return None
        if isinstance(v, int):
            return (v,)
        if isinstance(v, tuple):
            t:Tuple[int, ...] = tuple(int(x) for x in v)
            if not t:
                raise ValueError("work size tuple cannot be empty")
            return t
        raise TypeError(f"work size entry must be int or tuple of ints, got {type(v).__name__}")

    if isinstance(item, int):
        return norm(item), None
    if isinstance(item, tuple):
        if len(item) == 1:
            return norm(item[0]), None
        if len(item) == 2:
            return norm(item[0]), norm(item[1])
    raise TypeError("work size bracket must be n, (global,), or (global, local); use None for auto")


def _resolve_device(backend:str, device_sel:Optional[str])->Tuple[Any, str]:
    """Resolve a device within a backend: None/index/name substring ("name:ordinal" disambiguates); returns (device object, cache key)."""
    devices:List[Any] = _backend_devices(backend)

    if not devices:
        raise RuntimeError(f"no {backend} devices found")

    if device_sel is None or device_sel.strip() == "":
        return devices[0], devices[0].name

    sel:str = device_sel.strip()
    index:int = 0
    if ":" in sel:
        name_part, idx_part = sel.rsplit(":", 1)
        if idx_part.strip().isdigit() and name_part.strip():
            sel = name_part.strip()
            index = int(idx_part)

    if sel.isdigit():
        i:int = int(sel)
        if not (0 <= i < len(devices)):
            raise ValueError(f"device index {i} out of range (0..{len(devices)-1}) for backend '{backend}'")
        return devices[i], devices[i].name

    low:str = sel.lower()
    matches:List[Any] = [d for d in devices
                         if low in d.name.lower() or low in getattr(d, "vendor", "").lower()
                         or low in getattr(d, "arch", "").lower()]
    if not matches:
        raise ValueError(
            f"no {backend} device matches '{device_sel}'; available: " + ", ".join(d.name for d in devices))
    if index >= len(matches):
        raise ValueError(f"device match index {index} out of range for '{sel}' ({len(matches)} matches)")
    return matches[index], matches[index].name


def available_devices()->List[str]:
    """List all available "backend:device" selector strings, ordered by DEFAULT_BACKEND_ORDER.

    Returns a list like ["cuda:0", "hip:0", "opencl:0", "opencl:1", ...]; the numbering
    matches the device order used by plain-numeric selectors; backends whose runtime is
    missing are skipped. Results can be passed directly to `get_device()` or kernel brackets.
    """
    selectors:List[str] = []
    for backend in DEFAULT_BACKEND_ORDER:
        try:
            devices:List[Any] = _backend_devices(backend)
        except Exception:
            continue
        for i in range(len(devices)):
            selectors.append(f"{backend}:{i}")
    return selectors


def get_device(selector:str)->Any:
    """Return the Device object for a "backend:device" selector string (the backend's own Device class).

    The backend part supports aliases (opencl/ocl/cl, cuda/cu, hip/rocm/amd); the device part is an
    index or a name/vendor/arch substring ("name:ordinal" disambiguates); with only the device part,
    the default backend (probed, cuda -> hip -> opencl) is used.
    """
    backend:Optional[str]
    device_sel:Optional[str]
    backend, device_sel = _parse_selector(selector)
    if backend is None:
        backend = _default_backend()
    return _resolve_device(backend, device_sel)[0]


class _KernelCall:
    """Bracket-configuration form: k["cuda"][(64, 8)](args) / k[(64, 8)]["cuda"](args), any order."""

    def __init__(self, kernel:"PyKernel", instance:Any,
                 backend:Optional[str]=None, device:Optional[str]=None,
                 global_size:Optional[Tuple[int, ...]]=None, local_size:Optional[Tuple[int, ...]]=None):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance
        self._backend:Optional[str] = backend
        self._device:Optional[str] = device
        self._global_size:Optional[Tuple[int, ...]] = global_size
        self._local_size:Optional[Tuple[int, ...]] = local_size

    def __getitem__(self, item:Any)->"_KernelCall":
        if isinstance(item, str):
            backend, device = _parse_selector(item)
            return _KernelCall(
                self._kernel, self._instance,
                (backend if backend is not None else self._backend),
                (device if device is not None else self._device),
                self._global_size, self._local_size)
        global_size, local_size = _parse_work_item(item)
        return _KernelCall(
            self._kernel, self._instance, self._backend, self._device,
            (global_size if global_size is not None else self._global_size),
            (local_size if local_size is not None else self._local_size))

    def __call__(self, *args:Any)->Any:
        instance:Any = self._instance
        call_args:Tuple[Any, ...] = args
        if self._kernel._is_method and instance is None:
            # class form: Klass.method["cuda"](instance, args...)
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance = args[0]
            call_args = args[1:]
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device, self._global_size, self._local_size)

    def _launch(self, launch:str, args:Tuple[Any, ...])->Any:
        """Shared body of the three launch styles, honouring the bracket selectors."""
        instance:Any = self._instance
        call_args:Tuple[Any, ...] = args
        if self._kernel._is_method and instance is None:
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance = args[0]
            call_args = args[1:]
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device,
            self._global_size, self._local_size, launch=launch)

    def submit(self, *args:Any)->concurrent.futures.Future:
        """Non-blocking launch of the bracketed form, e.g. ``k["cuda"].submit(...)``."""
        return self._launch("submit", args)

    def async_call(self, *args:Any)->asyncio.Future:
        """Non-blocking asyncio launch of the bracketed form, e.g. ``k["cuda"].async_call(...)``."""
        return self._launch("async", args)


class _BoundKernelMethod:
    """Bound instance-method form: instance.kernel_method(...)"""

    def __init__(self, kernel:"PyKernel", instance:Any):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance

    def __getitem__(self, item:Any)->_KernelCall:
        return _KernelCall(self._kernel, self._instance)[item]

    def __call__(self, *args:Any)->Any:
        return self._kernel._invoke(self._instance, args, None, None, None, None)

    def submit(self, *args:Any)->concurrent.futures.Future:
        """Non-blocking launch: ``instance.kernel_method.submit(...)``."""
        return self._kernel._invoke(self._instance, args, None, None, None, None, launch="submit")

    def async_call(self, *args:Any)->asyncio.Future:
        """Non-blocking asyncio launch: ``instance.kernel_method.async_call(...)``."""
        return self._kernel._invoke(self._instance, args, None, None, None, None, launch="async")


class PyKernel:
    """Implementation of the `@kernel` decorator: translates the python function into backend source at call time, then compiles and runs it.

    Decorating class methods is supported: self is dropped during translation, self.xxx attribute
    references in the method body become extra parameters _self_xxx, filled from the instance at call
    time; staticmethod is supported too (either decoration order).

    Backend/device/work size are given with brackets at call time (same style as handwritten kernels);
    0, 1 or 2 brackets may be used, in any order::

        k(a, out)                       # default backend (probed, cuda -> hip -> opencl)/device, work size automatic
        k["cuda"](a, out)               # pick the backend
        k["opencl:gfx1036"](a, out)     # backend + device (index or name substring, "name:ordinal" disambiguates)
        k[(64, 8)](a, out)              # global/local (None means automatic)
        k["cuda"][(64, 8)](a, out)      # combined, any order
    """

    def __init__(self, func:Callable, backend:str):
        if isinstance(func, classmethod):
            raise PyKernelSyntaxError("@kernel cannot decorate classmethods (cls semantics are host-only)")
        if isinstance(func, staticmethod):
            func = func.__func__

        self.raw_func:Callable = func
        self.backend:str = backend
        self.is_pykernel:bool = True
        self.__name__:str = func.__name__
        self.__doc__:str = func.__doc__ or ""

        arg_names:List[str] = list(func.__code__.co_varnames[:func.__code__.co_argcount])
        self._is_method:bool = bool(arg_names and arg_names[0] == "self")
        self._self_param:str = (arg_names[0] if self._is_method else "")
        self._self_attrs:List[str] = (scan_self_attrs(func, self._self_param) if self._is_method else [])

        self._cache:Dict[str, Tuple[Any, List[Tuple[int, int]], List[Tuple[int, int]], List[int]]] = {}
        # The python backend executes the body as-is, so there is nothing to specialize or
        # cache: one runner per kernel object serves every call.
        self._python_runner:Any = None
        self._source_sig:str = self._source_signature()

    def __get__(self, obj:Any, objtype:Optional[type]=None)->Any:
        # descriptor protocol: instance access returns the bound form, class access/static methods return self
        if obj is None or not self._is_method:
            return self
        return _BoundKernelMethod(self, obj)

    def __getitem__(self, item:Any)->_KernelCall:
        return _KernelCall(self, None)[item]

    def __call__(self, *args:Any)->Any:
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            return self._invoke(args[0], args[1:], None, None, None, None)
        return self._invoke(None, args, None, None, None, None)

    def submit(self, *args:Any)->concurrent.futures.Future:
        """Launch without waiting, returning a ``concurrent.futures.Future``.

        Same arguments and same backend/work-size selection as calling the kernel
        directly; the difference is that this returns as soon as the launch is
        enqueued instead of blocking until the kernel finishes and its results
        are read back. The future completes when every transfer has landed, so
        the output arrays are only safe to read from its result (or after
        ``.result()``/``.wait()``).
        """
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            return self._invoke(args[0], args[1:], None, None, None, None, launch="submit")
        return self._invoke(None, args, None, None, None, None, launch="submit")

    def async_call(self, *args:Any)->asyncio.Future:
        """Launch without waiting, returning an ``asyncio.Future``.

        The asyncio counterpart of ``submit``; use ``await`` on the result. The
        kernel is enqueued immediately, so awaiting only waits for completion,
        and the returned future resolves once the read-back has finished.
        """
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            return self._invoke(args[0], args[1:], None, None, None, None, launch="async")
        return self._invoke(None, args, None, None, None, None, launch="async")

    def _invoke(self, instance:Any, args:Tuple[Any, ...],
                backend_sel:Optional[str], device_sel:Optional[str],
                global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                launch:str = "call")->Any:
        backend:str = (backend_sel or self.backend)
        if backend == "python":
            return self._invoke_python(instance, args, device_sel, global_size, local_size, launch)
        device_obj:Any
        device_key:str
        device_obj, device_key = _resolve_device(backend, device_sel)
        device_max_group, device_max_items = device_limits(device_obj)

        self_values:List[Any] = []
        if self._is_method:
            if instance is None:
                raise TypeError(f"{self.__name__}() requires an instance")
            for name in self._self_attrs:
                if not hasattr(instance, name):
                    raise PyKernelSyntaxError(
                        f"instance of {type(instance).__name__} has no attribute '{name}' "
                        f"required by kernel {self.__name__}")
                value:Any = getattr(instance, name)
                if callable(value) and not isinstance(value, np.ndarray):
                    raise PyKernelSyntaxError(
                        f"self.{name} is a method; calling self.{name}() inside a kernel is not supported")
                self_values.append(value)

        arg_types:List[Any] = [infer_ctype(a) for a in args]
        self_types:List[Any] = []
        for name, value in zip(self._self_attrs, self_values):
            try:
                self_types.append(infer_ctype(value))
            except PyKernelSyntaxError as e:
                raise PyKernelSyntaxError(f"self.{name}: {e}") from None

        sig:str = type_sig(arg_types + self_types)
        cache_key:str = f"{backend}|{device_key}|{sig}"
        entry = self._cache.get(cache_key)
        if entry is None:
            entry = self._compile(args, arg_types, self_types, sig, backend, device_obj)
            self._cache[cache_key] = entry
        kernel_obj, shape_sources, work_size_dims, local_sizes, uses_global_id = entry

        full:List[Any] = list(args) + list(self_values)
        launch_args:List[Any] = self._build_launch_args(full, arg_types + self_types, shape_sources)
        if backend == "opencl":
            # cuda local memory is a static __shared__ declaration; only opencl needs the byte size passed
            for local_bytes in local_sizes:
                launch_args.append(local_bytes)
        if global_size is None and work_size_dims:
            # A dim of -1 is an explicit gpu.work_size(x) declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give.
            extents_list:List[int] = []
            for pi, dim in work_size_dims:
                shape:Tuple[int, ...] = tuple(int(s) for s in full[pi].shape)
                if dim >= 0:
                    extents_list.append(shape[dim])
                else:
                    extents_list.extend(shape)
            if len(extents_list) > 3:
                raise ValueError(
                    f"gpu.work_size({self.__name__}) declared {len(extents_list)} dimensions "
                    f"from an array of shape {tuple(int(s) for s in full[work_size_dims[0][0]].shape)}; "
                    f"at most 3 are supported")
            extents:Tuple[int, ...] = tuple(extents_list)
            if local_size is None:
                global_size, local_size = detect_work_size(
                    256, extents, max_work_group_size=device_max_group,
                    max_work_item_sizes=device_max_items, allow_padding=True)
            elif len(local_size) == len(extents):
                # A user-supplied local size wins; round each extent up to a multiple of it.
                global_size = tuple(-(-extent // l) * l for extent, l in zip(extents, local_size))
            else:
                global_size = extents
        if global_size is None and local_size is not None:
            raise ValueError(
                "local work size requires a global size: specify both in the bracket "
                "k[(global, local)] or rely on a magic loop to derive the global size")
        if global_size is None and uses_global_id:
            # A gid-style kernel has no loop to read the domain from. Falling back to the
            # backend's buffer-shape guess used to silently launch a tiny grid (e.g. one work
            # item for a 100000-element reduction) and produce a wrong result, so refuse.
            raise ValueError(
                f"{self.__name__}() dispatches on gpu.get_global_id but declares no iteration "
                f"domain, so the work size cannot be derived. Either give it a magic loop "
                f"(`for i in range(x.shape[0]): ...`), declare `gpu.work_size(x)` in the body, "
                f"or pass a bracket: {self.__name__}[(total, local)](...)")
        if global_size is not None:
            # Uniform total-work-item semantics (OpenCL/SYCL): global = total number of work items, must be divisible by local;
            # cuda's gridDim is a block count, derived from total/local, transparent to the user
            if local_size is None:
                local_size = local_size_for_global(
                    global_size, max_work_group_size=device_max_group,
                    max_work_item_sizes=device_max_items)
            if len(local_size) != len(global_size):
                raise ValueError(
                    f"global {global_size} and local {local_size} work sizes must have the same dimension")
            for g, l in zip(global_size, local_size):
                if g % l != 0:
                    raise ValueError(
                        f"global work size {global_size} must be divisible by local {local_size}; "
                        f"use None for an automatically derived local size")
            if backend in ("cuda", "hip"):
                # cuda/hip gridDim is a block count, derived from total/local
                grid = tuple(g // l for g, l in zip(global_size, local_size))
                kernel_obj = kernel_obj[(grid, local_size)]
            else:
                kernel_obj = kernel_obj[(global_size, local_size)]
        if launch == "submit":
            return kernel_obj.submit(*launch_args)
        if launch == "async":
            return kernel_obj.async_call(*launch_args)
        return kernel_obj(*launch_args)

    def _invoke_python(self, instance:Any, args:Tuple[Any, ...],
                       device_sel:Optional[str],
                       global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                       launch:str)->Any:
        """Run the body in the Python interpreter (no translation, no device).

        The backend is device-free, so a device selector in the bracket is rejected
        rather than ignored; the work size and launch style mean what they mean on the
        GPU backends. The runner decides what a work item is (see python_backend.py): a
        magic-loop kernel runs once as written, a gid-style kernel runs once per work
        item of its domain.
        """
        if device_sel is not None:
            raise ValueError(
                f"the python backend has no devices, so '{self.__name__}[\"python:{device_sel}\"]' "
                f"has no meaning; use {self.__name__}[\"python\"]")
        if self._is_method and instance is None:
            raise TypeError(f"{self.__name__}() requires an instance")
        runner:Any = self._python_runner
        if runner is None:
            runner = PythonKernel(self.raw_func, self._is_method, self._self_param)
            self._python_runner = runner
        return runner.call(instance, args, global_size, local_size, launch=launch)

    # ---------- source generation ----------

    def _source_signature(self)->str:
        """Python source signature: source changes -> regenerate (recompile when cached content differs)."""
        try:
            source:str = inspect.getsource(self.raw_func)
        except OSError:
            source = ""
        return short_hash(f"{source}\nbackend={self.backend}")

    def _resolve_global(self, name:str)->Any:
        return self.raw_func.__globals__.get(name)

    def _c_kernel_name(self, sig:str, backend:str)->str:
        return f"pyk_{self.__name__}_{short_hash(sig + '_' + backend)}"

    def _compile(self, args:List[Any], arg_types:List[Any], self_types:List[Any], sig:str,
                 backend:str, device_obj:Any)->Tuple[Any, List[str]]:
        if backend in ("cuda", "hip"):
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise PyKernelSyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead")

        c_name:str = self._c_kernel_name(sig, backend)
        device_cache:Dict[Any, Any] = {}
        self_attr_types:Dict[str, Any] = dict(zip(self._self_attrs, self_types))

        # hip accepts CUDA syntax (threadIdx/atomicAdd/make_float4 etc.), so codegen goes through cuda
        codegen_backend:str = ("cuda" if backend == "hip" else backend)
        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), codegen_backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types)
        translator.translate()
        kernel_code:str = translator.render_kernel(c_name)

        # Device helper functions must be emitted in dependency order: C requires a callee to
        # be declared before its caller, and the cache is filled depth-first (a nested helper
        # is finished *after* the helper that calls it). Emit leaves first.
        dependency_codes:List[str] = [entry["code"] for entry in _dependency_order(device_cache)]

        source:str = ("\n".join(dependency_codes) + "\n\n" + kernel_code) if dependency_codes else kernel_code

        source_path:str = self._write_source(sig, source, backend)
        kernel:Any = self._backend_compile(source_path, c_name, backend, device_obj)

        # Atomic targets: static analysis cannot recognize pointer-cast writes, so force them to writable
        from ..constants import KernelFlags
        for target_name in translator.atomic_targets:
            if target_name in kernel._args:
                kernel._args[target_name].kernel_flags = (kernel._args[target_name].kernel_flags | KernelFlags.Writed)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        shape_sources:List[Tuple[int, int]] = [(param_index[array_name], dim) for array_name, dim in translator.shapes_used]
        work_size_dims:List[Tuple[str, int]] = [(param_index[name], dim) for name, dim in translator.work_size_dims]
        # An explicit gpu.work_size(x) declaration: expand each named array into its own
        # dimensions. The rank is only known at call time, so record (param, -1) markers and
        # let _invoke substitute the real per-dimension indices.
        if not work_size_dims and translator.declared_work_arrays:
            work_size_dims = [(param_index[name], -1) for name in translator.declared_work_arrays]
        # local memory: opencl passes byte sizes at launch by parameter position; cuda uses __shared__ declarations and needs nothing passed
        ctype_sizes:Dict[str, int] = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2,
                                      "int": 4, "uint": 4, "long": 8, "ulong": 8,
                                      "float": 4, "double": 8}
        local_sizes:List[int] = [ctype_sizes.get(ctype, 4) * count for _, ctype, count in translator.local_buffers]
        return kernel, shape_sources, work_size_dims, local_sizes, translator.uses_global_id

    def _cache_dir(self, backend:str)->str:
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        base_dir:str = (os.path.dirname(os.path.abspath(source_file)) if source_file else os.getcwd())
        folder:str = {"opencl": "__clcache__", "cuda": "__cucache__", "hip": "__hipcache__"}[backend]
        return os.path.join(base_dir, folder, "pykernels")

    def _write_source(self, sig:str, source:str, backend:str)->str:
        cache_dir:str = self._cache_dir(backend)
        os.makedirs(cache_dir, exist_ok=True)
        extension:str = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[backend]
        path:str = os.path.join(cache_dir, f"pyk_{self.__name__}_{sig}_{self._source_sig}.{extension}")

        existing:Optional[str] = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        if existing != source:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                file.write(source)
        return path

    def _backend_compile(self, source_path:str, c_name:str, backend:str, device_obj:Any):
        if backend == "opencl":
            from ..opencl.oop.program import Program
            program:Any = Program(device_obj.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        if backend == "hip":
            from ..hip.oop.program import Program as HIPProgram
            from ..hip.oop.hip_compiler import HipCompiler
            device_obj.default_context.make_current()
            compiler = HipCompiler(source_path)
            program:Any = HIPProgram(source_path, compiler=compiler)
            return program.kernels[c_name]

        from ..cuda.oop.program import Program as CUProgram
        device_obj.default_context.make_current()
        program:Any = CUProgram(source_path)
        return program.kernels[c_name]

    # ---------- launch ----------

    def _build_launch_args(self, args:List[Any], arg_types:List[Any], shape_sources:List[Tuple[int, int]])->List[Any]:
        launch_args:List[Any] = list(args)
        for param_index, dim in shape_sources:
            launch_args.append(int(args[param_index].shape[dim]))
        return launch_args

    def __repr__(self)->str:
        return f"PyKernel({self.__name__}, backend={self.backend})"


def _dependency_order(device_cache:Dict[Any, Any])->List[Dict[str, Any]]:
    """Order cached device functions so each callee precedes its callers.

    The cache is populated depth-first, so a nested helper (e.g. `clamp01` called by
    `weight`) is completed after the helper that calls it. Emitting in insertion order
    would then reference a function C has not seen yet. Names are matched textually
    against each entry's generated code.
    """
    entries:List[Dict[str, Any]] = [e for e in device_cache.values() if e.get("status") == "done"]
    if len(entries) <= 1:
        return entries

    # name -> entry, for the callee lookup
    by_name:Dict[str, Dict[str, Any]] = {e["c_name"]: e for e in entries}
    ordered:List[Dict[str, Any]] = []
    visiting:set = set()
    done:set = set()

    def visit(entry:Dict[str, Any]) -> None:
        name:str = entry["c_name"]
        if name in done or name in visiting:
            # `visiting` breaks cycles: mutual recursion is rejected elsewhere anyway.
            return
        visiting.add(name)
        for other_name in by_name:
            if other_name != name and other_name in entry["code"]:
                visit(by_name[other_name])
        visiting.discard(name)
        done.add(name)
        ordered.append(entry)
    for entry in entries:
        visit(entry)
    return ordered


def kernel(func:Optional[Callable]=None, *, backend:Optional[str]=None):
    """The `@kernel` decorator.

    When backend is not specified, probe in DEFAULT_BACKEND_ORDER (cuda -> hip -> opencl)
    and take the first backend that enumerates at least one device; the result is cached in-process.

    Usage::

        from pygpgpu.kernel import kernel
        import numpy as np

        @kernel
        def add(a, b, out):
            for i in range(out.shape[0]):
                out[i] = a[i] + b[i]

        add(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) * 2, out)

    - The outermost `for i in range(<array parameter>.shape[0])` loop is the work-item loop;
      the work size is taken automatically from that array's first dimension;
    - plain python functions called by the kernel are translated into device functions (monomorphized per argument types);
    - math.* and abs/min/max/round map to device builtins;
    - unsupported syntax raises SyntaxError.
    """
    def decorator(f:Callable)->PyKernel:
        return PyKernel(f, backend or _default_backend())

    if func is not None:
        return decorator(func)
    return decorator
