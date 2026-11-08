from __future__ import annotations
import asyncio
import ast
import concurrent.futures
import copy
import functools
import inspect
import os
import pathlib
import sys
import threading
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    overload,
)
import numpy as np
from typing_extensions import ParamSpec
from ...vec_types.layout import pack
from ...utils import local_size_for_global, detect_work_size, device_limits, blocks_to_global, _current_loop
from ..python_backend import PythonKernel, analyse
from .. import structs
from ...numpy import ndarray as gnp
from ..translator import (
    PTR,
    DeviceType,
    FunctionTranslator,
    _assigned_name_of,
    _unpack_count_message,
    infer_ctype,
    is_ptr,
    scan_self_attrs,
    short_hash,
    type_sig,
    validate_type_map,
)
from ..objects import (ObjectBinding, _complex_member_value,  # noqa: PLC2701 (shared
                    _is_complex_member_class, convert_object_arguments)
from .. import source_paths
from .binding import _native_argument_errors
from .devices import (
    DEFAULT_BACKEND_ORDER,
    _BACKEND_ALIASES,
    _VENDOR_ALIASES,
    _backend_devices,
    _parse_work_item,
    _default_backend,
    _default_device_lock,
    _default_devices,
    _default_target,
    _device_match_keys,
    _entry_usable,
    _parse_selector,
    _probe_backend,
    _probed_backend,
    _resolve_device,
    available_devices,
    get_default_device,
    get_device,
    list_devices,
    set_default_device,
)
from .type_map import (
    _effective_type_map,
    _type_map_default,
    _type_map_lock,
    get_type_map,
    set_type_map,
)
from .binding import (
    _RECEIVER_PLACEHOLDER,
    _STUB_SCACHE,
    _bind_and_reject,
    _bind_arguments,
    _body_only,
    _bound_arguments_for_call,
    _callable_signature,
    _qualname_for_errors,
    _signature_stub,
)
from .frames import (
    _FRAME_MARKER,
    _body_frame,
    _caller_traceback,
    _is_body_stand_in,
    _is_user_frame,
    _subscript_stand_in,
    _user_frame_syntax_error,
    _user_traceback,
)
from .call_proxies import (
    _BoundKernelMethod,
    _KernelCall,
    _TAIL_RETURN_NAME,
    _TailReturn,
    _TailReturnRewriter,
)
from .cuda_link import (
    _LinkedImage,
    _compile_linked_cuda,
    _cuda_tool,
    _cuda_toolkit,
)
from .images import (
    _check_image_address,
    _image_sampler,
)
from .slots import (
    _SCALAR_SLOT_DTYPES,
    _VEC_RETURN_TYPES,
    _kernel_scalar_argument,
    _outer_scope_value,
    _outer_slot_dtype,
    _return_value,
    _scalar_slot_value,
)
from .shape_check import (
    _check_shape_indices,
)
from .host_regions import (
    _called_modules,
    _created_arguments,
    _declares_a_magic_loop,
    _dependency_order,
    _host_created_values,
    _host_region_values,
    _host_regions,
    _host_scalar_values,
    _is_array_pointer_assignment,
    _positional_argument_names,
    _reads_name,
    _receiver_name,
)
from .factory import (
    kernel,
)
from ._typing import (
    _P,
    _KernelBase,
    _R,
    OuterState,
    ReturnSlots,
    ReturnState,
    WritableOuter,
)
from .pykernel_cache import _PykernelCacheMixin
from .pykernel_invoke import _PykernelInvokeMixin
from .pykernel_call import _PykernelCallMixin
from .pykernel_outer import _PykernelOuterMixin
from .pykernel_returns import _PykernelReturnsMixin
from .pykernel_identity import _PykernelIdentityMixin
from .pykernel_compile import _PykernelCompileMixin


if TYPE_CHECKING:
    # `Image` is imported inside the functions that use it (it is a heavy module and this
    # package keeps backend imports lazy), so an annotation needs the name here instead.
    from ...image import Image


if TYPE_CHECKING:
    from ..translator import ReturnSiteLayout


class PyKernel(_KernelBase[_P], _PykernelCallMixin, _PykernelInvokeMixin,
               _PykernelOuterMixin, _PykernelReturnsMixin, _PykernelIdentityMixin,
               _PykernelCacheMixin, _PykernelCompileMixin):

    def __init__(self, func:Callable[..., Any], backend:str, explicit_backend:bool=False,
                 domain:Optional[str]=None,
                 type_map:Optional[Dict[str, object]]=None):
        if isinstance(func, classmethod):
            raise SyntaxError("@kernel cannot decorate classmethods (cls semantics are host-only)") from None
        if isinstance(func, staticmethod):
            func = func.__func__

        self.raw_func:Callable = func
        self.backend:str = backend
        # True when the decorator named a backend, so set_default_device() must not
        # override it. A backend chosen by probing is only a snapshot, not a binding.
        self._explicit_backend:bool = explicit_backend
        # The parameter whose shape declares the iteration domain, or None. A gid-style body
        # has no magic loop to read a domain from, and naming the array here says so as part
        # of the signature rather than as a statement inside the body that the device never
        # sees.
        self._domain:Optional[str] = domain
        # The device width a python `int`/`float` travels as, resolved to device type names
        # when the decorator ran and **not** read again later: this decorator's mapping is
        # part of what this kernel *is*, so a later set_type_map() must not change kernels
        # that were already declared. Only the two builtins are here; a numpy scalar names
        # its own width and keeps it (see translator.infer_ctype).
        self._type_map:Dict[str, str] = _effective_type_map(type_map)
        self.is_pykernel:bool = True
        self.__name__:str = func.__name__
        self.__doc__:str = func.__doc__ or ""

        arg_names:List[str] = list(func.__code__.co_varnames[:func.__code__.co_argcount])
        self._is_method:bool = bool(arg_names and arg_names[0] == "self")
        self._self_param:str = (arg_names[0] if self._is_method else "")
        # A decorated body may read self.<attr> or call self.<helper>(...), and only the
        # latter can tell a method from a value -- which needs the instance's class, known
        # at call time rather than now. The attribute set is therefore collected in
        # _invoke; the decoration-time body scan this used to cache is gone.

        self._cache:Dict[str, Tuple[object, List[Tuple[int, int]], List[Tuple[int, int]], List[int]]] = {}
        # The python backend executes the body as-is, so there is nothing to specialize or
        # cache: one runner per kernel object serves every call.
        self._python_runner:object = None
        self._source_sig:str = self._source_signature()

    @overload
    def __get__(self, obj:None, objtype:Optional[type]=None)->"PyKernel[_P]": ...
    @overload
    def __get__(self, obj:object, objtype:Optional[type]=None)->_BoundKernelMethod[_P]: ...
    def __get__(self, obj:object, objtype:Optional[type]=None)->object:
        # descriptor protocol: instance access returns the bound form, class access/static methods return self
        if obj is None or not self._is_method:
            return self
        return _BoundKernelMethod(self, obj)


    def __getitem__(self, item:object)->_KernelCall[_P]:
        return _KernelCall(self, None)[item]


    @_native_argument_errors
    # The result is the kernel body's value, produced on the device. `_R` is the
    # decorated function's result type but is not bound on this class, so the
    # honest annotation is the untyped one -- `object` would forbid unpacking a
    # tuple return at the call site.
    def __call__(self, *args:object, **kwargs:object)->Any:
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            call_args:Tuple[object, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None)
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None)


    @_native_argument_errors
    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
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
            call_args:Tuple[object, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None, launch="submit")
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None, launch="submit")


    @_native_argument_errors
    def async_call(self, *args:object, **kwargs:object)->asyncio.Future:
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
            call_args:Tuple[object, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None, launch="async")
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None, launch="async")


    def __repr__(self)->str:
        return f"PyKernel({self.__name__}, backend={self.backend})"


__all__ = [
    "PyKernel",
    "__call__",
    "__get__",
    "__getitem__",
    "__init__",
    "__repr__",
    "_alloc_return_slots",
    "_apply_declared_domain",
    "_backend_compile",
    "_bind_images",
    "_bind_outer_values",
    "_build_launch_args",
    "_c_kernel_name",
    "_cache_dir",
    "_compile",
    "_deliver_error",
    "_deliver_result",
    "_finish_returns",
    "_invoke",
    "_invoke_inner",
    "_invoke_python",
    "_read_outer",
    "_resolve_global",
    "_run_host_region",
    "_source_signature",
    "_store_outer",
    "_sync_device",
    "_type_map_key",
    "_with_returns",
    "_with_tail",
    "_write_back_outer",
    "async_call",
    "submit",
]
