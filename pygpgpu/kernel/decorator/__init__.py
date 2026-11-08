# This module is the facade of a split: the module-level helpers and classes moved
# to the part module named after their subject, and the PyKernel class below keeps
# its methods (they are the next wave). Every published name stays importable from
# here, which is what `from pygpgpu.kernel.decorator import ...` relies on.
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


# ---------- typing hooks ----------

if TYPE_CHECKING:
    # `Image` is imported inside the functions that use it (it is a heavy module and this
    # package keeps backend imports lazy), so an annotation needs the name here instead.
    from ...image import Image

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

if TYPE_CHECKING:
    from ..translator import ReturnSiteLayout

from ._typing import (
    _P,
    _KernelBase,
    _R,
    OuterState,
    ReturnSlots,
    ReturnState,
    WritableOuter,
)



class PyKernel(_KernelBase[_P]):
    """Implementation of the `@kernel` decorator: translates the python function into backend source at call time, then compiles and runs it.

    Decorating class methods is supported: self is dropped during translation, self.xxx attribute
    references in the method body become extra parameters _self_xxx, filled from the instance at call
    time; staticmethod is supported too (either decoration order). A method may also call other
    methods of its own class (self.helper(...)), a static method of any class, or a module function;
    the attributes those helpers read become kernel arguments as well.

    Backend/device/work size are given with brackets at call time (same style as handwritten kernels);
    0, 1 or 2 brackets may be used, in any order::

        k(a, out)                       # default backend (probed, cuda -> hip -> opencl)/device, work size automatic
        k["cuda"](a, out)               # pick the backend
        k["opencl:gfx1036"](a, out)     # backend + device (index or name substring, "name:ordinal" disambiguates)
        k[(64, 8)](a, out)              # global/local (None means automatic)
        k["cuda"][(64, 8)](a, out)      # combined, any order

    The class is generic over the decorated function's own parameters (PEP 612), which is what
    gives an editor the caller's parameter names on a published kernel -- see _KernelBase above.
    """

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

    def _invoke(self, instance:object, args:Tuple[object, ...],
                backend_sel:Optional[str], device_sel:Optional[str],
                global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                launch:str = "call")->object:
        try:
            return self._invoke_inner(instance, args, backend_sel, device_sel,
                                      global_size, local_size, launch)
        except SyntaxError as caught:
            # Trim to the caller's frame and re-raise with that traceback. CPython prepends one
            # frame for the re-raise itself, which is why the exception carries the caller's
            # line rather than this one: the syntax rendering uses the fields on the
            # exception, not the chain.
            caller = _caller_traceback(caught.__traceback__)
            raise caught.with_traceback(caller) from None
        except ValueError as caught:
            # Two different things arrive as a ValueError, and python's own reading decides
            # which treatment each takes.
            #
            # An *argument* refusal -- an integer too wide for the type map (`_check_integer_fits`)
            # -- is about the call, not the body: it carries no source position, and it is
            # trimmed to the caller's frame the way a SyntaxError is. Falling through to the
            # body handler would dress it up as a failure inside the function.
            #
            # One the translator *positioned* (see `_position_error`) is a failure of the
            # body's own line -- an unpacking count that cannot match, a subscript that is
            # too wide -- and reads exactly as the interpreter reads it, so it gets the body
            # treatment below: the caller's line and then the line that failed.
            if not str(caught):
                raise
            if getattr(caught, "lineno", None):
                caught.__traceback__ = _user_traceback(caught, self.raw_func)
                raise caught from None
            caller = _caller_traceback(caught.__traceback__)
            raise caught.with_traceback(caller) from None
        except Exception as caught:
            # A failure of the *body* -- a wrong unpack, an out-of-range index -- has to read
            # exactly as it does without the decorator: the caller's line, then the line in
            # the function that failed. The body's own frame belongs to the execution
            # machinery, and the device backends never run the body at all, so it is
            # replaced by a stand-in carrying the same file, line and function name; the
            # library frames between the two are spliced out here.
            caught.__traceback__ = _user_traceback(caught, self.raw_func)
            raise caught from None

    def _sync_device(self, backend:str, device_obj:object)->None:
        """Wait until every launch this backend has queued has finished, callbacks included.

        The statements between two passes read shared slots on the host; the copies that
        fill them are stream callbacks, so a device-level sync is the one wait that is
        correct regardless of which stream each launch used.
        """
        if backend == "cuda":
            from ...cuda.driver import CUDA
            CUDA.cuCtxSynchronize()
        elif backend == "hip":
            from ...hip.driver import HIP
            HIP.hipDeviceSynchronize()
        elif backend == "metal":
            # Metal's launch waits on its own command buffer, so every launch that
            # returned has already completed; the queue is serial, which makes that
            # wait cover the launches before it too.
            device_obj.default_context.sync()

    def _run_host_region(self, statements:List[ast.stmt],
                         args:Tuple[object, ...], instance:object,
                         carried_slots:Dict[str, object], gap_slots:Dict[str, object],
                         globalns:Optional[Dict[str, object]]=None,
                         localns:Optional[Dict[str, object]]=None,
                         created_objects:Optional[Dict[str, object]]=None,
                         outer_state:Optional[OuterState]=None,
                         writable_outer:Optional[Dict[str, np.ndarray]]=None)->object:
        """Run one region of python statements on the host, and return a tail `return`.

        A region is the statements between two passes, or after the last one. A statement
        reads the launch arguments and whatever earlier statements and slots hold, exactly
        as python would. An array it touches is pulled back to the host first (the previous
        pass wrote it on the device) and marked dirty again afterwards, so the next launch
        pushes the changed values back. ``globalns``/``localns`` may pass a namespace in so
        consecutive regions share one scope, the way python's function body does.
        """
        if not statements:
            return None

        if globalns is None:
            globalns = dict(self.raw_func.__globals__)
        if localns is None:
            localns = {}
            for name, value in zip(_positional_argument_names(self.raw_func), args):
                localns[name] = value
            receiver:str = _receiver_name(self.raw_func)
            if receiver and instance is not None:
                localns[receiver] = instance
        # Closure cells the body reads (a factory returning a @kernel is the common shape)
        # and the outer-scope values the launch carries. A writable outer slot holds the
        # value the device may have updated, and a host write back into it travels to the
        # scope through the same buffer _write_back_outer copies from.
        for name, cell in zip(self.raw_func.__code__.co_freevars,
                              self.raw_func.__closure__ or ()):
            try:
                localns.setdefault(name, cell.cell_contents)
            except ValueError:
                continue                  # an empty cell: python would raise too
        for slot in (outer_state["writable"] if outer_state else []):
            localns[slot["name"]] = slot["buffer"][0]
        for name, buffer in (writable_outer or {}).items():
            localns[name] = buffer[0]
        # The slots the launch shares with the device: a carried scalar or a gap value that
        # an earlier phase updated is the current value, not the initial one. A created
        # array is the object itself -- indexing it would hand the statements its first row.
        for name, slot in list(carried_slots.items()) + list(gap_slots.items()):
            localns[name] = slot[0]
        # A created array is the object itself: unwrapping it would hand the statements its
        # first element (or row), which is not what python's name means.
        for name, value in (created_objects or {}).items():
            localns[name] = value

        # Every `return` in the region is rewritten into a raise before the statements are
        # compiled (see _TailReturnRewriter): a module body cannot hold a `return` at all,
        # and a region's `return` -- including one nested in an `if` -- is how a tail
        # answers a value. The rewriter runs on a copy, because these statements come from
        # the compiled cache and are shared by every call of the kernel.
        localns[_TAIL_RETURN_NAME] = _TailReturn
        rewriter:_TailReturnRewriter = _TailReturnRewriter()
        exec_statements:List[ast.stmt] = [
            rewriter.visit(copy.deepcopy(statement)) for statement in statements]

        touched:List[object] = []
        try:
            for statement, exec_statement in zip(statements, exec_statements):
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and node.id in localns:
                        value:object = localns[node.id]
                        if isinstance(value, gnp):
                            value.to_host()
                            if value not in touched:
                                touched.append(value)
                if isinstance(statement, ast.AugAssign):
                    # `count += 1` in a region updates the slot it names, the same way an
                    # assignment does; the shared helper does not answer for this spelling.
                    target_node:ast.expr = statement.target
                    target:Optional[str] = (target_node.id
                                            if isinstance(target_node, ast.Name) else None)
                else:
                    target:Optional[str] = _assigned_name_of(statement)
                # compile() needs a Module node, and the statement is one entry of the body.
                module:ast.Module = ast.Module(body=[exec_statement], type_ignores=[])
                exec(compile(module, "<kernel gap>", "exec"), globalns, localns)  # noqa: S102 - the kernel source itself
                if target is not None and target in localns:
                    value:object = localns[target]
                    if isinstance(value, gnp) and value not in touched:
                        touched.append(value)
                    elif isinstance(value, np.ndarray) and value.ndim > 0:
                        # A creation the gap statement built: the object itself is what the next
                        # pass receives, exactly as it does for a creation above the loops.
                        (created_objects if created_objects is not None else {})[target] = value
                    elif not isinstance(value, gnp):
                        # `or` cannot pick the slot: bool(np.array([0])) is False, which made a
                        # zero-valued slot read as missing and silently skipped the write-back.
                        slot:Optional[np.ndarray] = (gap_slots.get(target)
                                                     if gap_slots.get(target) is not None
                                                     else carried_slots.get(target))
                        if slot is None:
                            slot = (writable_outer or {}).get(target)
                        if slot is not None:
                            slot[0] = int(value)
                            localns[target] = slot[0]
        except _TailReturn as answered:
            return answered.value
        finally:
            # An array the region touched was pulled back to the host, so the next launch
            # must push it again -- on the `return` path too.
            for value in touched:
                value._set_host_dirty()
        return None

    def _invoke_inner(self, instance:object, args:Tuple[object, ...],
                      backend_sel:Optional[str], device_sel:Optional[str],
                      global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                      launch:str = "call")->object:
        if backend_sel is not None:
            backend:str = backend_sel
        elif getattr(self, "_explicit_backend", False):
            # Decorated as @kernel(backend=...): that backend wins over any default.
            backend = self.backend
        else:
            # No bracket and no explicit backend: consult set_default_device(), read now
            # rather than at decoration time so a later call takes effect.
            backend, default_device = _default_target()
            if device_sel is None:
                device_sel = default_device
        # Record what this call actually used, so `k.backend` describes the last launch
        # rather than the snapshot taken when the decorator ran.
        self.backend = backend
        if backend == "python":
            return self._invoke_python(instance, args, device_sel, global_size, local_size, launch)
        device_obj:object
        device_key:str
        device_obj, device_key = _resolve_device(backend, device_sel)
        device_max_group, device_max_items = device_limits(device_obj)

        self_values:List[object] = []
        self_attrs:List[str] = []
        owner:Optional[type] = None
        self_struct_members:Dict[str, str] = {}
        self_struct_type:object = None
        # The host regions -- the statements above the first magic loop, between two of
        # them, and after the last one -- run **once per call**, here, before anything else
        # looks at the body. Their values are this call's (a statement may read the caller's
        # arguments), and the lines tell the translator which statements are host work, so
        # it emits no device code for them (see _host_region_values).
        host_executed:List[int] = []
        host_values_now:Dict[str, object] = {}
        if not self._is_method or instance is not None:
            built:Dict[str, object] = _host_created_values(
                self.raw_func, tuple(args), instance)[0]
            host_values_now, host_executed = _host_region_values(
                self.raw_func, tuple(args), instance, set(built))
            host_values_now = {**built, **host_values_now}
        self.host_region_snapshot = host_values_now
        self.host_region_statements = host_executed
        if self._is_method:
            if instance is None:
                raise TypeError(f"{self.__name__}() requires an instance")
            # The decorated body was scanned at decoration time, before the class
            # existed, so self.<helper> could only look like data. Scan again now
            # that the instance is here: calling a self-method makes the attributes
            # *it* reads kernel arguments too, recursively.
            owner = type(instance)
            # Which statements the host regions run is a fact about the *source*, so the lines
            # computed above are what decides which methods are host-only (see `_host_region_values`).
            self_attrs = scan_self_attrs(self.raw_func, self._self_param, owner,
                                         set(host_executed))
            for name in self_attrs:
                if not hasattr(instance, name):
                    raise SyntaxError(
                        f"instance of {owner.__name__} has no attribute '{name}' "
                        f"required by kernel {self.__name__}") from None
                value:object = getattr(instance, name)
                if callable(value) and not isinstance(value, np.ndarray):
                    raise SyntaxError(
                        f"self.{name} is a function, not a value: a function cannot be passed "
                        f"to a device (call it as self.{name}(...) instead)") from None
                self_values.append(value)
            # The scalar members travel as ONE structure the body reads through (the way a
            # device helper already takes `Scene self`); an ndarray member stays its own
            # argument, because the read/write analysis and the shape plumbing need it
            # separately. The record is one element long so `self.<attr> = ...` writes
            # through it, and the struct pointer is read-only when the body never assigns
            # a member.
            struct_attrs:List[str] = [name for name in self_attrs
                                      if not isinstance(getattr(instance, name), np.ndarray)]
            array_attrs:List[str] = [name for name in self_attrs
                                     if isinstance(getattr(instance, name), np.ndarray)]
            if struct_attrs:
                from ..objects import _structure_for, member_type  # noqa: PLC0415 (import cycle)

                self_struct:type = _structure_for(
                    [(name, member_type(getattr(instance, name), name, self._type_map))
                     for name in struct_attrs], [])
                from ..structs import register_name_hint  # noqa: PLC0415 (import cycle)

                register_name_hint(np.dtype(self_struct), owner)
                # Each attribute is a member of the same name: the mapping is the identity,
                # never a zip against the dtype's names. numpy sorts a record's field names
                # alphabetically, so zipping them against the attribute order paired
                # `image_height` with `du_in_world` and every member read the wrong field.
                self_struct_members = {name: name for name in struct_attrs}
                from .. import structs as _structs  # noqa: PLC0415 (import cycle)

                self_struct_type = (PTR, _structs.register(np.dtype(self_struct)))
                # The value that travels is the ctypes structure itself: the launch packs
                # its bytes through the parsed layout, which handles vector members (whose
                # numpy record view uses a sub-array form that would misread the bytes).
                # A body that assigns `self.<attr> = ...` writes through the device buffer,
                # and the host copies the members home afterwards.
                holder:object = self_struct()
                for name in struct_attrs:
                    member_value:object = getattr(instance, name)
                    field_type:object = dict(self_struct._fields_).get(name)
                    if _is_complex_member_class(field_type) \
                            and not isinstance(member_value, field_type):
                        member_value = _complex_member_value(member_value, field_type)
                    setattr(holder, name, member_value)
                self_attrs = array_attrs
                self_types = [self_struct_type] + [
                    infer_ctype(getattr(instance, name), self._type_map)
                    for name in array_attrs]
                self_values = [holder] + [
                    getattr(instance, name) for name in array_attrs]

        # An arbitrary python object becomes a synthesized structure before its type is
        # inferred: the structure *is* the argument type, and every step after this sees the
        # same `ctypes.Structure` a caller would have declared by hand. The binding records
        # the pairing, so member writes travel back to the caller's object after the launch.
        # The receiver travels along too: a body that builds its own value (`ray = Ray(...)`)
        # types the construction's arguments from what the receiver's members are, and the
        # receiver's instance is the only place that answer lives.
        object_binding:ObjectBinding = ObjectBinding()
        args = convert_object_arguments(self.raw_func, args, self._resolve_global,
                                        self._self_param, object_binding,
                                        self_value=instance,
                                        type_map=self._type_map)
        arg_types:List[DeviceType] = [infer_ctype(a, self._type_map) for a in args]
        if not self_struct_members:
            self_types = []
            for name, value in zip(self_attrs, self_values):
                try:
                    self_types.append(infer_ctype(value, self._type_map))
                except (SyntaxError, ValueError) as e:
                    # `ValueError` is the width refusal (a python int too wide for the
                    # mapping); both lose their own context here, so each is re-raised with
                    # the attribute that carries it.
                    raise type(e)(f"self.{name}: {e}") from None

        sig:str = type_sig(arg_types + self_types)
        # The mapping is part of the cache key, not only part of the type signature: two
        # kernels over the same argument values under different mappings produce `int` and
        # `long` parameters, and only one `_cache` entry may answer for a launch. The types
        # already differ (`sig` carries them), so this is belt-and-braces for the case where
        # a mapping is changed to a width whose type name another entry already used.
        cache_key:str = f"{backend}|{device_key}|{sig}|{self._type_map_key()}"
        entry = self._cache.get(cache_key)
        if entry is None:
            # Create the device objects for any Image before compiling. On the driver
            # this was developed against, loading a module that contains texture or
            # surface calls makes every later texture/surface creation fail, so an
            # image bound after compilation would never get its device storage.
            self._bind_images(full_values=args, backend=backend)
            entry = self._compile(args, arg_types, self_types, self_attrs, owner, sig,
                                  backend, device_obj, instance,
                      self_struct_members=self_struct_members,
                      self_struct_type=self_struct_type)
            self._cache[cache_key] = entry
        (kernel_obj, shape_sources, work_size_dims, local_sizes, uses_global_id,
         image_samplers, returns, outer_vars, launch_meta) = entry
        # One entry per magic loop, launched in order; a single-loop body has exactly one.
        active_kernels:List[object] = list(launch_meta.get("kernels") or [kernel_obj])
        # The extent arguments each phase kernel declares, in phase order.
        kernel_shapes:List[List[Tuple[int, int]]] = list(
            launch_meta.get("kernel_shapes") or [[] for _ in active_kernels])
        kernel_created:List[List[str]] = list(
            launch_meta.get("kernel_created") or [[] for _ in active_kernels])
        # Each phase's parameters in order, so the launch can align values to the signature
        # instead of assuming the shared prefix: a phase may declare a carried name the
        # first kernel does not, or skip one it does.
        phase_param_names:List[List[str]] = list(
            launch_meta.get("phase_param_names") or [[] for _ in active_kernels])
        # Whether each phase kernel declares the return claim/site words. A pass whose only
        # return runs in its host tail never claims, so its signature omits them and its
        # argument list has to omit them too (empty for a single-phase body, whose one
        # kernel is the main translator's own).
        phase_claims:List[bool] = list(launch_meta.get("phase_claims") or [])

        outer_writable:List[Dict[str, object]] = []
        carried_slots:Dict[str, object] = {}
        # The names `full` holds, in order: the caller's arguments, then the carried slots.
        # `build_phase_args` aligns each phase's signature against these names.
        # The `_self_<attr>` parameters sit in `full` right after the caller's arguments,
        # so the prefix names must include them or the aligned loop would look up an
        # attribute value in the wrong place.
        # When the receiver travels as a structure, it is the first entry of self_values
        # and the first device parameter: name it so the aligned loop finds the holder.
        receiver_name:List[str] = ([self._self_param]
                                   if self_struct_members else [])
        self_attr_names:List[str] = [f"_self_{name}" for name in self_attrs]
        # The launch is packed in the order the device signature declares: the receiver
        # first (it is a method's first parameter), then the caller's arguments, then the
        # ndarray attributes. `arg_types` and `self_types` already lead with the receiver
        # type, so the values have to lead with the receiver too.
        receiver_values:List[object] = (list(self_values[:1]) if self_struct_members else [])
        argument_values:List[object] = (list(self_values[1:]) if self_struct_members
                                     else list(self_values))
        prefix_names:List[str] = (receiver_name
                                  + list(_positional_argument_names(self.raw_func))[:len(args)]
                                  + self_attr_names
                                  + list(launch_meta.get("carried") or []))
        full:List[object] = [_kernel_scalar_argument(value, t)
                          for value, t in zip(receiver_values + list(args) + argument_values,
                                              arg_types + self_types)]
        # A carried scalar is one of the kernel's own parameters (declared before the shape
        # extents), so its slot joins full here rather than the trailing created values.
        # The slot is the carried value's own width (`total = 0.0` is a float): the
        # device parameter is a `float*` there, and an int32 host buffer under it
        # truncated every contribution before the kernel even ran.
        carried_types:Dict[str, object] = launch_meta.get("carried_types") or {}
        for name in launch_meta.get("carried") or []:
            slot_t:object = carried_types.get(name, "int")
            dtype:np.dtype = np.dtype(np.float32 if slot_t == "float" else np.int32)
            slot = np.zeros(1, dtype=dtype)
            slot[0] = dtype.type(_host_scalar_values(
                self.raw_func, [name], tuple(args), instance).get(name, 0))
            full.append(slot)
            carried_slots[name] = slot
        own_count:int = len(full)

        # Outer-scope values are read before the launch arguments are built: a shape
        # source can name an outer array, and `full` is indexed by the device parameter
        # order (see _build_launch_args), so the outer values have to be part of it.
        outer_state:OuterState = self._bind_outer_values(outer_vars, backend)
        # Objects the kernel body creates are built here, on every call, exactly as the
        # interpreter would at that point in the body. They travel as trailing parameters.
        # A method's creation may read `self` (`np.zeros((self.height, ...))`), so the
        # instance is bound too.
        created:Dict[str, object] = _host_created_values(
            self.raw_func, tuple(args), instance,
            required_created=list(launch_meta.get("kernel_created_all") or []))[0]
        # A statement between two magic loops runs on the host, so its value is computed here
        # and travels to the pass that reads it. A scalar travels as a one-element slot,
        # exactly like a carried scalar; a creation the pass declares travels as the object
        # itself, exactly like any other created array.
        # One shared namespace per gap region: a later region reads what an earlier one
        # assigned, the way consecutive statements of a function body do.
        gap_namespaces:List[Tuple[Dict[str, object], Dict[str, object]]] = []
        for _ in (launch_meta.get("phase_gaps") or []):
            region_globals:Dict[str, object] = dict(self.raw_func.__globals__)
            region_locals:Dict[str, object] = {}
            for name, value in zip(_positional_argument_names(self.raw_func), args):
                region_locals[name] = value
            receiver:str = _receiver_name(self.raw_func)
            if receiver and instance is not None:
                region_locals[receiver] = instance
            gap_namespaces.append((region_globals, region_locals))
        gap_names:List[str] = list(launch_meta.get("gap_names") or [])
        created_all:List[str] = list(launch_meta.get("kernel_created_all") or [])
        # A gap value the interpreter bound between the passes travels as a host scalar of
        # its own type (see _bind_host_statement), not through the carried `int` slot below:
        # `shift = len(label)` holds a number, but the statement it came from is host code
        # and its value may be anything a python name can hold.
        host_scalar_names:Set[str] = set(launch_meta.get("host_values") or [])
        gap_slots:Dict[str, object] = {}
        for name, value in _host_scalar_values(
                self.raw_func, [n for n in gap_names
                                if n not in created_all and n not in host_scalar_names],
                tuple(args), instance).items():
            slot = np.zeros(1, dtype=np.int32)
            try:
                slot[0] = int(value)
            except (TypeError, ValueError):
                # Not a number: it travels as a host scalar of its own type instead (see
                # _bind_host_statement), and this slot is not its carrier.
                continue
            gap_slots[name] = slot
        # The created objects the device signature declares: the aligned prefix picks the
        # real array from here rather than a scalar slot.
        created_objects:Dict[str, object] = {name: created[name] for name in created_all
                                          if name in created}

        outer_writable = list(outer_state["writable"])
        # The plain values host-region statements bound, in device-parameter order: the
        # translator appended them after the created objects (see _bind_host_statement), so
        # the launch supplies them in that same order.
        host_value_names:List[str] = [name for name in launch_meta.get("host_values") or []
                                      if name not in created_all]
        # Their actual values, recomputed on *this* call: the compile that recorded the
        # signature happens once, but a host-region value can depend on the arguments
        # (`shift = len(label)`), so the launch reads the snapshot the call just made.
        host_region_values:Dict[str, object] = dict(
            getattr(self, "host_region_snapshot", None)
            or launch_meta.get("host_region_values") or {})
        runtime_values:List[object] = list(outer_state["full"]) + _created_arguments(created)
        return_state:Dict[str, object] = (self._alloc_return_slots(returns)
                                      if returns else {})
        # Whether this launch's first kernel declares the return claim/site words. A
        # single-phase body has one kernel -- the main translator's -- and this is its own
        # answer; with several phases each entry of `phase_claims` decides instead. The
        # value is recorded at compile time because the translator is not reachable here.
        main_claims:bool = bool(launch_meta.get("main_claims", True))
        # Launch order must match the device signature: own arguments, shape extents, then
        # the values the host supplies (outer scope, then created objects). The shape
        # extents are read from `full` by device-parameter index, so the runtime values are
        # part of `full` but are appended to the launch afterwards.
        full.extend(runtime_values)
        # `full` indexes like the device parameter list, which is what shape_sources were
        # recorded against. Validate the dimensions first: an index past an array's rank
        # used to raise a bare `IndexError: tuple index out of range`, which says nothing
        # about which array or loop was wrong.
        _check_shape_indices(self.__name__, full, shape_sources, work_size_dims, launch_meta)

        def build_phase_args(extents:List[Tuple[int, int]],
                             extra_created:Optional[List[str]],
                             phase_index:int=0,
                             claims:Optional[bool]=None)->List[object]:
            """The launch arguments for one phase kernel.

            The shared prefix is the device parameter list up to `own_count`; the extents
            follow it because the translator appends them to each kernel's signature. The
            created objects differ per phase -- `mid = np.zeros(4)` between two loops is a
            parameter of the pass that reads it and of no other -- so `extra_created` names
            the ones this kernel expects.
            """
            # The prefix `full[:own_count]` holds the caller's arguments and the carried
            # slots by name; a phase's own signature may name a different subset (a gap
            # value only one pass reads, or a carried scalar this pass writes). Align the
            # values to the signature rather than assuming the prefix matches it.
            prefix_values:List[object] = list(full[:own_count])
            by_name:Dict[str, object] = dict(zip(prefix_names, prefix_values))
            # Only the names declared *before* the shape extents participate in the aligned
            # prefix: outer-scope values and host-created objects are declared after them and
            # the blocks below append those, so including them here would shift the extents.
            declared:List[str] = [name for name in (
                phase_param_names[phase_index]
                if phase_index < len(phase_param_names) else [])
                if name not in outer_vars and name not in created
                and name not in host_value_names]
            if declared:
                # Align the values to this phase's own signature: a gap value only one pass
                # reads, or a carried scalar this pass writes, is not in the shared prefix.
                # Outer-scope names are excluded here -- the signature declares them last,
                # and the block below appends them in the host's own order.
                phase_args:List[object] = [by_name.get(name, gap_slots.get(name, 0))
                                        for name in declared]
            else:
                # No per-phase metadata (a gid-style body): the shared prefix *is* the
                # parameter list, in order.
                phase_args:List[object] = prefix_values
            for param_index, dim in extents:
                phase_args.append(int(full[param_index].shape[dim]))
            # The signature declares the sampler and local-memory arguments right after the
            # shape extents, and the outer/created values last (see render_kernel), so they
            # are appended in that order here. Getting this wrong shifts every later
            # argument: the local pointer would land where an outer scalar is expected.
            if backend == "opencl":
                for param_index in image_samplers:
                    phase_args.append(_image_sampler(full[param_index]))
                for local_bytes in local_sizes:
                    phase_args.append(local_bytes)
            phase_args.extend(outer_state["full"])
            for name, value in created.items():
                # A created name this phase's own signature already declares (a gap creation
                # reads like `middle = np.zeros(4)` and the pass that uses it declares it)
                # has been passed in the aligned prefix above; adding it again here would
                # shift every later argument.
                if (extra_created is None or name in extra_created) and \
                        name not in declared:
                    phase_args.append(_kernel_scalar_argument(value, infer_ctype(value)))
            # The plain values a *host-region statement* bound (a `w = len(label)` above the
            # loop, or an arbitrary-precision number the device receives as a double). They
            # are declared with the created objects, so they are appended with them -- and
            # skipped when this phase's signature already carries them in the prefix.
            for name in host_value_names:
                if name in created or name in outer_vars or name in declared:
                    continue
                if name not in host_region_values:
                    continue
                value = host_region_values[name]
                phase_args.append(_kernel_scalar_argument(value, infer_ctype(value)))
            if return_state:
                phase_args.extend(return_state["buffers"])
                # A pass that never claims -- one whose only return runs in its host tail --
                # does not declare the words, so it must not receive them either. `claims`
                # is None only for the single-phase fast path, whose one kernel is the main
                # translator's: its own signature is what `main_claims` records.
                if claims if claims is not None else main_claims:
                    phase_args.append(return_state["claim"])
                    phase_args.append(return_state["site"])
            return phase_args

        # The read-back of a previous launch is a stream callback, and the caller may have
        # written the host arrays since. The upload below reads those arrays' memory, so it
        # must see the callbacks' result: wait once, before anything is uploaded.
        self._sync_device(backend, device_obj)
        launch_args:List[object] = build_phase_args(shape_sources, None)
        # Vector values travel in the layout the backend's ABI defines: OpenCL pads a
        # 3-component vector to four slots, CUDA/HIP take the host's three. The host keeps
        # the three-component form everywhere, so each argument is converted here and its
        # result copied back once the launch has finished.
        # Every entry pairs the value handed to the launch with the callable that copies it
        # back into the caller's own form once the launch has finished.
        unpackers:List[Tuple[object, Callable[[object], None]]] = []
        for index, argument in enumerate(launch_args):
            packed_value, unpack = pack(argument, backend)
            if packed_value is not argument:
                launch_args[index] = packed_value
            if unpack is not None:
                unpackers.append((packed_value, unpack))
        # The extents the magic loop itself iterates, in launch order (fastest axis first).
        # Derived whenever the loop states a domain, because an explicit bracket has to agree
        # with it axis by axis, not merely dimension by dimension.
        loop_extents:List[int] = []
        if work_size_dims:
            # A dim of -1 is an `@kernel(domain=...)` declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            member_index:int = 0
            for pi, dim in work_size_dims:
                if pi is None:
                    loop_extents.append(dim)
                    continue
                if dim == -2:
                    # A host-side integer supplied as an argument, used as an extent:
                    # `np.ndindex(self.image_height, self.image_width)`.
                    loop_extents.append(int(full[pi]))
                    continue
                if dim == -3:
                    # The extent is a receiver-struct member: full[pi] is the ctypes
                    # structure the caller handed over, and the member holds the count.
                    member_name = launch_meta.get("work_size_members", [])[member_index]
                    member_index += 1
                    loop_extents.append(int(getattr(full[pi], member_name)))
                    continue
                shape:Tuple[int, ...] = tuple(int(s) for s in full[pi].shape)
                if dim >= 0:
                    if dim >= len(shape):
                        # The loop asks for an axis the array does not have. numpy raises
                        # this as an unpacking error from the `for` itself, so the launch
                        # says it in the same words instead of letting an index slip out.
                        raise ValueError(
                            f"{_unpack_count_message(dim + 1, len(shape))}: "
                            f"'{launch_meta['param_names'].get(pi, 'the array')}' has shape "
                            f"{shape}")
                    loop_extents.append(shape[dim])
                else:
                    loop_extents.extend(shape)
        if global_size is None and work_size_dims:
            # A dim of -1 is an `@kernel(domain=...)` declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            extents_list:List[int] = list(loop_extents)
            # A domain of more axes than a device grid has (three) is folded into one flat
            # axis of `prod(shape)` work items, which is what the original auto-detection
            # did. Two consequences follow and both are honest: a magic loop recovers the
            # extra axes by decomposing that position over its own domain (`_emit_nd_head`),
            # while a gid-style body reads the flat position directly -- `get_global_id(0)`
            # is that index and the higher axes read 0, so a body that wants the original
            # axes spelled out should iterate with `np.ndindex` instead.
            extents:Tuple[int, ...] = tuple(extents_list)
            if len(extents) > 3:
                total:int = 1
                for extent in extents:
                    total *= int(extent)
                extents = (total,)
            # `np.ndindex(arr.shape)` means the array's whole shape, but the translator only
            # sees a pointer and cannot know its rank, so it takes one dimension per loop
            # name. `for j, i in np.ndindex(image.shape)` over an RGB image would therefore
            # iterate two of the three axes and never vary the channel. Check it here, where
            # the rank is known.
            whole:int = int(launch_meta.get("whole_shape_index", -1))
            loop_names:int = int(launch_meta.get("loop_names", 0))
            if whole >= 0 and loop_names:
                rank:int = len(tuple(int(s) for s in full[whole].shape))
                if rank != loop_names:
                    # numpy raises the unpacking error itself for this shape, so the device
                    # reports it in the same words: a body should not have to learn a second
                    # vocabulary for the same mistake. The array and its shape follow, since
                    # only the launch knows either.
                    raise ValueError(
                        f"{_unpack_count_message(loop_names, rank)}: "
                        f"'{launch_meta['param_names'].get(whole, 'the array')}' has shape "
                        f"{tuple(int(s) for s in full[whole].shape)}, and "
                        f"np.ndindex(....shape) iterates every dimension of it")
            # The loop has to name one index per dimension of its domain: `for j, i in
            # np.ndindex(image.shape)` over an RGB image would silently iterate only two of
            # the three dimensions, so the third axis never varies. Say so instead.
            if loop_names and loop_names != len(extents_list):
                source = ""
                if work_size_dims[0][0] is not None:
                    source = (f" (the array has shape "
                              f"{tuple(int(s) for s in full[work_size_dims[0][0]].shape)})")
                raise ValueError(
                    f"{_unpack_count_message(loop_names, len(extents_list))}: "
                    f"the loop's iteration domain has "
                    f"{len(extents_list)} dimension{'s' if len(extents_list) != 1 else ''}"
                    f"{source}. Give one name per dimension, or iterate an explicit subset, "
                    f"e.g. np.ndindex(image.shape[0], image.shape[1])")
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
                f"(`for i in range(x.shape[0]): ...`), name the array in the decorator "
                f"(`@kernel(domain=\"x\")`), or pass a bracket: "
                f"{self.__name__}[(total, local)](...)")
        if global_size is not None:
            # Whether the bracket agrees with the loop is a question about *coverage*, not
            # about rank: the kernel folds the launch's ids into one flat position and
            # decomposes it over the loop's own domain, so any launch shape works as long as
            # it supplies enough work items (see FunctionTranslator._emit_nd_head). A launch
            # enumerating fewer work items than the loop traverses stops short of part of the
            # domain and silently leaves it untouched, which is what this rejects. A larger
            # one is merely wasteful: the per-axis bounds guard skips the excess.
            #
            # The check only applies to a *magic* loop (`loop_names >= 1`): a gid-style body
            # states its domain with the bracket alone, so there is nothing to compare it with.
            loop_names:int = int(launch_meta.get("loop_names", 0))
            if loop_names and loop_extents:
                needed:int = 1
                for extent in loop_extents:
                    needed *= int(extent)
                supplied:int = 1
                for size in global_size:
                    supplied *= int(size)
                if supplied < needed:
                    raise ValueError(
                        f"{self.__name__}(): the work size bracket {tuple(global_size)} "
                        f"launches {supplied} work items, but its magic loop traverses "
                        f"{tuple(loop_extents)} ({needed} of them), so the rest of the domain "
                        f"would never be visited. Any launch shape is accepted as long as it "
                        f"covers {needed} work items (the loop decomposes a flat work-item "
                        f"position, so even {needed} alone works); a larger grid is fine, and "
                        f"omitting the bracket lets the loop derive the size itself")
            # The bracket arrived as `(blocks, threads)` and was turned into the total
            # work-item form here (see _parse_work_item), so the two axes agree in rank and
            # the total always divides by the group: the quotient *is* the block count. A
            # caller's own `local=` still has to be a size that splits the total, which is
            # the one divisibility question that remains -- and it is answered for that
            # argument, not for the bracket.
            if local_size is None:
                local_size = local_size_for_global(
                    global_size, max_work_group_size=device_max_group,
                    max_work_item_sizes=device_max_items)
            if len(local_size) != len(global_size):
                raise ValueError(
                    f"the block size {local_size} names {len(local_size)} "
                    f"dimension(s) but the launch has {len(global_size)}; both have to name "
                    f"the same number of axes")
            # Every backend's `Kernel.__getitem__` takes the same bracket the caller wrote
            # (`blocks, threads`), so what travels here is the block count and the group
            # width -- never the total. Handing a backend the total would make it multiply it
            # by the group width again: `(1, 256)` became 65536 work items that way. CUDA and
            # HIP take `(grid_dims, block_dims)` directly; OpenCL converts the pair to the
            # `(global_work_size, local_work_size)` its API wants (see utils.blocks_to_global).
            blocks = tuple(g // l for g, l in zip(global_size, local_size))
            phase_geometry = (blocks, local_size)
        else:
            phase_geometry = None
        # Every phase of the body gets the same geometry: they are passes over one domain.
        # Each is launched in turn on the stream this call owns, so the arrays that travel
        # between them stay on the device instead of going back to the host and returning.
        # The asynchronous forms launch the first kernel only -- a multi-pass body is a
        # sequence of dependent launches, and a future per pass is an interface this
        # launch does not have -- so it is refused by name rather than silently running
        # one pass of several. A single-phase body (the common case) launches whole.
        if launch in ("submit", "async"):
            if len(active_kernels) > 1:
                raise ValueError(
                    f"{self.__name__}(): submit()/async_call() cannot run a body of "
                    f"{len(active_kernels)} passes -- each pass is its own launch and "
                    f"they depend on each other. Call the kernel synchronously, or "
                    f"split the body")
            bound_async = (active_kernels[0][phase_geometry]
                           if phase_geometry is not None else active_kernels[0])
            future:object = (bound_async.submit(*launch_args) if launch == "submit"
                             else bound_async.async_call(*launch_args))
            # The statements after the loop are the host tail: a body whose answer is
            # `return <expression>` after the loop has no claim word at all -- the
            # translator withheld it because no work item ever claims -- so the value
            # only exists once the tail has run. The synchronous path runs it inline
            # below; the asynchronous path runs it when the future settles, exactly
            # where its read-backs have landed. A body with no tail and nothing to
            # write back settles with no extra work.
            tail_statements:List[ast.stmt] = list(
                launch_meta.get("body_tail") or launch_meta.get("phase_tail") or [])
            if not tail_statements and not outer_state["writable"] and not return_state:
                return future
            return self._with_tail(
                future, tail_statements, args, instance,
                carried_slots, gap_slots, created_objects, outer_state,
                outer_writable, return_state, returns, full, created)
        for position, phase_kernel in enumerate(active_kernels):
            if position:
                # The previous pass wrote the shared slots on the device; the read-back is
                # a stream callback that may not have run yet. The gap statements read those
                # slots on the host, so they must wait for the copies to land first.
                self._sync_device(backend, device_obj)
                regions:List[List[ast.stmt]] = list(launch_meta.get("phase_gaps") or [])
                if position - 1 < len(regions):
                    region_writable:Dict[str, np.ndarray] = {
                        entry["name"]: entry["buffer"] for entry in outer_writable}
                    self._run_host_region(regions[position - 1], args, instance,
                                          carried_slots, gap_slots,
                                          gap_namespaces[position - 1][0],
                                          gap_namespaces[position - 1][1],
                                          created_objects, outer_state,
                                          region_writable)
            # A phase kernel declares only the shapes *it* indexes, so its argument list is
            # rebuilt from the same sources with its own extents -- a later phase commonly
            # indexes an array the first one never named.
            phase_shapes:List[Tuple[int, int]] = (
                kernel_shapes[position] if position < len(kernel_shapes) else [])
            bound_kernel = (phase_kernel[phase_geometry] if phase_geometry is not None
                            else phase_kernel)
            phase_created_names:List[str] = (
                kernel_created[position] if position < len(kernel_created) else [])
            # A phase is described by its extents *and* by the objects it created: a pass
            # that reads an array built between the loops declares it, one that does not
            # must not receive it.
            # The fast path is only safe when the phase declares exactly the parameters the
            # first kernel does: a gap value (`factor = 10` between two loops) is one pass's
            # own parameter, and reusing the first kernel's argument list there would leave
            # every later argument one slot short.
            phase_names:List[str] = (phase_param_names[position]
                                     if position < len(phase_param_names) else [])
            first_names:List[str] = (phase_param_names[0]
                                     if phase_param_names else [])
            if (phase_shapes == shape_sources
                    and phase_created_names == list(created)
                    and phase_names == first_names):
                bound_kernel(*launch_args)
            else:
                bound_kernel(*build_phase_args(phase_shapes, phase_created_names, position,
                                               phase_claims[position]
                                               if position < len(phase_claims) else True))
        # The statements after the last loop are python: they run once, on the host, now that
        # every work item has finished -- the same moment python would run them.
        tail_writable:Dict[str, np.ndarray] = {
            entry["name"]: entry["buffer"] for entry in outer_state["writable"]} \
            if outer_state else {}
        tail_result:object = self._run_host_region(
            list(launch_meta.get("body_tail") or launch_meta.get("phase_tail") or []),
            args, instance,
            carried_slots, gap_slots, created_objects=created_objects,
            outer_state=outer_state, writable_outer=tail_writable)
        for packed_value, unpack in unpackers:
            unpack(packed_value)
        if return_state and int(return_state["claim"][0]):
            # A work item reached an in-loop return first: python's "first return wins".
            result:object = self._finish_returns(return_state, returns, full, created)
        elif tail_result is not None:
            result:object = tail_result
        else:
            result:object = (self._finish_returns(return_state, returns, full, created)
                          if return_state else None)
        self._write_back_outer(outer_state)
        if object_binding.written:
            object_binding.write_back()
        return result

    # ---------- outer-scope variables ----------

    def _bind_outer_values(self, outer_vars:Dict[str, object],
                           backend:str)->OuterState:
        """The launch arguments carrying this call's outer-scope values.

        A read-only outer variable is passed by value, refreshed on every call, so a
        changed scope value is seen immediately -- that is the whole point of
        parameterizing it instead of inlining it. A declared `nonlocal`/`global` scalar
        travels as a one-element buffer, because the device writes through the pointer;
        the buffer is copied back into the scope once the launch has completed (see
        _write_back_outer and _with_returns). An outer array is passed as itself, and the
        backend's read/write analysis reads it back when the kernel wrote it.
        """
        args:List[object] = []
        full:List[object] = []
        writable:List[Dict[str, object]] = []
        for name, outer in outer_vars.items():
            kind:str = outer["kind"]
            value:object = self._read_outer(name, kind, outer.get("scope"))
            if is_ptr(outer["type"]):
                if value is None:
                    raise ValueError(
                        f"outer array '{name}' is undefined at launch time")
                args.append(value)
                full.append(value)
                continue
            if not outer["writable"]:
                converted:object = _kernel_scalar_argument(value, outer["type"])
                args.append(converted)
                full.append(converted)
                continue
            buffer:np.ndarray = np.zeros(1, dtype=_outer_slot_dtype(outer["type"]))
            buffer[0] = _scalar_slot_value(value, outer["type"])
            args.append(buffer)
            full.append(buffer)
            writable.append({"name": name, "kind": kind, "buffer": buffer,
                             "type": outer["type"]})
        return {"args": args, "writable": writable, "full": full}

    def _read_outer(self, name:str, kind:str,
                    scope:Optional[Dict[str, object]] = None)->object:
        """The current value of an outer name, read the way python would resolve it.

        ``scope`` is the namespace the translator resolved the name in, which is the
        kernel's module for a name the body reads itself and the *helper's* module for one
        a device helper reads. Both are read live so a changed value is seen on the next
        call, which is why the namespace is carried rather than the value.
        """
        if kind == "closure":
            cells:Dict[str, object] = {}
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                cells[cell_name] = cell
            cell = cells.get(name)
            if cell is None:
                raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
            return cell.cell_contents
        if scope is not None:
            return scope.get(name, None)
        return self.raw_func.__globals__.get(name, None)

    def _write_back_outer(self, outer_state:OuterState)->None:
        """Copy a declared outer scalar's final value back into its scope."""
        for entry in outer_state["writable"]:
            value:object = _outer_scope_value(entry["buffer"][0], entry["type"])
            self._store_outer(entry["name"], entry["kind"], value)

    def _store_outer(self, name:str, kind:str, value:object)->None:
        """Assign an outer name, through its closure cell or the module globals."""
        if kind == "closure":
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                if cell_name == name:
                    cell.cell_contents = value
                    return
            raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
        self.raw_func.__globals__[name] = value

    def _alloc_return_slots(self, returns:ReturnSlots)->ReturnState:
        """One device buffer per value slot, plus the shared claim and site words.

        Every buffer is zero-initialised: the claim word then starts unclaimed, so a
        launch where no work item returned reads back as None rather than as garbage.
        """
        buffers:List[np.ndarray] = [np.zeros(1, dtype=dtype) for dtype in returns["slot_types"]]
        return {"buffers": buffers,
                "claim": np.zeros(1, dtype=np.int32),
                "site": np.zeros(1, dtype=np.int32)}

    def _finish_returns(self, return_state:ReturnState,
                        returns:ReturnSlots, full:List[object],
                        created:Optional[Dict[str, object]]=None)->object:
        """Rebuild the python result of a finished launch.

        The winning return site decides the shape: `return a, b` gives a tuple,
        `return value` gives the bare value, and a bare `return` gives None. A launch in
        which no work item reached any `return` gives None too, exactly as a python
        function does when no returning branch was taken.

        An array position resolves to the host's own object: a parameter comes from the
        launch arguments, and an object the body created comes from `created`. Either way
        the caller receives the same object, never a copy.
        """
        if not int(return_state["claim"][0]):
            return None
        site:int = int(return_state["site"][0]) - 1
        layout:ReturnSiteLayout = returns["sites"][site]
        if not layout["positions"]:
            # The winning site was a bare `return` (or `return None`).
            return None
        values:List[object] = []
        for entry in layout["positions"]:
            if entry["kind"] == "array":
                name:str = entry["name"]
                if created is not None and name in created:
                    values.append(created[name])
                else:
                    values.append(full[returns["arg_indices"][name]])
            elif entry.get("reused"):
                # The value already lives in the argument the body returned, so the caller
                # gets it back from there -- no slot was allocated for it.
                # The parameter travels as a one-element buffer, so the value it holds is
                # what the caller passed -- and what the body may have updated. A numpy
                # record (`np.void`) is a parameter too, but indexing a record reads its
                # first *field*, so only a real 0-d/1-element array is unwrapped.
                shared_slot:object = full[returns["arg_indices"][entry["reused"]]]
                if isinstance(shared_slot, np.void):
                    values.append(shared_slot)
                else:
                    try:
                        values.append(shared_slot[0])
                    except (IndexError, TypeError):
                        values.append(shared_slot)
            else:
                dtype:object = returns["slot_types"][entry["slot"]]
                kind:str = returns["slot_kinds"][entry["slot"]]
                values.append(_return_value(return_state["buffers"][entry["slot"]][0], dtype, kind))
        return tuple(values) if layout["is_tuple"] else values[0]

    def _with_returns(self, future:object,
                      return_state:Dict[str, object],
                      returns:Optional[Dict[str, object]],
                      full:List[object],
                      outer_state:Optional[Dict[str, object]]=None,
                      created:Optional[Dict[str, object]]=None)->object:
        """Wrap an asynchronous launch so its future resolves to the returned value.

        The backend future resolves once every read-back has completed (see the kernel
        classes), so by the time this callback runs the slots hold the winner's values.
        A backend future may be bound to a loop other than the caller's (the python
        backend installs one when none is running), so an asyncio result is delivered
        through a fresh future on the caller's running loop when there is one, and is
        otherwise completed in place on whatever loop the backend future uses.
        """
        is_asyncio:bool = isinstance(future, asyncio.Future)
        deliver:object = concurrent.futures.Future()
        try:
            caller_loop:object = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if is_asyncio:
            deliver = caller_loop.create_future() if caller_loop is not None else future

        def settle(*_args:object)->None:
            error:Optional[BaseException] = None
            try:
                error = future.exception()
            except BaseException as caught:  # cancelled, or never resolved
                error = caught

            def apply() -> None:
                if deliver.done():
                    return
                if error is not None:
                    deliver.set_exception(error)
                else:
                    # Write-back happens here, after the launch has completed and its
                    # buffers have been read back, so a declared outer scalar holds the
                    # value the kernel left in it by the time the caller sees the result.
                    if outer_state is not None:
                        self._write_back_outer(outer_state)
                    deliver.set_result(self._finish_returns(return_state, returns, full, created)
                                       if return_state else None)

            if is_asyncio:
                try:
                    future.get_loop().call_soon_threadsafe(apply)
                except RuntimeError:
                    pass  # the loop was closed before the launch finished
                return
            apply()

        if future.done():
            settle()
        else:
            future.add_done_callback(settle)
        return deliver

    def _with_tail(self, future:object,
                   tail_statements:List[ast.stmt],
                   args:Tuple[object, ...], instance:object,
                   carried_slots:Dict[str, object],
                   gap_slots:Dict[str, object],
                   created_objects:Dict[str, object],
                   outer_state:OuterState,
                   outer_writable:List[Dict[str, object]],
                   return_state:Dict[str, object],
                   returns:Optional[Dict[str, object]],
                   full:List[object],
                   created:Optional[Dict[str, object]]=None)->object:
        """Wrap an asynchronous launch so its future resolves the way a call would.

        The synchronous launch runs the host tail (the statements after the last
        loop, whose `return` is often the answer) and writes the outer scalars back
        after the read-backs land. The future must hand back the same thing: the
        backend future resolves once every read-back has completed, so the settle
        callback runs the tail there and finishes exactly as the inline path does.
        A body with neither a tail nor a write-back (and no device claim) never
        reaches this method. The backend future resolves from the driver's callback,
        and the tail must not race it: the value is delivered through a fresh future
        (a concurrent one, or one on the caller's loop for asyncio), settled only
        after the tail has run.
        """
        tail_writable:Dict[str, np.ndarray] = {
            entry["name"]: entry["buffer"] for entry in outer_writable}
        is_asyncio:bool = isinstance(future, asyncio.Future)
        try:
            caller_loop:object = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        deliver:object = (caller_loop.create_future()
                          if is_asyncio and caller_loop is not None
                          else (asyncio.Future(loop=_current_loop())
                                if is_asyncio else concurrent.futures.Future()))

        def settle(*_args:object)->None:
            error:Optional[BaseException] = None
            try:
                error = future.exception()
            except BaseException as caught:  # cancelled, or never resolved
                error = caught

            def settle_error() -> None:
                if not deliver.done():
                    deliver.set_exception(error)

            def apply() -> None:
                if error is not None:
                    if is_asyncio:
                        try:
                            future.get_loop().call_soon_threadsafe(settle_error)
                        except RuntimeError:
                            pass  # the loop was closed before the launch finished
                    else:
                        settle_error()
                    return
                try:
                    tail_result:object = self._run_host_region(
                        tail_statements, args, instance,
                        carried_slots, gap_slots, created_objects=created_objects,
                        outer_state=outer_state, writable_outer=tail_writable)
                    self._write_back_outer(outer_state)
                    if return_state and int(return_state["claim"][0]):
                        # A work item reached an in-loop return first: python's
                        # "first return wins".
                        result:object = self._finish_returns(
                            return_state, returns, full, created)
                    elif tail_result is not None:
                        result = tail_result
                    else:
                        result = (self._finish_returns(return_state, returns, full, created)
                                  if return_state else None)
                except BaseException as tail_error:
                    if is_asyncio:
                        try:
                            future.get_loop().call_soon_threadsafe(
                                self._deliver_error, deliver, tail_error)
                        except RuntimeError:
                            pass
                    else:
                        self._deliver_error(deliver, tail_error)
                    return
                if is_asyncio:
                    try:
                        future.get_loop().call_soon_threadsafe(
                            self._deliver_result, deliver, result)
                    except RuntimeError:
                        pass  # the loop was closed before the launch finished
                else:
                    self._deliver_result(deliver, result)

            apply()

        if future.done():
            settle()
        else:
            future.add_done_callback(settle)
        return deliver

    @staticmethod
    def _deliver_result(future:object, result:object)->None:
        """Complete a future with a value, unless something already settled it."""
        if not future.done():
            future.set_result(result)

    @staticmethod
    def _deliver_error(future:object, error:BaseException)->None:
        """Complete a future with an exception, unless something already settled it."""
        if not future.done():
            future.set_exception(error)

    def _invoke_python(self, instance:object, args:Tuple[object, ...],
                       device_sel:Optional[str],
                       global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                       launch:str)->object:
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
        runner:object = self._python_runner
        if runner is None:
            runner = PythonKernel(self.raw_func, self._is_method, self._self_param,
                                  domain=self._domain or "")
            self._python_runner = runner
        return runner.call(instance, args, global_size, local_size, launch=launch)

    # ---------- source generation ----------

    def _type_map_key(self)->str:
        """A stable spelling of this kernel's `int=`/`float=` mapping, for cache keys."""
        if not self._type_map:
            return ""
        return ",".join(f"{key}={self._type_map[key]}"
                        for key in sorted(self._type_map))

    def _source_signature(self)->str:
        """Python source signature: source changes -> regenerate (recompile when cached content differs).

        `@kernel(domain=...)` is deliberately **not** part of this hash, and that is safe only
        because the domain never reaches the generated code: it picks which array supplies the
        launch geometry, and geometry is a host-side, per-call fact. Two decorators over one
        body with different domains therefore share one source (they even compute one C name),
        and each instance keeps its own `_cache` entry so neither reuses the other's launch.
        If the domain ever starts to shape the device code, it has to be added here -- otherwise
        the two would share a source they no longer agree on.

        The `int=`/`float=` mapping **is** in this hash, because it does shape the device
        code: it decides which C type a python `int`/`float` parameter is declared as, so two
        mappings are two different translation units. Without it here, a kernel declaring
        `int=` would share the cache file (and the C name) of the default one, and the second
        launch would compile against a signature the caller did not ask for.
        """
        try:
            source:str = inspect.getsource(self.raw_func)
        except OSError:
            source = ""
        return short_hash(f"{source}\nbackend={self.backend}\ntypes={self._type_map_key()}")

    def _resolve_global(self, name:str)->object:
        """Resolve a name the kernel body reads from outside the function.

        A closure cell takes precedence over a module global, exactly as python resolves
        it, so a nested kernel sees its enclosing function's variable (which is what the
        outer-scope feature parameterizes).
        """
        for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                   self.raw_func.__closure__ or ()):
            if cell_name == name:
                try:
                    return cell.cell_contents
                except ValueError:  # the cell is still empty
                    return None
        return self.raw_func.__globals__.get(name)

    def _c_kernel_name(self, sig:str, backend:str, declaration:str="")->str:
        """The kernel's C name, unique per module and per what it declares.

        Two modules can define a kernel with the same name and the same argument types --
        two test modules both declaring calls_a_method, say -- and one generated file
        #includes the other. Without the module in the hash the two kernels share one C
        name, and the source defines it twice.

        `declaration` is the rendered parameter list, when the caller already has one. The
        name has to cover it as well as `sig`: `sig` is built from the *argument types* the
        caller passed, while the generated signature also carries the extents derived from
        the body, the outer-scope values and the shared buffers. Without them in the hash a
        body whose derived signature changed kept its old name, and the file then held two
        declarations of that name. It is folded into the *same* hash rather than appended
        as a second one, so a kernel name stays `name_<hash>` (see docs/generated-source.md).
        """
        source_file:str = getattr(getattr(self.raw_func, "__code__", None),
                                  "co_filename", "") or ""
        module_mark:str = os.path.basename(source_file)
        # A method reads `Class_method`, exactly as a device helper does: the receiver is
        # the first parameter, so the class is part of what the kernel is. A plain function
        # keeps its own name.
        base:str = self.__name__
        qualname:str = getattr(self.raw_func, "__qualname__", "")
        if "." in qualname:
            owner_name:str = qualname.rsplit(".", 1)[0]
            if "<" not in owner_name:
                base = f"{owner_name}_{self.__name__}"
        payload:str = module_mark + "|" + sig + "_" + backend
        if declaration:
            payload += "|" + declaration
        return f"{base}_{short_hash(payload)}"

    def _bind_images(self, full_values:List[object], backend:str)->None:
        """Create the device objects for every Image argument, before compilation.

        Each backend decides what an image becomes (a texture for reading, a surface
        for writing); the objects are cached on the Image so later launches reuse
        them. Only CUDA needs this ahead of compilation; OpenCL images are created
        per launch without that ordering constraint.
        """
        from ...image import Image

        for value in full_values:
            if not isinstance(value, Image):
                continue
            binding = value._bindings.get(backend)
            if binding is not None:
                continue
            if backend == "cuda":
                value.bind_cuda()
            # opencl binds at launch time; nothing to do here

    def _apply_declared_domain(self, translator:object)->None:
        """Record the array `@kernel(domain=...)` named as the iteration domain.

        The parameter has to exist and be an array: only a pointer carries the shape
        dimensions a launch is derived from, and a scalar has none to give. Naming a
        nonexistent parameter is the failure a string-valued decorator argument invites --
        renaming the parameter leaves the string behind -- so it is reported with the ones
        that do exist.

        A magic loop already fixes the domain, so a kernel carrying both would have two
        declarations of the same thing; that is checked here before translation, because the
        loop scan is what records `magic_loop_present`.
        """
        name:str = self._domain
        parameter_names:List[str] = _positional_argument_names(self.raw_func)
        if name not in parameter_names:
            available:str = ", ".join(parameter_names) or "none"
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) names no parameter of this "
                f"function (its parameters are: {available}). The domain is the array whose "
                f"shape declares the iteration domain, spelled as the parameter's own name")
        if _declares_a_magic_loop(self.raw_func):
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) declares the iteration domain, "
                f"but the body already declares one with a magic loop "
                f"(`for i in range(x.shape[0])` or `np.ndindex`); keep exactly one of them")
        index:Optional[int] = None
        for position, (parameter, _) in enumerate(translator.params):
            if parameter == name:
                index = position
                break
        if index is None or not is_ptr(translator.params[index][1]):
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) requires an array parameter; "
                f"'{name}' is not an array, so it has no shape to derive a work size from")
        translator.declared_work_arrays = [name]

    def _compile(self, args:List[object], arg_types:List[object], self_types:List[object],
                 self_attrs:List[str], owner:Optional[type], sig:str,
                 backend:str, device_obj:object, instance:object=None,
                 self_struct_members:Optional[Dict[str, str]]=None,
                 self_struct_type:object=None)->Tuple[object, List[str]]:
        if backend in ("cuda", "hip"):
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise SyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead") from None

        c_name:str = self._c_kernel_name(sig, backend)
        device_cache:Dict[object, object] = {}
        # self_types leads with the receiver struct when the receiver travels as one, so
        # the attribute map pairs the array attributes with the types after it.
        self_attr_types:Dict[str, object] = dict(zip(
            self_attrs, self_types[1:] if self_struct_members else self_types))

        # hip accepts CUDA syntax (threadIdx/atomicAdd/make_float4 etc.), so codegen goes through cuda
        codegen_backend:str = ("cuda" if backend == "hip" else backend)
        # Statements above the magic loop that only build data are executed by the
        # interpreter, not translated: numpy decides the shape and dtype, and the resulting
        # object is what the kernel receives as a parameter (see
        # FunctionTranslator._host_created_assignment). The values and the line numbers were
        # computed once for this call, before the body was scanned (see _invoke_inner).
        creation_values, _ = _host_created_values(self.raw_func, args, instance)
        executed_lines:List[int] = list(getattr(self, "host_region_statements", []))
        region_values:Dict[str, object] = dict(
            getattr(self, "host_region_snapshot", None) or creation_values)
        # The class each STRUCTURE argument was actually built from. One dtype can stand
        # for several classes, so the dtype alone cannot say whose method `p.total` is --
        # but the launch has the instance the caller handed over, and its type is the
        # answer. Keyed by parameter name, which is what the translator's expressions carry.
        # `structure_dtype_of` excludes the vector types, which are ctypes classes too.
        struct_arg_classes:Dict[str, type] = {}
        parameter_names:List[str] = _positional_argument_names(self.raw_func)
        for parameter_name, value in zip(parameter_names, args):
            # A union is a `ctypes` record as well but not a `Structure` subclass, so the
            # class is recognized through the shared predicate rather than `ctypes.Structure`.
            if not structs.is_ctypes_record(value):
                continue
            # structure_dtype_of excludes the vector types, which are ctypes classes too.
            if structs.structure_dtype_of(value) is None:
                continue
            struct_arg_classes[parameter_name] = type(value)
        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), codegen_backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types,
            self_owner=owner, launch_backend=backend,
            self_struct_members=self_struct_members,
            self_struct_type=self_struct_type,
            struct_arg_classes=struct_arg_classes,
            host_creation_values=region_values,
            host_executed_lines=executed_lines,
            type_map=self._type_map)
        # `@kernel(domain="x")` names the parameter whose shape declares the iteration
        # domain. It is applied before `translate()` so the declaration is in place while the
        # magic loop is scanned -- a body cannot carry one of its own any more.
        if self._domain:
            self._apply_declared_domain(translator)
        translator.translate()
        # The name has to cover what the translator actually declares. `sig` is built from
        # the *argument types* the caller passed, which do not include the extent parameters
        # derived from the body, the outer-scope values or the shared buffers -- so a body
        # whose derived signature changed kept its old name, and the generated file ended up
        # holding two declarations of it ("more than one instance of overloaded function").
        # Re-deriving the name with the rendered declaration folds that into the same hash.
        c_name = self._c_kernel_name(sig, backend,
                                     declaration=";".join(translator.parameter_declarations()))
        # One kernel per magic loop: a body with several renders several, launched in order
        # so the arrays stay on the device between them.
        rendered:List[Tuple[str, str]] = translator.render_kernels(c_name)
        kernel_code:str = "\n\n".join(source for _, source in rendered)

        # The generated device code is split the way the python code is (see source_paths):
        # a kernel and every helper written in its own module go to that module's source
        # file, and a helper from another module goes to that module's file and is reached
        # with one `#include`. One file per python module means one compilation unit per
        # module, so a callee from another module still has to be reachable -- which is what
        # the include does (the preprocessor expands it before the compiler runs).
        kernel_source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        # What the generated file records so a later call can tell whether it still
        # describes this module's text. mtime alone is not reliable (`git checkout` restores
        # the old stamp), so the digest is what actually decides.
        source_digest:str = source_paths.python_source_digest(kernel_source_file)

        def module_of(entry:Dict[str, object])->str:
            """The python file a device function came from, spelled as a cache path."""
            return source_paths.source_path(kernel_source_file, entry.get("source_file", ""),
                                            backend)

        # Device helper functions are emitted in dependency order: C requires a callee to be
        # declared before its caller, and the cache is filled depth-first (a nested helper
        # is finished *after* the helper that calls it). Emit leaves first.
        ordered_entries:List[Dict[str, object]] = _dependency_order(device_cache)
        own_entries:List[Dict[str, object]] = [
            entry for entry in ordered_entries
            if module_of(entry) == module_of({"source_file": kernel_source_file})]
        foreign_entries:Dict[str, List[Dict[str, object]]] = {}
        for entry in ordered_entries:
            path:str = module_of(entry)
            if path == module_of({"source_file": kernel_source_file}):
                continue
            foreign_entries.setdefault(path, []).append(entry)

        # A nested launch needs the callee's own `__global__` function in this same source:
        # the launch statement names a device function, and C requires it before its caller.
        # A built-in that a backend spells with a helper contributes one too, and it is a
        # device function the kernel calls, so it goes in the same place.
        device_prefix:List[str] = (list(translator.device_helpers)
                                   + list(translator.nested_helpers)
                                   + list(translator.nested_kernels))
        device_prefix.extend(entry["code"] for entry in own_entries)

        structure_dtypes:List[DeviceType] = list(translator.structure_dtypes())
        for entry in device_cache.values():
            for dtype in entry.get("structs", ()):
                if dtype not in structure_dtypes:
                    structure_dtypes.append(dtype)
        # Every structure this compilation reaches is declared by the kernel's own file. A
        # helper in another module names its types too, and that module's file is reached
        # from here, so the declaration has to be in scope before the include is read --
        # putting it in the helper's own file instead made a cycle of includes and, when two
        # modules shared one structure, the same declaration twice in one file.
        typedefs:str = (structs.render_structures(structure_dtypes, translator.backend)
                        if structure_dtypes else "")
        # A structure member whose type a fixed header declares needs that header even when
        # the body never operates on the value: `struct S { c64 z; };` names `c64`, so the
        # declaration has to be in scope. Walking the member dtypes here is what covers that
        # case -- the translator only records a width when an operation uses one.
        from .. import complex as complex_support  # noqa: PLC0415 (import cycle)
        from .. import quaternion as quaternion_support  # noqa: PLC0415 (import cycle)

        member_quaternion_needs:Set[str] = set()
        member_complex_needs:Set[str] = set()
        for dtype in structure_dtypes:
            record:np.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
            for name in record.names or ():
                field_dtype:np.dtype = np.dtype(record.fields[name][0])
                if field_dtype.subdtype is not None:
                    field_dtype = field_dtype.subdtype[0]
                quat_need:Optional[str] = quaternion_support.needs(field_dtype)
                if quat_need is not None:
                    member_quaternion_needs.add(quat_need)
                complex_need:Optional[str] = complex_support.needs(field_dtype)
                if complex_need is not None:
                    member_complex_needs.add(complex_need)

        # Quaternion structures and their arithmetic both come from a fixed header, copied
        # into the cache's include folder beside the generated source. The source needs only
        # the enable macros for the widths in use, then one include -- no declaration of its
        # own, because the header spells the two type names in advance.
        quaternion_needs:Set[str] = set(translator.quaternion_needs)
        # A member declared by the fixed header needs it even when the body never
        # operates on the value (see the walk above).
        quaternion_needs.update(member_quaternion_needs)
        for entry in device_cache.values():
            quaternion_needs.update(entry.get("quaternion_needs") or ())
        include_lines:str = ""
        if quaternion_needs:
            from .. import quaternion as quaternion_header

            include_lines = quaternion_header.include_line(
                tuple(sorted(quaternion_needs)), backend, self._cache_dir(backend))

        # Complex values work the same way, from their own fixed header: the two structures
        # and every operation on them are declared there, so a source that includes it says
        # nothing else about a complex number. A helper's blocks travel with its cache entry,
        # exactly as a helper's quaternion widths do.
        complex_needs:Set[str] = set(translator.complex_needs)
        complex_needs.update(member_complex_needs)
        for entry in device_cache.values():
            complex_needs.update(entry.get("complex_needs") or ())
        if complex_needs:
            from .. import complex as complex_header

            complex_lines:str = complex_header.include_line(
                tuple(sorted(complex_needs)), backend, self._cache_dir(backend))
            include_lines = (f"{include_lines}\n{complex_lines}" if include_lines
                             else complex_lines)

        # A registered device-side library function may need headers the compilation does
        # not have yet. A header the user ships as a *path* is copied into the cache's
        # include folder and included by name; a header given as a full `#include` line
        # (a system header, say) is carried into the generated source verbatim.
        library_header_lines:List[str] = []
        if translator.device_library_headers:
            from ..device_library import include_lines_for

            library_header_lines = include_lines_for(
                backend, translator.device_library_headers, self._cache_dir(backend))
        if library_header_lines:
            library_includes:str = "\n".join(library_header_lines)
            include_lines = (f"{include_lines}\n{library_includes}" if include_lines
                             else library_includes)

        # Every top-level definition is separated by one blank line: helper from helper,
        # helper from kernel. A trailing blank on each piece makes the join produce exactly
        # one, and the final trim drops the one after the last definition.
        pieces:List[str] = [piece.rstrip("\n") for piece in device_prefix]
        pieces.append(kernel_code.rstrip("\n"))

        own_path:str = module_of({"source_file": kernel_source_file})
        # The includes come first: a callee from another module is a declaration this file
        # needs before anything that calls it. The order is the order the callees were
        # emitted, so a foreign helper that itself includes another is already covered.
        # A structure declared by another module: that module's file holds the `struct`,
        # and this file includes it rather than repeating the declaration. A structure with
        # no python class behind it stays with the file that uses it.
        # The includes are ordered the same way the definitions are: a module whose code
        # calls another module's function has to be reached after that callee, because C
        # requires a function before its caller. The definitions are already in dependency
        # order, so the modules are taken in the order their first definition appears.
        ordered_foreign:List[str] = []
        for entry in ordered_entries:
            path:str = module_of(entry)
            if path in foreign_entries and path not in ordered_foreign:
                ordered_foreign.append(path)
        foreign_includes:List[str] = [
            source_paths.include_line(own_path, path) for path in ordered_foreign]
        # A generated file is a *header* plus a list of definitions. The header names every
        # structure the file's own code uses and includes every other module it reaches;
        # the definitions are appended one at a time, so a module with several kernels keeps
        # all of them in one file. The header comes first because C needs a type before the
        # function that names it, and a callee before its caller.
        own_header:List[str] = []
        if backend == "metal":
            # Metal's prelude: the `#include <metal_stdlib>` and the macros that give the
            # shared body its OpenCL index spellings (see kernel/metal_codegen.py). It
            # precedes the typedefs because the macros are the first thing the kernel body
            # needs -- and it must be first in the *file*, before any declaration.
            from ..metal_codegen import prelude
            own_header.append(prelude())
        if include_lines:
            # The fixed headers come BEFORE the typedef block: a structure member can name
            # `quatf` or `c64`, and that declaration has to be in scope when the structure
            # is declared, not after it.
            own_header.append(include_lines)
        if typedefs:
            own_header.append(typedefs)
        if foreign_includes:
            own_header.append("\n".join(foreign_includes))

        written:Dict[str, str] = {}
        for path, entries in foreign_entries.items():
            # A foreign module's file declares no structures of its own. The kernel's file
            # carries the whole typedef block ahead of the includes that reach these files,
            # and each declaration is guarded by its structure name, so the type is declared
            # once no matter how many files are compiled from it. Repeating the block here
            # would put every declaration in every file, which is a cycle of includes and a
            # duplicate definition in the one file two modules share.
            foreign_header:List[str] = []
            if include_lines:
                foreign_header.append(include_lines)
            # A foreign module may itself call into a third module: every structure is
            # declared by the top-level file, but a *function* it calls has to be reached
            # from its own file, and only by the files that call it.
            callee_includes:List[str] = [
                source_paths.include_line(path, other) for other in ordered_foreign
                if other != path and other in _called_modules(path, foreign_entries)]
            if callee_includes:
                foreign_header.append("\n".join(callee_includes))
            written[path] = source_paths.append_source(
                path, foreign_header,
                [entry["code"] for entry in entries if entry["code"]],
                entries[0].get("source_file", "") or "",
                source_digest=source_digest)

        source_path:str = source_paths.append_source(
            own_path, own_header, [piece for piece in pieces if piece],
            kernel_source_file, source_digest=source_digest)
        # The whole file is one translation unit: a nested launch anywhere in it makes NVRTC
        # reject the source, so the *file* decides the build path, not the kernel being
        # compiled. A module with one nesting kernel and one ordinary kernel must be built
        # by nvcc even when the ordinary kernel is launched first, so the decision reads the
        # file that was actually written (it accumulates every kernel of the module).
        linked:bool = translator.nests_a_launch
        if not linked:
            try:
                with open(source_path, "r", encoding="utf-8") as file:
                    linked = "<<<" in file.read()
            except OSError:
                linked = False
        # The write above and the compile below are one critical section. A sibling kernel
        # of this same module may be appending its definition concurrently, and the compile
        # keys its cached binary on the file's *content*: a compile that started between two
        # appends would cache a binary of a half-populated file, and the kernel appended last
        # would find no entry point of its own in it (`KeyError` naming the kernel). Holding
        # the lock across both steps makes the file's content and the binary that describes
        # it change together.
        with source_paths.write_lock(source_path):
            kernels:List[object] = [
                self._backend_compile(source_path, name, backend, device_obj, linked=linked)
                for name, _ in rendered]
        # A phase kernel declares only the shapes it indexes itself, so its launch needs its
        # own trailing extent arguments rather than the ones the first kernel wants.
        extra_shapes:List[List[Tuple[int, int]]] = []
        # The created objects each phase declares: mid = np.zeros(4) between two loops is
        # a parameter of the pass that reads it and of no other.
        phase_created:List[List[str]] = []
        # The per-phase parameter metadata (empty for a single-phase body, which never
        # enters _translate_phase_kernels; the fallback below fills it from the translator).
        # A gid-style body has no phases and no magic loop: its launch arguments are the
        # shared prefix plus the extents, which the fast path already builds, and inventing
        # a per-phase list here would double-count the outer-scope parameters.
        is_magic_body:bool = bool(getattr(translator, "phases", None)) and \
            len(getattr(translator, "phases", [])) == 1 and \
            any(isinstance(stmt, ast.For) for stmt in translator.tree.body)
        phase_carried:List[List[str]] = []
        phase_param_names:List[List[str]] = []
        # Per phase, whether its kernel declares the return claim/site words (see
        # render_return_params): a pass whose only return runs in its host tail does not.
        phase_claims:List[bool] = []
        if translator.phase_translators:
            for phase in translator.phase_translators:
                phase_index:Dict[str, int] = {
                    name: position for position, (name, _) in enumerate(phase.params)}
                extra_shapes.append([(phase_index[array_name], dim)
                                     for array_name, dim in phase.shapes_used])
                phase_created.append(list(phase.host_values))
                phase_claims.append(bool(phase._claims_on_device))
        else:
            # One magic loop: the main translator *is* the only phase, and it binds the
            # created objects on itself.
            phase_index:Dict[str, int] = {
                name: position for position, (name, _) in enumerate(translator.params)}
            extra_shapes.append([(phase_index[array_name], dim)
                                 for array_name, dim in translator.shapes_used])
            phase_created.append(list(translator.host_values))
            # The one phase is the main translator: its parameters and carried slots are
            # the launch's, and the per-phase metadata would otherwise stay empty (the
            # launch then aligns against nothing and drops every argument after the first).
            if is_magic_body:
                phase_carried.append(list(translator.carried_scalars))
                phase_param_names.append(list(dict(translator.params)))
                phase_claims.append(bool(translator._claims_on_device))

        launch:object = kernels[0]

        # Atomic targets: static analysis cannot recognize pointer-cast writes, so force them to writable
        from ...constants import KernelFlags
        for target_name in translator.atomic_targets:
            if target_name in launch._args:
                launch._args[target_name].kernel_flags = (launch._args[target_name].kernel_flags | KernelFlags.Writed)

        # The return slots are written by the device and read back on the host, so they stay
        # fully bidirectional. `atomicCAS(&_ret_claim[0], 0, 1)` *reads* the claim word to
        # compare it against zero, and a slot that was never uploaded would compare against
        # whatever the device buffer happened to hold -- the claim would fire at random or
        # never, and a launch would return the wrong site's value. That is the same reason a
        # write-only argument is still uploaded, applied to a slot the library owns rather
        # than to the caller's data.
        for slot_name in (translator.RETURN_CLAIM, translator.RETURN_SITE):
            slot_info:object = launch._args.get(slot_name)
            if slot_info is not None:
                slot_info.kernel_flags = (KernelFlags.Readed | KernelFlags.Writed)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        # A multi-phase body is translated per phase, so the main translator indexes nothing;
        # the first phase's shapes are the ones this launch derives its work size from.
        shapes_owner:object = (translator.phase_translators[0]
                            if translator.phase_translators else translator)
        shape_index:Dict[str, int] = {
            name: position for position, (name, _) in enumerate(shapes_owner.params)}
        shape_sources:List[Tuple[int, int]] = [
            (shape_index[array_name], dim) for array_name, dim in shapes_owner.shapes_used]
        if not extra_shapes:
            # A single kernel: its own extents are the launch's.
            extra_shapes = [list(shape_sources)]
        # Each entry is (parameter index, dimension) for an array-sourced extent, or
        # (None, count) for a literal one (np.ndindex(2, 3)).
        work_size_dims:List[Tuple[Optional[int], int]] = [
            (param_index.get("self"), dim)
            if name.startswith("(self)->") else
            (shape_index[name], dim) if name else (None, dim)
            for name, dim in shapes_owner.work_size_dims]
        # An `@kernel(domain=...)` declaration: expand each named array into its own
        # dimensions. The rank is only known at call time, so record (param, -1) markers and
        # let _invoke substitute the real per-dimension indices.
        if not work_size_dims and shapes_owner.declared_work_arrays:
            work_size_dims = [(shape_index[name], -1)
                              for name in shapes_owner.declared_work_arrays]
        # local memory: opencl passes byte sizes at launch by parameter position; cuda uses __shared__ declarations and needs nothing passed
        ctype_sizes:Dict[str, int] = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2,
                                      "int": 4, "uint": 4, "long": 8, "ulong": 8,
                                      "float": 4, "double": 8}
        local_sizes:List[int] = [ctype_sizes.get(ctype, 4) * count for _, ctype, count in translator.local_buffers]
        # Images an OpenCL kernel samples need a sampler argument each; record which
        # parameter supplies the sampler so the launch can append it.
        image_samplers:List[int] = [param_index[name]
                                    for name in sorted(translator.required_image_samplers)]
        # Return slots: one per distinct value type any return site writes, plus the
        # claim/site words. `arg_indices` maps an array a return site names to its
        # position in the launch arguments, so the result can hand back that same object.
        returns:Dict[str, object] = {
            "slot_types": [translator.return_slot_dtype(index)
                           for index in range(len(translator.return_layout))],
            "slot_kinds": [entry["type"] for entry in translator.return_layout],
            "sites": translator.return_site_layout(),
            "arg_indices": {name: position for name, position in param_index.items()},
        }
        return (kernel, shape_sources, work_size_dims, local_sizes,
                shapes_owner.uses_global_id, image_samplers,
                returns if translator.has_returns else None,
                translator.outer_vars,
                {"param_names": {position: name for name, position in param_index.items()},
                 # Every kernel this body produced, in launch order: one per magic loop.
                 "carried": list(translator.carried_scalars),
                 # Each carried slot's element type: the host allocates and reads the
                 # slot at this width (`total = 0.0` is a float slot, and an int32 host
                 # buffer under a `float*` parameter truncated every answer).
                 "carried_types": {name: t for name, t in
                                   ((parameter_name, parameter_t[1])
                                    for parameter_name, parameter_t in translator.params
                                    if parameter_name in translator.carried_scalars
                                    and isinstance(parameter_t, tuple)
                                    and len(parameter_t) > 1)},
                 "gap_names": list(translator.gap_names),
                 # The extra parameters each phase declares beyond the shared prefix: a gap
                 # value belongs to the pass that reads it (see _translate_phase_kernels).
                 "phase_carried": (phase_carried if phase_carried else
                                   [list(phase.carried_scalars)
                                    for phase in translator.phase_translators]),
                 # The statements between two passes: the host runs region i between
                 # launch i and launch i+1 (see _run_host_region).
                 "phase_gaps": getattr(translator, "phase_gaps", []),
                 # Each phase's parameters in order, so the launch aligns values to the
                 # signature (a phase may declare a carried name the first kernel does not).
                 "phase_param_names": (phase_param_names if phase_param_names else
                                       [list(dict(phase.params))
                                        for phase in translator.phase_translators]),
                 # Whether each phase's kernel declares the return claim/site words. A
                 # single-phase body has one kernel (the main translator's) and no entry
                 # here: the launch then treats it as claiming, which its signature says.
                 "phase_claims": (phase_claims if phase_claims else
                                  [bool(phase._claims_on_device)
                                   for phase in translator.phase_translators]),
                 # Whether the main kernel declares the claim/site words. The fast path
                 # launches that kernel with the argument list built once up front, so this
                 # has to describe the *first* kernel: a phase of a multi-loop body is one
                 # kernel of its own, while a single-loop body's only kernel is the main
                 # translator's (see render_return_params).
                 "main_claims": (bool(translator.phase_translators[0]._claims_on_device)
                                 if translator.phase_translators
                                 else bool(translator._claims_on_device)),
                 # The statements after the last magic loop, for a body split into passes:
                 # `body_tail` above belongs to a single-loop body, whose translation never
                 # enters _translate_phase_kernels. The last phase's own tail is the one the
                 # host runs after the final launch.
                 "phase_tail": (list(translator.phase_translators[-1].body_tail)
                                if translator.phase_translators else []),
                 # The receiver-struct members whose values are loop extents, in the
                 # order the -3 markers appear in work_size_dims (see _invoke's -3 branch).
                 "work_size_members": list(getattr(translator, "work_size_members", [])),
                 # The statements after the last magic loop of a single-phase body: the host
                 # runs them once, after the launch (see _run_host_region).
                 "body_tail": getattr(translator, "body_tail", []),
                 "kernels": kernels,
                 # The trailing extent arguments each phase kernel needs, by phase.
                 "kernel_shapes": extra_shapes,
                 "kernel_created": phase_created,
                 # Every created name any phase declares, so the host knows which objects
                 # the launch has to supply (a returned creation below the loop included).
                 "kernel_created_all": sorted({name
                                               for names in phase_created for name in names}),
                 # The plain values a host-region statement bound, in the device-parameter
                 # order they were appended (a `w = len(label)` above the loop, an
                 # arbitrary-precision number the kernel receives as a scalar), and the
                 # values themselves for the launch to hand over.
                 "host_values": list(translator.host_scalars),
                 "host_region_values": {name: value for name, value
                                        in self.host_region_snapshot.items()
                                        if name in translator.host_scalars},
                 "subscript_indices": dict(translator.subscript_indices),
                 # name -> the line of its widest subscript, and the offset that turns a
                 # parsed-copy line into the user's own: the launch blames that statement,
                 # not the loop, when a subscript names more indices than the rank.
                 "subscript_lines": dict(translator.subscript_lines),
                 "subscript_columns": dict(translator.subscript_columns),
                 # name -> the subscript as the user wrote it, so the frame stand-in can
                 # re-run that expression and let CPython draw its own caret.
                 "subscript_texts": dict(translator.subscript_texts),
                 # The arrays a local was sliced from (`part = a[lo:hi]`): a slice is one
                 # pointer at an offset, which is only the same thing as python's view when
                 # the array has one dimension. Checked at launch, where the rank is known.
                 "sliced_arrays": list(translator.sliced_arrays),
                 "source_first_line": int(getattr(translator, "first_line", 1) or 1),
                 "loop_line": int(getattr(translator, "magic_loop_line", 0) or 0),
                 # How many index names the magic loop unpacks: the launch checks it against
                 # the number of dimensions its iteration domain actually has.
                 "loop_names": len(translator.work_item_indices),
                 # The array behind a `np.ndindex(arr.shape)` domain (index into the device
                 # parameter list, or -1), whose rank the loop must name one index per.
                 "whole_shape_index": param_index.get(translator.nd_whole_shape_param, -1)})

    def _cache_dir(self, backend:str)->str:
        """The pykernels folder this kernel's generated sources live in.

        It is the shared root the per-module files sit under, so the quaternion header can
        be placed there once and every module's source reach it.
        """
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        return source_paths.cache_root(source_file, backend)

    def _backend_compile(self, source_path:str, c_name:str, backend:str, device_obj:object,
                         linked:bool=False):
        if backend == "opencl":
            from ...opencl.oop.program import Program
            program:object = Program(device_obj.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        if backend == "hip":
            from ...hip.oop.program import Program as HIPProgram
            from ...hip.oop.hip_compiler import HipCompiler
            device_obj.default_context.make_current()
            compiler = HipCompiler(source_path)
            program:object = HIPProgram(source_path, compiler=compiler)
            return program.kernels[c_name]

        if backend == "metal":
            from ...metal.oop.program import Program as MetalProgram
            program:object = MetalProgram(source_path, device=device_obj)
            return program.kernels[c_name]

        from ...cuda.oop.program import Program as CUProgram

        device_obj.default_context.make_current()
        if linked:
            # A kernel that launches another kernel needs *linked* device code: the child is
            # resolved through the device runtime, which NVRTC does not link in (and cuLink
            # on Windows will not read cudadevrt.lib). nvcc performs that link, so only such
            # a source pays for it; everything else stays on NVRTC.
            return _compile_linked_cuda(source_path, c_name, device_obj)

        program:object = CUProgram(source_path)
        return program.kernels[c_name]

    # ---------- launch ----------

    def _build_launch_args(self, args:List[object], arg_types:List[object], shape_sources:List[Tuple[int, int]])->List[object]:
        launch_args:List[object] = list(args)
        for param_index, dim in shape_sources:
            launch_args.append(int(args[param_index].shape[dim]))
        return launch_args

    def __repr__(self)->str:
        return f"PyKernel({self.__name__}, backend={self.backend})"