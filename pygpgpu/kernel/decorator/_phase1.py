from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from .launch_plan import ANSWERED
import numpy as np
from ...utils import device_limits
from ..translator import (
    PTR,
    DeviceType,
    baked_shared_shapes,
    infer_ctype,
    scan_self_attrs,
    type_sig,
)
from ..objects import (ObjectBinding, _complex_member_value,  # noqa: PLC2701 (shared
                    _is_complex_member_class, convert_object_arguments)
from .devices import (
    _default_target,
    _resolve_device,
)
from .host_regions import (
    _host_created_values,
    _host_region_values,
)


class _PrepareLaunchMixin:

    def _prepare_launch(self, plan:object, instance:object, args:Tuple[object, ...],
                    backend_sel:Optional[str], device_sel:Optional[str],
                    global_size:Optional[Tuple[int, ...]],
                    local_size:Optional[Tuple[int, ...]],
                    launch:str = "call")->object:
        """Everything that decides WHICH kernel runs: the backend, the device, the
        receiver's structure, the argument types and the cache key.

        The compiled kernel is cached, so a second call with the same types and the
        same mapping reuses it. The mapping is part of the key, not only of the
        signature: two kernels over the same argument values under different
        mappings produce `int` and `long` parameters, and only one cache entry may
        answer for a launch.
        """
        if backend_sel is not None:
            plan.backend:str = backend_sel
        else:
            # No bracket: the kernel's own @kernel(default_device=...) entries, when the
            # decorator named any, replace the walk; otherwise config.default_device
            # decides and the built-in probe order is the last resort. Read now rather
            # than at decoration time so a later assignment takes effect.
            _backend, _default_device = _default_target(self._default_devices)
            plan.backend = _backend
            plan.default_device = _default_device
            if device_sel is None:
                device_sel = plan.default_device
        # Record what this call actually used, so `k.backend` describes the last launch
        # rather than the snapshot taken when the decorator ran.
        self.backend = plan.backend
        if plan.backend == "python":
            # The interpreter backend answers the call here, so the other two phases have
            # nothing to do. The answer goes through the plan: a kernel that returns
            # nothing answers None, so the facade cannot read the answer off the return
            # value alone.
            plan.result = self._invoke_python(instance, args, device_sel, global_size,
                                              local_size, launch)
            return ANSWERED
        plan.device_obj:object
        plan.device_key:str
        _device_obj, _device_key = _resolve_device(plan.backend, device_sel)
        plan.device_obj = _device_obj
        plan.device_key = _device_key
        _device_max_group, _device_max_items = device_limits(plan.device_obj)
        plan.device_max_group = _device_max_group
        plan.device_max_items = _device_max_items

        plan.self_values:List[object] = []
        plan.self_attrs:List[str] = []
        plan.owner:Optional[type] = None
        plan.self_struct_members:Dict[str, str] = {}
        plan.self_struct_type:object = None
        # The host regions -- the statements above the first magic loop, between two of
        # them, and after the last one -- run **once per call**, here, before anything else
        # looks at the body. Their values are this call's (a statement may read the caller's
        # arguments), and the lines tell the translator which statements are host work, so
        # it emits no device code for them (see _host_region_values).
        plan.host_executed:List[int] = []
        plan.host_values_now:Dict[str, object] = {}
        if not self._is_method or instance is not None:
            plan.built:Dict[str, object] = _host_created_values(
                self.raw_func, tuple(args), instance)[0]
            _host_values_now, _host_executed = _host_region_values(
                self.raw_func, tuple(args), instance, set(plan.built))
            plan.host_values_now = _host_values_now
            plan.host_executed = _host_executed
            plan.host_values_now = {**plan.built, **plan.host_values_now}
        self.host_region_snapshot = plan.host_values_now
        self.host_region_statements = plan.host_executed
        if self._is_method:
            if instance is None:
                raise TypeError(f"{self.__name__}() requires an instance")
            # The decorated body was scanned at decoration time, before the class
            # existed, so self.<helper> could only look like data. Scan again now
            # that the instance is here: calling a self-method makes the attributes
            # *it* reads kernel arguments too, recursively.
            plan.owner = type(instance)
            # Which statements the host regions run is a fact about the *source*, so the lines
            # computed above are what decides which methods are host-only (see `_host_region_values`).
            plan.self_attrs = scan_self_attrs(self.raw_func, self._self_param, plan.owner,
                                         set(plan.host_executed))
            for name in plan.self_attrs:
                if not hasattr(instance, name):
                    raise SyntaxError(
                        f"instance of {plan.owner.__name__} has no attribute '{name}' "
                        f"required by kernel {self.__name__}") from None
                value:object = getattr(instance, name)
                if callable(value) and not isinstance(value, np.ndarray):
                    raise SyntaxError(
                        f"self.{name} is a function, not a value: a function cannot be passed "
                        f"to a device (call it as self.{name}(...) instead)") from None
                plan.self_values.append(value)
            # The scalar members travel as ONE structure the body reads through (the way a
            # device helper already takes `Scene self`); an ndarray member stays its own
            # argument, because the read/write analysis and the shape plumbing need it
            # separately. The record is one element long so `self.<attr> = ...` writes
            # through it, and the struct pointer is read-only when the body never assigns
            # a member.
            plan.struct_attrs:List[str] = [name for name in plan.self_attrs
                                      if not isinstance(getattr(instance, name), np.ndarray)]
            plan.array_attrs:List[str] = [name for name in plan.self_attrs
                                     if isinstance(getattr(instance, name), np.ndarray)]
            if plan.struct_attrs:
                from ..objects import _structure_for, member_type  # noqa: PLC0415 (import cycle)

                plan.self_struct:type = _structure_for(
                    [(name, member_type(getattr(instance, name), name, self._type_map))
                     for name in plan.struct_attrs], [])
                from ..structs import register_name_hint  # noqa: PLC0415 (import cycle)

                register_name_hint(np.dtype(plan.self_struct), plan.owner)
                # Each attribute is a member of the same name: the mapping is the identity,
                # never a zip against the dtype's names. numpy sorts a record's field names
                # alphabetically, so zipping them against the attribute order paired
                # `image_height` with `du_in_world` and every member read the wrong field.
                plan.self_struct_members = {name: name for name in plan.struct_attrs}
                from .. import structs as _structs  # noqa: PLC0415 (import cycle)

                plan.self_struct_type = (PTR, _structs.register(np.dtype(plan.self_struct)))
                # The value that travels is the ctypes structure itself: the launch packs
                # its bytes through the parsed layout, which handles vector members (whose
                # numpy record view uses a sub-array form that would misread the bytes).
                # A body that assigns `self.<attr> = ...` writes through the device buffer,
                # and the host copies the members home afterwards.
                plan.holder:object = plan.self_struct()
                for name in plan.struct_attrs:
                    plan.member_value:object = getattr(instance, name)
                    plan.field_type:object = dict(plan.self_struct._fields_).get(name)
                    if _is_complex_member_class(plan.field_type) \
                            and not isinstance(plan.member_value, plan.field_type):
                        plan.member_value = _complex_member_value(plan.member_value, plan.field_type)
                    setattr(plan.holder, name, plan.member_value)
                plan.self_attrs = plan.array_attrs
                plan.self_types = [plan.self_struct_type] + [
                    infer_ctype(getattr(instance, name), self._type_map)
                    for name in plan.array_attrs]
                plan.self_values = [plan.holder] + [
                    getattr(instance, name) for name in plan.array_attrs]

        # An arbitrary python object becomes a synthesized structure before its type is
        # inferred: the structure *is* the argument type, and every step after this sees the
        # same `ctypes.Structure` a caller would have declared by hand. The binding records
        # the pairing, so member writes travel back to the caller's object after the launch.
        # The receiver travels along too: a body that builds its own value (`ray = Ray(...)`)
        # types the construction's arguments from what the receiver's members are, and the
        # receiver's instance is the only place that answer lives.
        plan.object_binding:ObjectBinding = ObjectBinding()
        plan.args = convert_object_arguments(self.raw_func, args, self._resolve_global,
                                             self._self_param, plan.object_binding,
                                             self_value=instance,
                                             type_map=self._type_map)
        plan.arg_types:List[DeviceType] = [infer_ctype(a, self._type_map) for a in plan.args]
        if not plan.self_struct_members:
            plan.self_types = []
            for name, value in zip(plan.self_attrs, plan.self_values):
                try:
                    plan.self_types.append(infer_ctype(value, self._type_map))
                except (SyntaxError, ValueError) as e:
                    # `ValueError` is the width refusal (a python int too wide for the
                    # mapping); both lose their own context here, so each is re-raised with
                    # the attribute that carries it.
                    raise type(e)(f"self.{name}: {e}") from None

        plan.sig:str = type_sig(plan.arg_types + plan.self_types)
        # A `local_array_per_group` shape read from a kernel parameter bakes the *value* into the
        # generated source, so the value specializes the kernel the way a type does: it has
        # to be part of the identity both the `_cache` entry and the C name hash, or a second
        # call with a different shape would reuse the first one's binary and launch a grid
        # against a buffer of the wrong size. Only the parameters the source actually names
        # are hashed, so a kernel without a shared buffer keys exactly as it always did.
        baked:Tuple[Tuple[str, int], ...] = baked_shared_shapes(self.raw_func, list(args),
                                                                self._resolve_global)
        if baked:
            plan.sig:str = f"{plan.sig}|shapes={','.join(f'{name}={value}' for name, value in baked)}"
        # The mapping is part of the cache key, not only part of the type signature: two
        # kernels over the same argument values under different mappings produce `int` and
        # `long` parameters, and only one `_cache` entry may answer for a launch. The types
        # already differ (`sig` carries them), so this is belt-and-braces for the case where
        # a mapping is changed to a width whose type name another entry already used.
        # The resolved owner is part of the key for the same reason: a method kernel's body
        # compiles against the hook the *receiver's* class supplies, and two subclasses of
        # one `@kernel` declare the same argument types -- without the class here the
        # second subclass to launch reused the first one's binary and silently answered its
        # numbers (see tests/regressions/subclass_kernel_identity.py). The kernel's C
        # name carries the same mark (see `_c_kernel_name`), so the two halves agree.
        owner_mark:str = (f"{plan.owner.__module__}.{plan.owner.__qualname__}"
                       if plan.owner is not None else "")
        plan.cache_key:str = (f"{plan.backend}|{plan.device_key}|{plan.sig}"
                          f"|{self._type_map_key()}|{owner_mark}")
        plan.entry = self._cache.get(plan.cache_key)
        if plan.entry is not None and self._source_files_changed(plan.entry[8]):
            # A generated file this entry was built from changed on disk (damage the
            # digest sidecar cannot see -- it describes the python, not the generated
            # text). Drop the entry: the re-render below rewrites the files from the
            # current shape, and the build-failure rewrite covers whatever the merge
            # cannot (see _compile_kernels).
            plan.entry = None
        if plan.entry is None:
            # Create the device objects for any Image before compiling. On the driver
            # this was developed against, loading a module that contains texture or
            # surface calls makes every later texture/surface creation fail, so an
            # image bound after compilation would never get its device storage.
            self._bind_images(full_values=args, backend=plan.backend)
            plan.entry = self._compile(args, plan.arg_types, plan.self_types, plan.self_attrs, plan.owner, plan.sig,
                                  plan.backend, plan.device_obj, instance,
                      self_struct_members=plan.self_struct_members,
                      self_struct_type=plan.self_struct_type)
            self._cache[plan.cache_key] = plan.entry
        (_kernel_obj, _shape_sources, _work_size_dims, _local_sizes, _uses_global_id,
         _image_samplers, _returns, _outer_vars, _launch_meta) = plan.entry
        plan.kernel_obj = _kernel_obj
        plan.shape_sources = _shape_sources
        plan.work_size_dims = _work_size_dims
        plan.local_sizes = _local_sizes
        plan.uses_global_id = _uses_global_id
        plan.image_samplers = _image_samplers
        plan.returns = _returns
        plan.outer_vars = _outer_vars
        plan.launch_meta = _launch_meta
        # One entry per magic loop, launched in order; a single-loop body has exactly one.
