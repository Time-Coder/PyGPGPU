from __future__ import annotations
import ctypes
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
    receiver_passed_as_value,
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

    def _owner_class_of(self)->type:
        """The class a classmethod kernel's receiver names, resolved from its qualname.

        The kernel's own qualified name is `K.g` (or `Base.g` for the class it was
        *defined* on), and the module globals hold the class that attribute belongs to.
        A subclass reached through the descriptor (`Sub.g`) carries the same qualname,
        so a caller wanting subclass ownership passes the class explicitly -- which is
        what the descriptor-bound form does (`Sub.g.__self__` never reaches here, but
        `_invoke(Sub, ...)` through `_KernelCall`'s class form can).
        """
        qualname:str = getattr(self.raw_func, "__qualname__", "") or ""
        if "." in qualname and "<" not in qualname:
            owner_name:str = qualname.rsplit(".", 1)[0]
            owner:object = self.raw_func.__globals__.get(owner_name)
            if isinstance(owner, type):
                return owner
        # A function-local class (`<locals>` in the qualname) lives in no module
        # globals, so the qualname cannot name it. The descriptor recorded what
        # its last attribute access bound -- for a classmethod kernel the receiver
        # IS the class -- and that record is the only thing that can answer here.
        last:object = getattr(self, "_last_cls_receiver", None)
        if isinstance(last, type):
            return last
        raise TypeError(
            f"{self.__name__}() is a classmethod; call it on its class, which binds "
            f"cls automatically")

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
            # A classmethod kernel's receiver IS the class: the caller reached the launch
            # through the class attribute (python's own binding rule), so the value passed
            # as `instance` -- when one arrived -- is the class itself, not an instance of
            # it. Without one, the owner is resolved from the kernel's own qualified name
            # (`K.g` names the class `K` in the module it was defined in), which is what
            # keeps a launch through the undecorated PyKernel object working too. The
            # `cls` receiver name carries the same meaning as the `@classmethod` wrapper:
            # either spelling is how this package spells a classmethod kernel.
            is_cls_kernel:bool = self._is_classmethod or self._self_param == "cls"
            if is_cls_kernel:
                if isinstance(instance, type):
                    plan.owner = instance
                else:
                    plan.owner = self._owner_class_of()
                    instance = plan.owner
            else:
                if instance is None:
                    raise TypeError(f"{self.__name__}() requires an instance")
                plan.owner = type(instance)
            # The decorated body was scanned at decoration time, before the class
            # existed, so self.<helper> could only look like data. Scan again now
            # that the instance is here: calling a self-method makes the attributes
            # *it* reads kernel arguments too, recursively.
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
            # The receiver itself can be a VALUE the body passes on: `helper(self, ...)`
            # hands the callee the whole receiver, and the callee's own method calls
            # (`r.other(v)`) resolve the members they read from the same structure. No
            # `self.<attr>` read in THIS body is required for that -- without the
            # receiver struct the name `self` would not even resolve, which is the
            # refusal a body that only passes it on used to hit.
            if not plan.struct_attrs and self._self_param \
                    and receiver_passed_as_value(self.raw_func, self._self_param,
                                                 set(plan.host_executed)):
                plan.struct_attrs:List[str] = ["__pygpgpu_pad__"]
            plan.receiver_as_value:bool = bool(self._self_param) and \
                receiver_passed_as_value(self.raw_func, self._self_param,
                                         set(plan.host_executed))
            if plan.struct_attrs:
                from ..objects import _structure_for, member_type  # noqa: PLC0415 (import cycle)

                plan.self_struct:type = _structure_for(
                    [(name,
                      ctypes.c_uint8 if name == "__pygpgpu_pad__"
                      else member_type(getattr(instance, name), name, self._type_map))
                     for name in plan.struct_attrs], [], plan.backend)
                from ..structs import register_name_hint  # noqa: PLC0415 (import cycle)

                register_name_hint(np.dtype(plan.self_struct), plan.owner)
                # A receiver the body PASSES ON (`helper(self, ...)`) resolves the
                # methods the callee calls on it from this class: the helper's
                # `r.other(v)` looks the owner up through the dtype, so the class
                # has to be registered as the dtype's declaring one. A receiver
                # only read member-wise needs no registration -- the attribute
                # paths above answer those reads -- and registering it there
                # changed call paths that never asked for it.
                if plan.receiver_as_value:
                    from ..structs import register_declared_class  # noqa: PLC0415 (import cycle)

                    register_declared_class(np.dtype(plan.self_struct), plan.owner)
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
                    if name == "__pygpgpu_pad__":
                        # A value-only receiver: the pad keeps the zero it was
                        # declared with -- nothing reads it, and the instance has
                        # no attribute of that name to copy.
                        continue
                    plan.member_value:object = getattr(instance, name)
                    plan.field_type:object = dict(plan.self_struct._fields_).get(name)
                    if _is_complex_member_class(plan.field_type) \
                            and not isinstance(plan.member_value, plan.field_type):
                        plan.member_value = _complex_member_value(plan.member_value, plan.field_type)
                    setattr(plan.holder, name, plan.member_value)
                # A body that assigns `self.<attr> = ...` writes through the device buffer,
                # whose read-back lands in this holder; the host copies the members home
                # after the launch (see _write_back_receiver). The pairing is the record
                # that copy reads: holder members -> the receiver they came from. It is
                # kept on the plan (the synchronous tail reads it there) and on the kernel
                # (the asynchronous tail runs after the launch phases have let the plan
                # go, and `_with_tail` has no other way back to it).
                plan.receiver_write_back:object = (instance, plan.holder,
                                                   list(plan.struct_attrs))
                self._receiver_write_back = plan.receiver_write_back
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
                                             type_map=self._type_map,
                                             backend=plan.backend)
        plan.arg_types:List[DeviceType] = [infer_ctype(a, self._type_map) for a in plan.args]
        # An object ARGUMENT's ndarray members travel as their own kernel parameters,
        # appended after the converted arguments (`_<param>_<member>`, the same shape
        # the receiver's `_self_<attr>` spells): the launch passes the caller's array
        # itself, so a store through it lands in the caller's buffer. The map travels
        # to the translator, which binds `item.arr` to the parameter, and to the launch
        # packing, which appends the names so the aligned loop finds the holder.
        plan.object_array_members:Dict[str, List[Tuple[str, object]]] = dict(
            plan.object_binding.array_members)
        if plan.object_array_members:
            for param_name, members in plan.object_array_members.items():
                for member, array in members:
                    plan.args = list(plan.args) + [array]
                    plan.arg_types = list(plan.arg_types) + [
                        infer_ctype(array, self._type_map)]
        # The map reaches the translator through the compile plan (the members carry
        # their device types, which `infer_ctype` answered off the caller's arrays);
        # the translator declares each member a parameter named `_<param>_<member>`
        # and binds every `item.arr` read in the body to it.
        self._object_array_members = {
            param: [(member, "ptr") for member, _array in members]
            for param, members in plan.object_array_members.items()}
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
        # The inlined helper closure is part of the identity, exactly as the baked shapes
        # above are: a helper is monomorphized inside this kernel's own translation unit,
        # so its text is this kernel's device code. The key has to fold the closure's
        # source digest *here* -- before the entry is looked up -- because a helper edit
        # does not touch any generated file on disk until a recompile happens: the
        # per-file mtime check below only sees files the last compile wrote, so without
        # this mark the entry survives and the launch silently answers the old helper's
        # numbers. The kernel's C name folds the same digest (see _c_kernel_name).
        helpers_mark:str = self._helper_closure_digest()
        if helpers_mark:
            plan.cache_key = f"{plan.cache_key}|helpers={helpers_mark}"
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
        if plan.backend == "metal" and _local_sizes:
            # Metal allocates a `threadgroup`-space parameter per threadgroup, and the
            # encoder learns its byte size only through `setThreadgroupMemoryLength:`
            # (see BoundKernel._encode). The sizes follow the translator's
            # `local_buffers` order, which is the order the signature declares them in;
            # the kernel object is the one place both the encode half and any later
            # launch of the same kernel can read them from. `_kernel_obj` IS the
            # MetalKernel (`_backend_compile` answered it); the duck-typed probe keeps
            # a foreign object from breaking the launch.
            set_allocations = getattr(_kernel_obj, "set_threadgroup_allocations", None)
            if callable(set_allocations):
                set_allocations(_local_sizes)
        plan.uses_global_id = _uses_global_id
        plan.image_samplers = _image_samplers
        plan.returns = _returns
        plan.outer_vars = _outer_vars
        plan.launch_meta = _launch_meta
        # One entry per magic loop, launched in order; a single-loop body has exactly one.
