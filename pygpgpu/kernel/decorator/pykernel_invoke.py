# The launch itself: one continuous flow over the call's values, kept whole
# so its locals stay readable; the surrounding call surface is pykernel_call.py.
from __future__ import annotations
import ast
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
import numpy as np
from ...vec_types.layout import pack
from ...utils import local_size_for_global, detect_work_size, device_limits
from ..translator import (
    PTR,
    DeviceType,
    _unpack_count_message,
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
from .images import (
    _image_sampler,
)
from .slots import (
    _kernel_scalar_argument,
)
from .shape_check import (
    _check_shape_indices,
)
from .host_regions import (
    _created_arguments,
    _host_created_values,
    _host_region_values,
    _host_scalar_values,
    _positional_argument_names,
    _receiver_name,
)
from ._typing import (
    OuterState,
)


class _PykernelInvokeMixin:

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
