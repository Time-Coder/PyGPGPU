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
from ...utils import local_size_for_global, detect_work_size
from ..translator import (
    _unpack_count_message,
)


class _RunLaunchMixin:

    def _run_launch(self, plan:object, instance:object, args:Tuple[object, ...],
                    backend_sel:Optional[str], device_sel:Optional[str],
                    global_size:Optional[Tuple[int, ...]],
                    local_size:Optional[Tuple[int, ...]],
                    launch:str = "call")->object:
        """The phases themselves: align each phase's signature, launch it, and
        settle the result.

        Every phase of the body gets the same geometry -- they are passes over one
        domain -- and each is launched in turn on the stream this call owns, so the
        arrays that travel between them stay on the device instead of going back to
        the host and returning.
        """
        self._sync_device(plan.backend, plan.device_obj)
        plan.launch_args:List[object] = self._build_phase_args(plan, plan.shape_sources, None)
        # A gap statement may assign a kernel argument (`n = n + 1` between two loops).
        # A parameter is python's function-local, so a later pass must receive the updated
        # value while the caller's own variable stays untouched. `plan.full` is the packed
        # prefix and `prefix_names` is that order; a name handled as a slot (carried, gap,
        # writable outer) never reaches the parameter branch.
        plan.parameter_slots:Dict[str, int] = {
            name: index for index, name in enumerate(plan.prefix_names)}
        plan.written_parameters:Set[str] = set()
        # Vector values travel in the layout the backend's ABI defines: OpenCL pads a
        # 3-component vector to four slots, CUDA/HIP take the host's three. The host keeps
        # the three-component form everywhere, so each argument is converted here and its
        # result copied back once the launch has finished.
        # Every entry pairs the value handed to the launch with the callable that copies it
        # back into the caller's own form once the launch has finished.
        plan.unpackers:List[Tuple[object, Callable[[object], None]]] = []
        for plan.index, plan.argument in enumerate(plan.launch_args):
            _packed_value, _unpack = pack(plan.argument, plan.backend)
            plan.packed_value = _packed_value
            plan.unpack = _unpack
            if plan.packed_value is not plan.argument:
                plan.launch_args[plan.index] = plan.packed_value
            if plan.unpack is not None:
                plan.unpackers.append((plan.packed_value, plan.unpack))
        # The extents the magic loop itself iterates, in launch order (fastest axis first).
        # Derived whenever the loop states a domain, because an explicit bracket has to agree
        # with it axis by axis, not merely dimension by dimension.
        plan.loop_extents:List[int] = []
        if plan.work_size_dims:
            # A dim of -1 is an `@kernel(domain=...)` declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            plan.member_index:int = 0
            for plan.pi, dim in plan.work_size_dims:
                if plan.pi is None:
                    plan.loop_extents.append(dim)
                    continue
                if dim == -2:
                    # A host-side integer supplied as an argument, used as an extent:
                    # `np.ndindex(self.image_height, self.image_width)`.
                    plan.loop_extents.append(int(plan.full[plan.pi]))
                    continue
                if dim == -3:
                    # The extent is a receiver-struct member: full[pi] is the ctypes
                    # structure the caller handed over, and the member holds the count.
                    plan.member_name = plan.launch_meta.get("work_size_members", [])[plan.member_index]
                    plan.member_index += 1
                    plan.loop_extents.append(int(getattr(plan.full[plan.pi], plan.member_name)))
                    continue
                plan.shape:Tuple[int, ...] = tuple(int(s) for s in plan.full[plan.pi].shape)
                if dim >= 0:
                    if dim >= len(plan.shape):
                        # The loop asks for an axis the array does not have. numpy raises
                        # this as an unpacking error from the `for` itself, so the launch
                        # says it in the same words instead of letting an index slip out.
                        raise ValueError(
                            f"{_unpack_count_message(dim + 1, len(plan.shape))}: "
                            f"'{plan.launch_meta['param_names'].get(plan.pi, 'the array')}' has shape "
                            f"{plan.shape}")
                    plan.loop_extents.append(plan.shape[dim])
                else:
                    plan.loop_extents.extend(plan.shape)
        if global_size is None and plan.work_size_dims:
            # A dim of -1 is an `@kernel(domain=...)` declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            plan.extents_list:List[int] = list(plan.loop_extents)
            # A domain of more axes than a device grid has (three) is folded into one flat
            # axis of `prod(shape)` work items, which is what the original auto-detection
            # did. Two consequences follow and both are honest: a magic loop recovers the
            # extra axes by decomposing that position over its own domain (`_emit_nd_head`),
            # while a gid-style body reads the flat position directly -- `get_global_id(0)`
            # is that index and the higher axes read 0, so a body that wants the original
            # axes spelled out should iterate with `np.ndindex` instead.
            extents:Tuple[int, ...] = tuple(plan.extents_list)
            if len(extents) > 3:
                plan.total:int = 1
                for plan.extent in extents:
                    plan.total *= int(plan.extent)
                extents = (plan.total,)
            # `np.ndindex(arr.shape)` means the array's whole shape, but the translator only
            # sees a pointer and cannot know its rank, so it takes one dimension per loop
            # name. `for j, i in np.ndindex(image.shape)` over an RGB image would therefore
            # iterate two of the three axes and never vary the channel. Check it here, where
            # the rank is known.
            plan.whole:int = int(plan.launch_meta.get("whole_shape_index", -1))
            plan.loop_names:int = int(plan.launch_meta.get("loop_names", 0))
            if plan.whole >= 0 and plan.loop_names:
                plan.rank:int = len(tuple(int(s) for s in plan.full[plan.whole].shape))
                if plan.rank != plan.loop_names:
                    # numpy raises the unpacking error itself for this shape, so the device
                    # reports it in the same words: a body should not have to learn a second
                    # vocabulary for the same mistake. The array and its shape follow, since
                    # only the launch knows either.
                    raise ValueError(
                        f"{_unpack_count_message(plan.loop_names, plan.rank)}: "
                        f"'{plan.launch_meta['param_names'].get(plan.whole, 'the array')}' has shape "
                        f"{tuple(int(s) for s in plan.full[plan.whole].shape)}, and "
                        f"np.ndindex(....shape) iterates every dimension of it")
            # The loop has to name one index per dimension of its domain: `for j, i in
            # np.ndindex(image.shape)` over an RGB image would silently iterate only two of
            # the three dimensions, so the third axis never varies. Say so instead.
            if plan.loop_names and plan.loop_names != len(plan.extents_list):
                plan.source = ""
                if plan.work_size_dims[0][0] is not None:
                    plan.source = (f" (the array has shape "
                              f"{tuple(int(s) for s in plan.full[plan.work_size_dims[0][0]].shape)})")
                raise ValueError(
                    f"{_unpack_count_message(plan.loop_names, len(plan.extents_list))}: "
                    f"the loop's iteration domain has "
                    f"{len(plan.extents_list)} dimension{'s' if len(plan.extents_list) != 1 else ''}"
                    f"{plan.source}. Give one name per dimension, or iterate an explicit subset, "
                    f"e.g. np.ndindex(image.shape[0], image.shape[1])")
            if local_size is None:
                global_size, local_size = detect_work_size(
                    256, extents, max_work_group_size=plan.device_max_group,
                    max_work_item_sizes=plan.device_max_items, allow_padding=True)
            elif len(local_size) == len(extents):
                # A user-supplied local size wins; round each extent up to a multiple of it.
                global_size = tuple(-(-plan.extent // l) * l for plan.extent, l in zip(extents, local_size))
            else:
                global_size = extents
        if global_size is None and local_size is not None:
            raise ValueError(
                "local work size requires a global size: specify both in the bracket "
                "k[(global, local)] or rely on a magic loop to derive the global size")
        if global_size is None and plan.uses_global_id:
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
            plan.loop_names:int = int(plan.launch_meta.get("loop_names", 0))
            if plan.loop_names and plan.loop_extents:
                plan.needed:int = 1
                for plan.extent in plan.loop_extents:
                    plan.needed *= int(plan.extent)
                plan.supplied:int = 1
                for plan.size in global_size:
                    plan.supplied *= int(plan.size)
                if plan.supplied < plan.needed:
                    raise ValueError(
                        f"{self.__name__}(): the work size bracket {tuple(global_size)} "
                        f"launches {plan.supplied} work items, but its magic loop traverses "
                        f"{tuple(plan.loop_extents)} ({plan.needed} of them), so the rest of the domain "
                        f"would never be visited. Any launch shape is accepted as long as it "
                        f"covers {plan.needed} work items (the loop decomposes a flat work-item "
                        f"position, so even {plan.needed} alone works); a larger grid is fine, and "
                        f"omitting the bracket lets the loop derive the size itself")
            # The bracket arrived as `(blocks, threads)` and was turned into the total
            # work-item form here (see _parse_work_item), so the two axes agree in rank and
            # the total always divides by the group: the quotient *is* the block count. A
            # caller's own `local=` still has to be a size that splits the total, which is
            # the one divisibility question that remains -- and it is answered for that
            # argument, not for the bracket.
            if local_size is None:
                local_size = local_size_for_global(
                    global_size, max_work_group_size=plan.device_max_group,
                    max_work_item_sizes=plan.device_max_items)
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
            plan.blocks = tuple(g // l for g, l in zip(global_size, local_size))
            plan.phase_geometry = (plan.blocks, local_size)
        else:
            plan.phase_geometry = None
        # Every phase of the body gets the same geometry: they are passes over one domain.
        # Each is launched in turn on the stream this call owns, so the arrays that travel
        # between them stay on the device instead of going back to the host and returning.
        # The asynchronous forms launch the first kernel only -- a multi-pass body is a
        # sequence of dependent launches, and a future per pass is an interface this
        # launch does not have -- so it is refused by name rather than silently running
        # one pass of several. A single-phase body (the common case) launches whole.
        if launch in ("submit", "async"):
            if len(plan.active_kernels) > 1:
                # Dependent passes: each waits for the previous one's completion event,
                # so the chain is driven by the backend's futures, not by a waiting
                # thread (see _phase_async).
                return self._launch_phases_async(plan, args, instance, launch)
            plan.bound_async = (plan.active_kernels[0][plan.phase_geometry]
                           if plan.phase_geometry is not None else plan.active_kernels[0])
            plan.future:object = (plan.bound_async.submit(*plan.launch_args) if launch == "submit"
                             else plan.bound_async.async_call(*plan.launch_args))
            # The statements after the loop are the host tail: a body whose answer is
            # `return <expression>` after the loop has no claim word at all -- the
            # translator withheld it because no work item ever claims -- so the value
            # only exists once the tail has run. The synchronous path runs it inline
            # below; the asynchronous path runs it when the future settles, exactly
            # where its read-backs have landed. A body with no tail and nothing to
            # write back settles with no extra work.
            plan.tail_statements:List[ast.stmt] = list(
                plan.launch_meta.get("body_tail") or plan.launch_meta.get("phase_tail") or [])
            if not plan.tail_statements and not plan.outer_state["writable"] and not plan.return_state:
                return plan.future
            return self._with_tail(
                plan.future, plan.tail_statements, args, instance,
                plan.carried_slots, plan.gap_slots, plan.created_objects, plan.outer_state,
                plan.outer_writable, plan.return_state, plan.returns, plan.full, plan.created)
        for plan.position, plan.phase_kernel in enumerate(plan.active_kernels):
            if plan.position:
                # The previous pass wrote the shared slots on the device; the read-back is
                # a stream callback that may not have run yet. The gap statements read those
                # slots on the host, so they must wait for the copies to land first.
                self._sync_device(plan.backend, plan.device_obj)
                plan.regions:List[List[ast.stmt]] = list(plan.launch_meta.get("phase_gaps") or [])
                if plan.position - 1 < len(plan.regions):
                    plan.region_writable:Dict[str, np.ndarray] = {
                        plan.entry["name"]: plan.entry["buffer"] for plan.entry in plan.outer_writable}
                    self._run_host_region(plan.regions[plan.position - 1], args, instance,
                                          plan.carried_slots, plan.gap_slots,
                                          plan.gap_namespaces[plan.position - 1][0],
                                          plan.gap_namespaces[plan.position - 1][1],
                                          plan.created_objects, plan.outer_state,
                                          plan.region_writable,
                                          parameter_slots=plan.parameter_slots,
                                          parameter_values=plan.full,
                                          written_parameters=plan.written_parameters,
                                          region_values=plan.host_region_values)
            # A phase kernel declares only the shapes *it* indexes, so its argument list is
            # rebuilt from the same sources with its own extents -- a later phase commonly
            # indexes an array the first one never named.
            plan.phase_shapes:List[Tuple[int, int]] = (
                plan.kernel_shapes[plan.position] if plan.position < len(plan.kernel_shapes) else [])
            plan.bound_kernel = (plan.phase_kernel[plan.phase_geometry] if plan.phase_geometry is not None
                            else plan.phase_kernel)
            plan.phase_created_names:List[str] = (
                plan.kernel_created[plan.position] if plan.position < len(plan.kernel_created) else [])
            # A phase is described by its extents *and* by the objects it created: a pass
            # that reads an array built between the loops declares it, one that does not
            # must not receive it.
            # The fast path is only safe when the phase declares exactly the parameters the
            # first kernel does: a gap value (`factor = 10` between two loops) is one pass's
            # own parameter, and reusing the first kernel's argument list there would leave
            # every later argument one slot short.
            plan.phase_names:List[str] = (plan.phase_param_names[plan.position]
                                     if plan.position < len(plan.phase_param_names) else [])
            plan.first_names:List[str] = (plan.phase_param_names[0]
                                     if plan.phase_param_names else [])
            if (plan.phase_shapes == plan.shape_sources
                    and plan.phase_created_names == list(plan.created)
                    and plan.phase_names == plan.first_names
                    and not plan.written_parameters):
                plan.bound_kernel(*plan.launch_args)
            else:
                plan.bound_kernel(*self._build_phase_args(plan, plan.phase_shapes, plan.phase_created_names, plan.position,
                                               plan.phase_claims[plan.position]
                                               if plan.position < len(plan.phase_claims) else True))
        # The statements after the last loop are python: they run once, on the host, now that
        # every work item has finished -- the same moment python would run them.
        plan.tail_writable:Dict[str, np.ndarray] = {
            plan.entry["name"]: plan.entry["buffer"] for plan.entry in plan.outer_state["writable"]} \
            if plan.outer_state else {}
        plan.tail_result:object = self._run_host_region(
            list(plan.launch_meta.get("body_tail") or plan.launch_meta.get("phase_tail") or []),
            args, instance,
            plan.carried_slots, plan.gap_slots, created_objects=plan.created_objects,
            outer_state=plan.outer_state, writable_outer=plan.tail_writable)
        for plan.packed_value, plan.unpack in plan.unpackers:
            plan.unpack(plan.packed_value)
        if plan.return_state and int(plan.return_state["claim"][0]):
            # A work item reached an in-loop return first: python's "first return wins".
            plan.result:object = self._finish_returns(plan.return_state, plan.returns, plan.full, plan.created)
        elif plan.tail_result is not None:
            plan.result:object = plan.tail_result
        else:
            plan.result:object = (self._finish_returns(plan.return_state, plan.returns, plan.full, plan.created)
                          if plan.return_state else None)
        self._write_back_outer(plan.outer_state)
        if plan.object_binding.written:
            plan.object_binding.write_back()
        return plan.result
