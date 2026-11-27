from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from .. import rng as random_support
from ..translator import (
    infer_ctype,
)
from .images import (
    _image_sampler,
)
from .slots import (
    _kernel_scalar_argument,
)


class _BuildPhaseArgsMixin:

    @staticmethod
    def _phase_metal_flags(plan:object, phase_index:int)->Tuple[bool, bool]:
        """`(has_barrier, has_atomic64)` for the kernel at `phase_index`.

        A single-loop body records a plain flag per key; a body split into passes
        records one entry per pass, because each pass is its own signature and the
        two may differ. Reading the per-pass list as a scalar (or the scalar as a
        list) is what silently bound the wrong arguments, so the shape is decided
        here, once.
        """
        def pick(key:str)->bool:
            value:object = getattr(plan, key, None)
            if isinstance(value, list):
                return bool(value[phase_index]) if phase_index < len(value) else False
            return bool(value)

        return pick("has_barrier"), pick("has_atomic64")

    @staticmethod
    def _phase_local_meta(plan:object, phase_index:int)->Tuple[List[int], List[np.dtype]]:
        """The shared-buffer element counts and dtypes for the kernel at `phase_index`."""
        counts:object = getattr(plan, "local_element_counts", None) or []
        dtypes:object = getattr(plan, "local_dtypes", None) or []
        if counts and isinstance(counts[0], list):
            index:int = min(phase_index, len(counts) - 1)
            count_list:List[int] = list(counts[index])
            dtype_list:List[np.dtype] = (list(dtypes[index])
                                         if index < len(dtypes) else [])
            return count_list, dtype_list
        return list(counts), list(dtypes)

    @staticmethod
    def _resize_metal_shared(plan: object, launch_args: List[object]) -> None:
        """Re-size the Metal shared buffers the group count `plan.blocks` now carries.

        `_build_phase_args` runs before the geometry pass, so the allocations it
        made carry the one-group default; this replaces them in place (the argument
        positions are fixed, so no later value may shift) with the real sizes: one
        `groups * elements` array per shared buffer, one `groups` slot per barrier
        counter. The buffers are the ones the build recorded on
        `self._metal_shared_buffers` (shared buffers first, the counter last), and
        a launch whose geometry was known up front already carries the right sizes.
        """
        blocks: Optional[tuple] = getattr(plan, "blocks", None)
        if not blocks:
            return
        groups: int = int(blocks[0])
        buffers: List[np.ndarray] = list(getattr(plan, "_metal_shared_buffers", None) or [])
        if not buffers:
            return
        counts: List[int] = list(getattr(plan, "local_element_counts", None) or [])
        dtypes: List[np.dtype] = list(getattr(plan, "local_dtypes", None) or [])
        for buffer, count, dtype in zip(buffers, counts, dtypes):
            expected: int = groups * int(count)
            if buffer.shape[0] != expected:
                buffer.resize(expected, refcheck=False)
                buffer.fill(0)
        counter: Optional[np.ndarray] = (buffers[len(counts)]
                                         if len(buffers) > len(counts) else None)
        if counter is not None and counter.shape[0] != groups:
            counter.resize(groups, refcheck=False)
            counter.fill(0)
        locks: Optional[np.ndarray] = (buffers[len(counts) + 1]
                                       if len(buffers) > len(counts) + 1 else None)
        if locks is not None and locks.shape[0] != groups:
            locks.resize(groups, refcheck=False)
            locks.fill(0)

    def _build_phase_args(self, plan:object,
                             extents:List[Tuple[int, int]],
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
            prefix_values:List[object] = list(plan.full[:plan.own_count])
            by_name:Dict[str, object] = dict(zip(plan.prefix_names, prefix_values))
            # An array no host region touches stays on the device between passes: bound
            # through the deferred-read-back view, its bytes are not copied home and back
            # for a pass that only reads them on the GPU (see _phase_readback).
            deferred:Dict[str, object] = getattr(plan, "deferred_arrays", None) or {}
            for name, view in deferred.items():
                if name in by_name:
                    by_name[name] = view
                    prefix_values[plan.prefix_names.index(name)] = view
            # Only the names declared *before* the shape extents participate in the aligned
            # prefix: outer-scope values and host-created objects are declared after them and
            # the blocks below append those, so including them here would shift the extents.
            declared:List[str] = [name for name in (
                plan.phase_param_names[phase_index]
                if phase_index < len(plan.phase_param_names) else [])
                if name not in plan.outer_vars and name not in plan.created
                and name not in plan.host_value_names]
            if declared:
                # Align the values to this phase's own signature: a gap value only one pass
                # reads, or a carried scalar this pass writes, is not in the shared prefix.
                # Outer-scope names are excluded here -- the signature declares them last,
                # and the block below appends them in the host's own order.
                phase_args:List[object] = [by_name.get(name, plan.gap_slots.get(name, 0))
                                        for name in declared]
            else:
                # No per-phase metadata (a gid-style body): the shared prefix *is* the
                # parameter list, in order.
                phase_args:List[object] = prefix_values
            for param_index, dim in extents:
                if dim >= 0:
                    phase_args.append(int(plan.full[param_index].shape[dim]))
                else:
                    # A negative dim is a sub-array extent (see translator `_tail_var`):
                    # the product of the axes a partial store's index leaves. A full-rank
                    # index leaves none, so the empty product binds 1.
                    tail:int = 1
                    for extent in tuple(plan.full[param_index].shape[-dim - 2:]):
                        tail *= int(extent)
                    phase_args.append(tail)
            # The signature declares the sampler and local-memory arguments right after the
            # shape extents, and the outer/created values last (see render_kernel), so they
            # are appended in that order here. Getting this wrong shifts every later
            # argument: the local pointer would land where an outer scalar is expected.
            if plan.backend == "opencl":
                for param_index in plan.image_samplers:
                    phase_args.append(_image_sampler(plan.full[param_index]))
                for local_bytes in plan.local_sizes:
                    phase_args.append(local_bytes)
            elif plan.backend == "metal":
                # Metal's shared buffers travel as one device allocation covering
                # every group (`groups * total` elements, see `metal_codegen.barrier`
                # for why threadgroup memory is not used): the launch allocates it
                # here, zeroed, and the body indexes its own group's slice. The spin
                # barrier's counter rides beside it when the body called `barrier()`,
                # appended AFTER the outer-scope values below -- the same position
                # `parameter_declarations` declares it in. `plan.blocks` is known
                # only after the geometry pass, so `_resize_metal_shared` re-sizes
                # these buffers once it is.
                #
                # The list is the *signature's*: a body split into passes declares its
                # shared buffers on the pass that uses them, not on the main translator.
                blocks:Optional[tuple] = getattr(plan, "blocks", None)
                groups:int = int(blocks[0]) if blocks else 1
                local_counts, local_dtypes = self._phase_local_meta(plan, phase_index)
                plan._metal_shared_buffers:List[np.ndarray] = []
                for count, dtype in zip(local_counts, local_dtypes):
                    buffer:np.ndarray = np.zeros(int(groups) * int(count), dtype=dtype)
                    plan._metal_shared_buffers.append(buffer)
                    phase_args.append(buffer)
            phase_args.extend(plan.outer_state["full"])
            for name, value in plan.created.items():
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
            for name in plan.host_value_names:
                if name in plan.created or name in plan.outer_vars or name in declared:
                    continue
                if name not in plan.host_region_values:
                    continue
                value = plan.host_region_values[name]
                phase_args.append(_kernel_scalar_argument(value, infer_ctype(value)))
            if plan.return_state:
                phase_args.extend(plan.return_state["buffers"])
                # A pass that never claims -- one whose only return runs in its host tail --
                # does not declare the words, so it must not receive them either. `claims`
                # is None only for the single-phase fast path, whose one kernel is the main
                # translator's: its own signature is what `main_claims` records.
                if claims if claims is not None else plan.main_claims:
                    phase_args.append(plan.return_state["claim"])
                    phase_args.append(plan.return_state["site"])
            # The exception slot and stop flag come after the return slots, the same order
            # `render_exception_params` declares them in.
            exception_state = getattr(plan, "exception_state", None) or {}
            if exception_state:
                phase_args.append(exception_state["site"])
                phase_args.append(exception_state["stop"])
            # The spin barrier's counter is the very last argument: the signature
            # declares it after the outer parameters AND after the return/exception
            # slots (see `parameter_declarations`'s append order), so the launch
            # appends it here, at the end, with one zeroed slot per group. `plan.blocks`
            # is known only after the geometry pass, so `_resize_metal_shared`
            # re-sizes both it and the shared buffers once it is.
            #
            # Both flags are read for *this* kernel: a body split into passes gives the
            # barrier to one pass and not necessarily the next.
            needs_barrier, needs_atomic64 = self._phase_metal_flags(plan, phase_index)
            if plan.backend == "metal" and needs_barrier:
                blocks:Optional[tuple] = getattr(plan, "blocks", None)
                groups:int = int(blocks[0]) if blocks else 1
                counter:np.ndarray = np.zeros(int(groups), dtype=np.uint32)
                plan._metal_shared_buffers.append(counter)
                phase_args.append(counter)
            if plan.backend == "metal" and needs_atomic64:
                # One lock word per group for the 64-bit RMW spin (see
                # `atomics._emit_metal_cas_loop`), the very last argument.
                blocks:Optional[tuple] = getattr(plan, "blocks", None)
                groups:int = int(blocks[0]) if blocks else 1
                locks:np.ndarray = np.zeros(int(groups), dtype=np.uint32)
                plan._metal_shared_buffers.append(locks)
                phase_args.append(locks)
            # The launch seed, for a kernel that draws random numbers: `render` declares it
            # as the last host-bound parameter (after the barrier counter and the atomic
            # lock, before Metal's prologue parameters, which the dispatch fills in), so the
            # launch appends its value here, last.
            #
            # The seed is the *pass's* seed: a body split into passes launches one kernel
            # per pass, and two passes drawing from one stream would answer the same
            # numbers. The pass index is mixed into the launch's base seed, so the passes
            # are separate streams and the whole launch still replays from one seed.
            if self._phase_random(plan, phase_index):
                seed: int = random_support.phase_seed(plan.random_seed, phase_index)
                phase_args.append(np.uint32(seed))
            return phase_args


    @staticmethod
    def _phase_random(plan:object, phase_index:int)->bool:
        """Whether the kernel at `phase_index` draws random numbers.

        A body split into passes gives each pass its own signature, and only the passes
        that draw declare the launch seed; a single-phase body records a plain flag. The
        shape is read the same way `_phase_metal_flags` reads its two.
        """
        value:object = getattr(plan, "uses_random", None)
        if isinstance(value, list):
            return bool(value[phase_index]) if phase_index < len(value) else False
        return bool(value)
