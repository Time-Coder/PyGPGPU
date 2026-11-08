from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
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
                phase_args.append(int(plan.full[param_index].shape[dim]))
            # The signature declares the sampler and local-memory arguments right after the
            # shape extents, and the outer/created values last (see render_kernel), so they
            # are appended in that order here. Getting this wrong shifts every later
            # argument: the local pointer would land where an outer scalar is expected.
            if plan.backend == "opencl":
                for param_index in plan.image_samplers:
                    phase_args.append(_image_sampler(plan.full[param_index]))
                for local_bytes in plan.local_sizes:
                    phase_args.append(local_bytes)
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
            return phase_args
