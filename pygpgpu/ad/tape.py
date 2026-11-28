# `Tape`: launch recording, seeding, epoch validation, adjoint launches in reverse order.
#
# The tape records kernel launches only, never device operations: the launch is exactly
# the unit that has a device meaning, and the unit whose adjoint is a kernel. It records
# each launch's kernel and its argument bindings, plus the `(array, epoch)` of every
# tracked array -- `pygpgpu.numpy` already versions its buffers, so a caller that
# overwrites an array between the forward and the backward is refused by name instead of
# being handed a plausible wrong number. The adjoint of each recorded launch is the same
# fused kernel the forms build: the forward is recomputed inside it, with
# `drop_launch_wide` skipping the loss's own accumulation so a backward does not double
# it, and scatter adjoints go through the translator's atomic conversion.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..kernel import tape_hook
from ..kernel.translator.source_text import parse_kernel_body
from ..numpy import ndarray as gnp
from .body_scan import BodyScan, parameter_names_of
from .build import _build_grad_kernel
from .reverse import OutCotangent, SEED_COT
from . import rules as rule_table


class Tape:
    """Record the launches inside a `with` block, then run their adjoints backwards.

    Usage::

        with Tape() as tape:
            out = gnp.zeros_like(x)
            saxpy(2.0, x, y, out)
            dot_kernel(out, out, loss)

        tape.backward(loss)          # seed 1.0 into the one-element array
        print(x.grad)                # still on the device

    The tape is memory-free between the passes: the arrays keep their forward contents in
    place, and what it must check is that those contents are still the ones the forward
    used -- the `epoch` comparison at `backward`. `reset()` drops the recorded cotangent
    buffers, so a second `backward` starts clean (gradients otherwise overwrite `.grad`
    per backward call, not accumulate -- accumulate is the forms' contract, and a tape
    replay is one backward of one recorded graph).
    """

    def __enter__(self)->"Tape":
        if tape_hook.active is not None:
            raise RuntimeError("a Tape is already active: tapes do not nest -- exit the "
                               "outer one first")
        self.launches:List[Tuple[object, Tuple[object, ...], List[Tuple[object, int]]]] = []
        tape_hook.active = self
        return self

    def __exit__(self, exc_type:object, exc:object, tb:object)->None:
        tape_hook.active = None

    def reset(self)->None:
        """Drop the cotangent buffers of the last backward, keeping the recordings."""
        self._cotangents:Dict[int, object] = {}

    def record(self, kernel:object, args:Tuple[object, ...],
               kwargs:Dict[str, object])->None:
        """One launch, as the launch pipeline hands it over.

        The tape's own adjoint launches are skipped: a backward inside its own recording
        would chase itself. The python backend never reaches this hook (it answers from
        the first launch phase), so a tape records device launches, which is what its
        adjoints are.
        """
        if getattr(kernel, "_pygpgpu_adjoint", False):
            return
        epochs:List[Tuple[object, int]] = []
        for value in args:
            if isinstance(value, gnp) and bool(getattr(value, "requires_grad", False)):
                epochs.append((value, int(value._state.epoch)))
        self.launches.append((kernel, args, epochs))

    def backward(self, loss:object)->None:
        """Seed the loss array with ones and run every recorded adjoint, last first.

        Takes the array itself, not a `"return"` spelling: a launch that already ran has
        an object to point at, which is also why the tape needs no `wrt` -- the
        `requires_grad` flags already marked the inputs.
        """
        if not isinstance(loss, gnp):
            raise TypeError("tape.backward seeds the loss *array* the forward wrote")
        if not self.launches:
            raise RuntimeError("the tape recorded no launches; there is nothing to "
                               "differentiate")
        self._validate_epochs()
        if not hasattr(self, "_cotangents"):
            self.reset()
        # The seed: the cotangent of the loss is one number, one per element.
        from ..numpy import ones_like
        self._cotangents[id(loss)] = ones_like(loss)
        loss_param_of:Dict[object, Optional[str]] = {}
        for kernel, args, _epochs in self.launches:
            loss_param_of[id(kernel)] = self._param_of(kernel, args, loss)
        for kernel, args, _epochs in reversed(self.launches):
            self._backward_launch(kernel, args, loss, loss_param_of[id(kernel)])
        # The tracked inputs' cotangents land in `.grad`, on the device.
        for kernel, args, _epochs in self.launches:
            for value in args:
                if isinstance(value, gnp) and bool(getattr(value, "requires_grad", False)):
                    buffer:object = self._cotangents.get(id(value))
                    if buffer is not None:
                        value.grad = buffer

    def _validate_epochs(self)->None:
        """Refuse a backward whose arrays moved since the forward ran, naming the array."""
        for _kernel, _args, epochs in self.launches:
            for array, epoch in epochs:
                if int(array._state.epoch) != epoch:
                    raise RuntimeError(
                        f"the array `{array._arg_name or '<unnamed>'}` was written "
                        f"between the forward and the backward (epoch {epoch} -> "
                        f"{int(array._state.epoch)}): the adjoint would read bytes the "
                        f"forward did not use, which is a wrong number with no crash -- "
                        f"re-run the forward, or copy the array")

    def _param_of(self, kernel:object, args:Tuple[object, ...],
                  target:object)->Optional[str]:
        """The parameter name an argument value arrived as, or None."""
        names:List[str] = parameter_names_of(kernel.raw_func)
        for name_, value in zip(names, args):
            if value is target:
                return name_
        return None

    def _backward_launch(self, kernel:object, args:Tuple[object, ...],
                         loss:object, loss_name:Optional[str])->None:
        """One recorded launch's adjoint: cotangent buffers in, gradient buffers out."""
        scan:BodyScan = self._scan_of(kernel)
        array_params:List[Tuple[str, object]] = _array_params(kernel, args)
        if not array_params and loss_name is None:
            return          # the launch touches no arrays: nothing to differentiate
        outputs:List[str] = [name_ for name_ in scan.store_targets()
                             if name_ in dict(array_params)]
        if len(outputs) > 1:
            raise RuntimeError(
                f"{kernel.__name__} writes {len(outputs)} array parameters in one "
                f"launch; a tape's adjoint carries one cotangent per output in v1 -- "
                f"split the launch")
        out_name:Optional[str] = outputs[0] if outputs else None
        if out_name is None and loss_name is None:
            return          # no array result the loss can reach through
        wrt:List[str] = [name_ for name_, _value in array_params]
        # Every array input gets a cotangent buffer: the ones the loss reaches will
        # accumulate, the rest stay zero -- which is also what a later launch's adjoint
        # reads, because the walk is in reverse order.
        buffers:Dict[str, object] = {}
        for name_, value in array_params:
            buffers[name_] = self._cotangent_of(value)
        adjoint:object = self._adjoint_of(
            kernel, wrt, out_name, loss_name,
            key=(f"tape:{kernel.__name__}", tuple(wrt), out_name, loss_name,
                 rule_table.RULES_VERSION))
        call_args:List[object] = list(args)
        if kernel._is_method:
            pass            # the recorded args already exclude the receiver
        call_args.extend(buffers[name_] for name_ in wrt)
        if out_name is not None:
            call_args.append(self._cotangents[id(_value_of(array_params, out_name))])
        elif loss_name is not None:
            call_args.append(self._cotangents[id(loss)])
        adjoint(*call_args)

    def _cotangent_of(self, value:object)->object:
        """The cotangent buffer of one array object, allocated as zeros on first need."""
        buffer:Optional[object] = self._cotangents.get(id(value))
        if buffer is None:
            from ..numpy import zeros_like
            buffer = zeros_like(value)
            self._cotangents[id(value)] = buffer
        return buffer

    def _adjoint_of(self, kernel:object, wrt:List[str], out_name:Optional[str],
                    loss_name:Optional[str], key:Tuple[object, ...])->object:
        """The fused adjoint kernel for one recorded launch, cached like the forms'."""
        cache:Dict[Tuple[object, ...], object] = getattr(kernel, "_ad_forms", None)
        if cache is None:
            cache = {}
            kernel._ad_forms = cache
        if key not in cache:
            cot_param:Optional[str] = "_c_out" if out_name is not None \
                else ("_c_loss" if loss_name is not None else None)
            cache[key] = _build_grad_kernel(
                kernel, wrt, loss_name, SEED_COT,
                {name_: f"_g_{name_}" for name_ in wrt},
                cot_param, None,
                OutCotangent(out_name, cot_param) if out_name is not None else None,
                allow_scatter=True, drop_launch_wide=True, suffix="_tape")
        return cache[key]

    def _scan_of(self, kernel:object)->BodyScan:
        scans:Dict[int, BodyScan] = getattr(self, "_scans", None) or {}
        if id(kernel) not in scans:
            origin:object = kernel.raw_func
            module:object = parse_kernel_body(origin)
            tree:object = module.body[0]
            scans[id(kernel)] = BodyScan(tree, parameter_names_of(origin),
                                         receiver_param=getattr(kernel, "_self_param",
                                                                None) or None)
            self._scans = scans
        return scans[id(kernel)]


def _array_params(kernel:object, args:Tuple[object, ...])->List[Tuple[str, object]]:
    """The launch's array parameters, in order, receiver excluded."""
    names:List[str] = parameter_names_of(kernel.raw_func)
    found:List[Tuple[str, object]] = []
    for name_, value in zip(names, args):
        if isinstance(value, np.ndarray):
            found.append((name_, value))
    return found


def _value_of(array_params:List[Tuple[str, object]], name_:str)->object:
    for param_name, value in array_params:
        if param_name == name_:
            return value
    raise KeyError(name_)
