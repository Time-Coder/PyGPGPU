# The static multi-pass adjoint: forward passes in order, then adjoint passes in reverse.
#
# A body with several magic loops launches as several kernels (the phase machinery); the
# fused single-kernel adjoint cannot see the later passes, so this module builds one fused
# adjoint kernel PER PASS and launches them in reverse order. The connection between the
# passes is cotangent buffers, not stored traces: each pass's adjoint recomputes its own
# forward from its own inputs -- so nothing needs the forward's intermediate bytes, and
# the only state that flows backwards is the cotangent of each intermediate array, which
# the later pass's adjoint accumulates into before the earlier pass's adjoint reads it.
#
# The supported shape is deliberately narrow, and everything outside it is refused with
# the reason: the passes must be consecutive magic loops (no host statements between
# them -- a gap's adjoint would be numpy AD, which this package does not do), the
# intermediates must be caller-passed parameters (a body-created intermediate would be
# re-created as fresh zeros by the adjoint's host region, not the forward's bytes), and
# each pass must write exactly one array whose cotangent flows (the loss accumulation is
# not such an array -- it is the seed sink, and any pass may hold it). The carried-scalar
# `loss="return"` spelling is refused: the per-phase replays would re-accumulate into the
# carried slot and double the value the forward computed.
from __future__ import annotations

import ast
import copy
from typing import Dict, List, Optional, Tuple

from .adjoint import COT_PARAM, GRAD_PARAM, AdjointSpec, assemble
from .ast_support import NameAllocator
from .body_scan import BodyScan, parameter_names_of
from .build import (
    _gradient_buffers,
    _origin_tree,
    _scan_of,
    _zeros_like,
)
from .errors import ad_error
from .reverse import GradBuilder, OutCotangent, SEED_COT, SEED_ONE
from . import rules as rule_table


def _error(kernel:object, message:str, node:Optional[ast.AST])->SyntaxError:
    from ..kernel.translator.source_text import _source_first_line, kernel_source_text
    return ad_error(message, node,
                    getattr(kernel.raw_func.__code__, "co_filename", "<kernel>"),
                    kernel_source_text(kernel.raw_func),
                    _source_first_line(kernel.raw_func))


def _creates_array(stmt:ast.stmt)->bool:
    """True for a binding whose right-hand side builds an array (`np.zeros(...)`)."""
    value:object = getattr(stmt, "value", None)
    if not isinstance(value, ast.Call):
        return False
    func:ast.expr = value.func
    if isinstance(func, ast.Attribute):
        return func.attr in ("zeros", "zeros_like", "ones", "ones_like", "empty",
                             "empty_like", "full", "full_like", "arange")
    return False


class MultiPassPlan:
    """One multi-pass configuration: the per-phase adjoint kernels and the launch order.

    Built once per configuration and cached on the parent kernel, like a form's single
    adjoint. `launch` runs the plain forward once (the caller's result and the loss value
    come from it), then the per-phase adjoints, last phase first; the gradient buffers of
    the tracked arrays accumulate across every phase that reads them, and the intermediate
    cotangent buffers are private to the plan.
    """

    def __init__(self, kernel:object, wrt:List[str], loss_name:Optional[str],
                 seed_mode:str, cot_of_out:Optional[OutCotangent]):
        scan:BodyScan = _scan_of(kernel)
        if seed_mode == SEED_ONE and (loss_name is None
                                      or loss_name not in scan.param_set):
            # The carried-scalar `loss="return"` spelling: the per-phase forward replays
            # would re-accumulate into the carried slot and double the value the forward
            # computed. The loss-array spelling carries no such slot.
            node:object = _origin_tree(kernel.raw_func)
            raise _error(kernel,
                         "the carried-scalar `loss=\"return\"` spelling is refused for a "
                         "multi-pass body: the per-phase forward replays would "
                         "re-accumulate into the carried slot and double the value the "
                         "forward computed -- accumulate the loss into a shape-(1,) "
                         "array through `loss=` instead", node)
        self.kernel:object = kernel
        self.wrt:List[str] = list(wrt)
        self.loss_name:Optional[str] = loss_name
        self.seed_mode:str = seed_mode
        self.cot_of_out:Optional[OutCotangent] = cot_of_out
        self.scan:BodyScan = scan
        self._refuse_unsupported()
        # A body-created array (`scratch = np.zeros(...)` above a loop) is the one shape
        # the per-phase adjoints cannot carry: the later pass's adjoint would see fresh
        # zeros where the forward wrote bytes, so it is refused before the per-pass
        # write analysis reports a confusing "writes 0 arrays".
        for name, stores in self.scan.body_created_stores().items():
            raise _error(self.kernel,
                         f"`{name}` is created inside the body: the adjoint of the "
                         f"later pass needs the forward's bytes, which only a "
                         f"caller-passed parameter carries -- pass it as an argument",
                         stores[0])
        self.phases:List[ast.For] = list(self.scan.dispatches)
        # Per pass: the synthetic single-pass scan, the one array whose cotangent flows,
        # and the array parameters the pass reads.
        self._phase_info:List[Tuple[BodyScan, str, List[str]]] = []
        for loop in self.phases:
            phase_scan:BodyScan = self._phase_scan(loop)
            array_writes:List[str] = [name for name in phase_scan.store_targets()
                                      if not phase_scan.is_member_root(name)
                                      and name != self.loss_name]
            if len(array_writes) > 1:
                raise _error(kernel,
                             f"this pass writes {len(array_writes)} array parameters "
                             f"besides the loss; the multi-pass adjoint carries one "
                             f"cotangent per pass -- split the pass", loop)
            reads:List[str] = sorted(name for name in phase_scan.value_reads()
                                     if not phase_scan.is_member_root(name))
            out_name:str = array_writes[0] if array_writes                 else (self.loss_name or "")
            if not out_name:
                raise _error(kernel,
                             "this pass writes no array and no loss: its adjoint has "
                             "nothing to distribute -- drop the pass", loop)
            self._phase_info.append((phase_scan, out_name, reads))
        # The intermediates: written by one pass and read by a later one. Their cotangent
        # buffers are the connection between the adjoints.
        self.intermediates:List[str] = []
        for index in range(len(self._phase_info) - 1):
            written:str = self._phase_info[index][1]
            if any(written in reads for _scan, _w, reads in self._phase_info[index + 1:]):
                self.intermediates.append(written)
        for name in self.intermediates:
            if name not in self.scan.param_set:
                raise _error(kernel,
                             f"the intermediate `{name}` is created inside the body: the "
                             f"adjoint of the later pass needs the forward's bytes, which "
                             f"only a caller-passed parameter carries -- pass it as an "
                             f"argument", self.phases[0])
        if self.cot_of_out is not None \
                and self.cot_of_out.out_param != self._phase_info[-1][1]:
            raise _error(kernel,
                         f"the output `{self.cot_of_out.out_param}` is not written by the "
                         f"last pass, so its cotangent cannot enter there -- reorder the "
                         f"passes", self.phases[-1])

    def _first_line_node(self, kernel:object)->Optional[ast.AST]:
        tree:ast.FunctionDef = _origin_tree(kernel.raw_func)
        return tree

    def _refuse_unsupported(self)->None:
        """The shape checks made before anything is built."""
        body:List[ast.stmt] = list(self.scan.tree.body)
        loops:List[int] = [index for index, stmt in enumerate(body)
                           if any(stmt is dispatch for dispatch in self.scan.dispatches)]
        first:int = loops[0]
        last:int = loops[-1]
        if last - first + 1 != len(loops):
            raise _error(self.kernel,
                         "this multi-pass body has statements between its magic loops: a "
                         "gap runs on the host, and the host has no AD in v1 -- move the "
                         "statement into a pass, or pass its value as a parameter",
                         body[first + 1])
        for index in range(first + 1, last):
            if _creates_array(body[index]):
                raise _error(self.kernel,
                             "this multi-pass body creates an array between its passes: "
                             "the adjoint of the later pass would re-create it as fresh "
                             "zeros instead of the forward's bytes -- pass the "
                             "intermediate as a parameter to the kernel", body[index])

    def _phase_scan(self, loop:ast.For)->BodyScan:
        """The scan of a synthetic single-pass kernel: this pass's loop alone."""
        tree:ast.FunctionDef = _origin_tree(self.kernel.raw_func)
        tree.body = [copy.deepcopy(loop)]
        return _scan_of(self.kernel, tree)

    # ------------------------------------------------------------------ launching

    def launch(self, form:object, arguments:List[Tuple[str, object]],
               weight:Optional[object], forward:bool,
               raw_args:Tuple[object, ...], raw_kwargs:Dict[str, object])->object:
        """The plain forward, then the per-phase adjoints, last phase first.

        `forward` is False when the caller already ran the plain kernel this call (the
        vjp spelling), True when this launch is the only forward (grad).
        """
        values:Dict[str, object] = dict(arguments)
        wrt_set:set = set(self.wrt)
        if forward:
            form._forward(raw_args, raw_kwargs)
        # The cotangent buffer per array object: the intermediates get zeros, the loss
        # array gets ones (the seed), and the tracked arrays' buffers are their `.grad`.
        cotangents:Dict[str, object] = {name: _zeros_like(values[name])
                                        for name in self.intermediates}
        if self.loss_name is not None and self.seed_mode == SEED_COT:
            from ..numpy import ones_like
            cotangents[self.loss_name] = ones_like(values[self.loss_name])
        gradients:Dict[str, object] = _gradient_buffers(form, arguments, self.wrt,
                                                        mode="reverse")
        for index in range(len(self.phases) - 1, -1, -1):
            _scan, out_name, reads = self._phase_info[index]
            is_last:bool = index == len(self.phases) - 1
            # Every array this pass reads whose cotangent exists is a distribution
            # target: the tracked arrays (the requested gradients) and the intermediates.
            phase_wrt:List[str] = [name for name in reads
                                   if name in wrt_set or name in self.intermediates]
            adjoint:object = self._adjoint_of(index, phase_wrt, out_name)
            extra:List[object] = [gradients[name] if name in wrt_set
                                  else cotangents[name] for name in phase_wrt]
            if is_last and self.seed_mode == SEED_COT and self.cot_of_out is not None:
                extra.append(weight)
            elif out_name in cotangents:
                # The pass's own output cotangent (an intermediate's), which the later
                # pass's adjoint accumulated into before this launch.
                extra.append(cotangents[out_name])
            call_args:List[object] = _positional_of(form, arguments)
            call_args.extend(extra)
            bracket:object = adjoint[form.backend] if form.backend else adjoint
            bracket(*call_args)
        return gradients

    def _adjoint_of(self, index:int, phase_wrt:List[str], out_name:str)->object:
        """The fused adjoint kernel of one pass, cached like a form's own."""
        key:Tuple[object, ...] = ("multi-pass", index, tuple(phase_wrt), out_name,
                                  self.loss_name, self.seed_mode, self.cot_of_out,
                                  rule_table.RULES_VERSION,
                                  tuple(sorted(self.kernel._type_map.items())))
        cache:Dict[Tuple[object, ...], object] = getattr(self.kernel, "_ad_forms", None)
        if cache is None:
            cache = {}
            self.kernel._ad_forms = cache
        if key not in cache:
            cache[key] = self._build(index, phase_wrt, out_name)
        return cache[key]

    def _build(self, index:int, phase_wrt:List[str], out_name:str)->object:
        origin:object = self.kernel.raw_func
        # The builder's scan and tree must be one object pair: `build()` splits the body
        # by `stmt is self.scan.dispatch`, and a scan built over a different deepcopy
        # would never match, dropping the loop into the host region.
        tree:ast.FunctionDef = _origin_tree(origin)
        tree.body = [copy.deepcopy(self.phases[index])]
        phase_scan:BodyScan = BodyScan(tree, parameter_names_of(origin),
                                       receiver_param=getattr(self.kernel,
                                                              "_self_param", None)
                                       or None)
        allocator:NameAllocator = NameAllocator(set(phase_scan.all_names)
                                                | set(rule_table.INJECTS))
        gradient_of:Dict[str, str] = {name_: allocator.fresh(f"_g_{name_}_{index}")
                                      for name_ in phase_wrt}
        is_last:bool = index == len(self.phases) - 1
        cot_param:Optional[str] = None
        phase_cot:Optional[OutCotangent] = None
        if is_last and self.seed_mode == SEED_COT and self.cot_of_out is not None:
            cot_param = allocator.fresh("_c_out")
            phase_cot = OutCotangent(out_name, cot_param)
        elif out_name in self.intermediates:
            cot_param = allocator.fresh(f"_c_{out_name}")
            phase_cot = OutCotangent(out_name, cot_param)
        builder:GradBuilder = GradBuilder(
            phase_scan, phase_wrt, gradient_of, self.loss_name, self.seed_mode,
            phase_cot, cot_param, None, allocator, origin,
            _error_filename(origin), _error_text(origin), _error_line(origin),
            allow_scatter=True, drop_launch_wide=True)
        builder.build(tree)
        extra:List[Tuple[str, str]] = [(gradient_of[name_], GRAD_PARAM)
                                       for name_ in phase_wrt]
        if cot_param:
            extra.append((cot_param, COT_PARAM))
        spec:AdjointSpec = AdjointSpec(tree, extra,
                                       f"{self.kernel.__name__}_pass{index}_adjoint",
                                       builder.helper_injects)
        return assemble(spec, self.kernel, allocator)


def _positional_of(form:object, arguments:List[Tuple[str, object]])->List[object]:
    """The argument values in the kernel's own parameter order, receiver prepended."""
    call_args:List[object] = [value for _name, value in arguments]
    if form.instance is not None:
        call_args = [form.instance] + call_args
    return call_args


def _error_filename(origin:object)->str:
    code:object = getattr(origin, "__code__", None)
    return getattr(code, "co_filename", "<kernel>") or "<kernel>"


def _error_text(origin:object)->str:
    from ..kernel.translator.source_text import kernel_source_text
    return kernel_source_text(origin)


def _error_line(origin:object)->int:
    from ..kernel.translator.source_text import _source_first_line
    return _source_first_line(origin)
