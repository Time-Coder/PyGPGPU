# The shared half of the forms: running the transforms and assembling the adjoint kernels.
#
# Split from the form objects by subject, per the file-size policy's mixin rule: the forms
# decide the configuration and the call shapes (`forms.py`); this module decides what a
# configured form builds -- the tree copies, the scan, the buffers, the trackability
# refusals and the loss resolution both kinds of form read.
from __future__ import annotations

import ast
import copy

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from ..kernel.translator.source_text import (
    _source_first_line,
    kernel_source_text,
    parse_kernel_body,
)
from ..numpy import ndarray as gnp
from .adjoint import AdjointSpec, COT_PARAM, GRAD_PARAM, TANGENT_PARAM, assemble
from .ast_support import NameAllocator
from .body_scan import BodyScan, parameter_names_of
from .forward import JVPBuilder
from .reverse import GradBuilder, OutCotangent
from .rules import INJECTS

if TYPE_CHECKING:
    # The form object the helpers report errors against: importing it at module level
    # would close the forms <-> build cycle, and only the annotations need the name.
    from .forms import _ADForm

# The one spelling `loss=`/`out=` accept for "the value the body returned"; no parameter
# can shadow it, because parameters are user-named and python refuses a parameter spelled
# `return` in the first place.
RETURN_SPELL = "return"

# ------------------------------------------------------------------ shared building

def _build_grad_kernel(kernel:object, wrt:List[str], loss_name:Optional[str],
                       seed_mode:str, gradient_of:Dict[str, str],
                       cot_param:Optional[str], ret_cot_param:Optional[str],
                       cot_of_out:Optional[OutCotangent], allow_scatter:bool,
                       drop_launch_wide:bool, suffix:str)->object:
    """Run the reverse transform and assemble the adjoint kernel.

    The proposed parameter names are allocated through the body's own namespace, so a
    user local called `_g_x` cannot collide with the gradient buffer's parameter; the
    buffers travel positionally in `wrt` order either way.
    """
    origin:object = kernel.raw_func
    tree:ast.FunctionDef = _origin_tree(origin)
    scan:BodyScan = _scan_of(kernel, tree)
    allocator:NameAllocator = NameAllocator(set(scan.all_names) | set(INJECTS))
    gradient_of = {name_: allocator.fresh(proposed)
                   for name_, proposed in gradient_of.items()}
    if cot_param:
        cot_param = allocator.fresh(cot_param)
    if ret_cot_param:
        ret_cot_param = allocator.fresh(ret_cot_param)
    if cot_of_out is not None:
        cot_of_out = OutCotangent(cot_of_out.out_param, cot_param or "")
    builder:GradBuilder = GradBuilder(
        scan, wrt, gradient_of, loss_name, seed_mode, cot_of_out, cot_param,
        ret_cot_param, allocator, origin,
        _filename_of(origin), kernel_source_text(origin), _source_first_line(origin),
        allow_scatter=allow_scatter, drop_launch_wide=drop_launch_wide)
    builder.build(tree)
    extra:List[Tuple[str, str]] = [(gradient_of[wrt_name], GRAD_PARAM)
                                   for wrt_name in wrt]
    if cot_param:
        extra.append((cot_param, COT_PARAM))
    if ret_cot_param:
        extra.append((ret_cot_param, COT_PARAM))
    spec:AdjointSpec = AdjointSpec(tree, extra, f"{kernel.__name__}{suffix}",
                                   builder.helper_injects)
    return assemble(spec, kernel, allocator)


def _build_jvp_kernel(kernel:object, wrt:List[str], tangents:Dict[str, str],
                      out_name:Optional[str], out_tangent:Optional[str])->object:
    """Run the forward transform and assemble the JVP kernel."""
    origin:object = kernel.raw_func
    tree:ast.FunctionDef = _origin_tree(origin)
    scan:BodyScan = _scan_of(kernel, tree)
    allocator:NameAllocator = NameAllocator(set(scan.all_names) | set(INJECTS))
    tangents = {name_: allocator.fresh(proposed)
                for name_, proposed in tangents.items()}
    if out_tangent:
        out_tangent = allocator.fresh(out_tangent)
    builder:JVPBuilder = JVPBuilder(
        scan, wrt, tangents, out_name, out_tangent, allocator, origin,
        _filename_of(origin), kernel_source_text(origin), _source_first_line(origin))
    builder.build(tree)
    extra:List[Tuple[str, str]] = [(tangents[tracked], TANGENT_PARAM)
                                   for tracked in wrt]
    if out_tangent:
        extra.append((out_tangent, TANGENT_PARAM))
    spec:AdjointSpec = AdjointSpec(tree, extra, f"{kernel.__name__}_jvp")
    return assemble(spec, kernel, allocator)


def _scan_of(kernel:object, tree:Optional[ast.FunctionDef] = None)->BodyScan:
    """The body scan of a kernel, over its own tree."""
    origin:object = kernel.raw_func
    if tree is None:
        tree = _origin_tree(origin)
    return BodyScan(tree, parameter_names_of(origin),
                    receiver_param=getattr(kernel, "_self_param", None) or None)


def _check_tracked_members(form:"_ADForm",
                           arguments:List[Tuple[str, object]])->None:
    """Refuse a `requires_grad` flag on a receiver or object member, at the call.

    The `wrt` inference reads the call's ARGUMENTS, so a flag on `self.points` or
    `obj.field` would be silently ignored and the member's gradient would come back as
    zero -- the worst failure mode available here. The check runs even for an explicit
    `wrt=`, because the flag says the caller expected the member to be tracked.
    """
    for name, value in arguments:
        if isinstance(value, np.ndarray) or isinstance(value, type):
            # A plain array is the tracked thing itself; a class carries its members as
            # compile-time constants, which no flag can make differentiable.
            continue
        attrs:Dict[str, object] = vars(value) if hasattr(value, "__dict__") else {}
        for attr, member in attrs.items():
            if isinstance(member, gnp) and bool(getattr(member, "requires_grad", False)):
                raise form._error(
                    f"`{name}.{attr}` carries `requires_grad`, and receiver or object "
                    f"members are not differentiable in the fused form: the flag would "
                    f"be silently ignored and the gradient would come back zero -- pass "
                    f"the member as its own array argument "
                    f"(`kernel(value.{attr}, ...)`)")


def _gradient_buffers(form:"_ADForm", arguments:List[Tuple[str, object]],
                      wrt:List[str], mode:str)->Dict[str, object]:
    """One gradient buffer per tracked parameter, accumulated into the array's own `.grad`.

    `.grad` accumulates (PyTorch semantics): the first backward allocates zeros, later
    ones add, and `x.zero_grad()` clears. A gradient that silently narrowed would be
    invisible in every shape check, so the buffer takes the tracked array's own dtype and
    shape; only gnp arrays carry `.grad`, because a plain numpy array cannot hold one.
    """
    values:Dict[str, object] = dict(arguments)
    buffers:Dict[str, object] = {}
    for wrt_name in wrt:
        value:object = values[wrt_name]
        _check_trackable(form, wrt_name, value, mode)
        if isinstance(value, gnp):
            existing:object = value.grad
            buffer:object = existing if existing is not None else _zeros_like(value)
            value.grad = buffer
        else:
            buffer = _zeros_like(value)
        buffers[wrt_name] = buffer
    return buffers


def _zeros_like(value:object)->object:
    """A zero gradient buffer, on the tracked array's own device through gnp."""
    if isinstance(value, gnp):
        from ..numpy import zeros_like
        return zeros_like(value)
    return np.zeros(value.shape, dtype=value.dtype)


def _check_trackable(form:"_ADForm", name:str, value:object, mode:str)->None:
    """What a tracked array must be: numeric, scalar-dtyped, and real.

    Reverse mode refuses integers as well: a gradient through an integer array is not
    representable, and a buffer that truncated every partial to a whole number would be
    the worst kind of wrong answer. Forward mode allows them, because a tangent is a
    direction of the same width as the value it rides on.
    """
    if not isinstance(value, np.ndarray):
        raise form._error(f"`wrt` names `{name}`, whose value is not an array")
    if value.dtype.kind == "c":
        raise form._error(
            f"`{name}` is complex ({value.dtype}), and complex AD needs the Wirtinger "
            f"convention settled, which is a documented decision rather than a branch "
            f"added in passing -- carry the real and imaginary parts as two float arrays")
    if value.dtype.kind not in "fiu":
        raise form._error(f"`{name}` has dtype {value.dtype}: only numeric arrays "
                          f"differentiate")
    if mode == "reverse" and value.dtype.kind not in "f":
        raise form._error(
            f"`{name}` is {value.dtype}: a gradient through an integer array is not "
            f"representable -- mark a float array, or differentiate a float copy")


def _returned_carried_scalar(scan:BodyScan, form:"_ADForm")->str:
    """The carried scalar the body returns, which is what `loss="return"` reads.

    A body that returns nothing, whose return sites disagree, or whose returned value is
    not launch-wide is refused with a message naming both spellings -- the reduction the
    body answered is where the API reads an answer the body wrote down, never an invented
    `sum(out)`.
    """
    returns:List[ast.Return] = scan.returns()
    if not returns:
        raise form._error(
            "the body returns nothing, so there is no number to take a gradient of: "
            "spell `loss=\"name\"` for the parameter the body accumulates into through "
            "atomic_add, or `out=\"name\"` with `k.vjp` for an array result")
    names:Set[str] = set()
    for ret in returns:
        if not isinstance(ret.value, ast.Name):
            raise form._error(
                "the body's `return` carries an expression, not a name: the loss must be "
                "one launch-wide number -- return the carried scalar itself, or spell "
                "`loss=\"name\"`")
        names.add(ret.value.id)
    if len(names) != 1:
        raise form._error(
            "the body's returns disagree on which value is the loss; the gradient needs "
            "one number -- spell `loss=\"name\"`")
    loss_name:str = names.pop()
    if loss_name not in scan.carried_scalars:
        raise form._error(
            f"the returned `{loss_name}` is not a launch-wide carried scalar, so its "
            f"value is not one number for the launch: initialize it above the loop and "
            f"accumulate with `+=` inside it, or spell `loss=\"name\"`")
    return loss_name


def _origin_tree(origin:object)->ast.FunctionDef:
    """A private deep copy of the origin's tree, for the transform to rewrite."""
    module:ast.Module = parse_kernel_body(origin)
    tree:ast.FunctionDef = copy.deepcopy(module.body[0])
    tree.lineno = module.body[0].lineno
    tree.col_offset = module.body[0].col_offset
    return tree


def _filename_of(origin:object)->str:
    code:object = getattr(origin, "__code__", None)
    return getattr(code, "co_filename", "<kernel>") or "<kernel>"
