# The AST questions asked before a body is translated.
from __future__ import annotations
from .predicates import is_ptr
from .source_text import _parse_kernel_source, get_dedented_source
import ast
import builtins
import functools
import hashlib
from typing import (Callable, Dict, List, Optional, Set)
import numpy as np



# numpy names a kernel body may use as a type in a creation statement (`dtype=np.float32`).
_NUMPY_DEVICE_TYPES:Set[str] = {
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
}

# numpy functions that build or transform host data, and can therefore be executed by the
# interpreter instead of translated (see _creation_assignments). This is the whole family
# of array constructors and shape/layout transforms: any of them may appear on the
# right-hand side of a statement above the magic loop. Device-side arithmetic on the
# result is not part of a creation statement, which is why only these names are listed --
# a `np.sum(...)` or `np.dot(...)` there is a computation the kernel itself should do.
_HOST_CREATION_FUNCTIONS:Set[str] = {
    # constructors
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like",
    "full_like", "arange", "linspace", "logspace", "geomspace",
    "identity", "eye", "tri", "tril", "triu", "diag", "diagflat", "vander",
    "indices", "meshgrid", "fromfunction", "frombuffer", "fromiter", "fromfile",
    "array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray",
    "copy", "copyto", "filled",
    # shape and layout transforms
    "reshape", "ravel", "flatten", "transpose", "swapaxes", "moveaxis",
    "flip", "fliplr", "flipud", "roll", "rot90", "squeeze", "expand_dims",
    "broadcast_to", "broadcast_arrays", "atleast_1d", "atleast_2d", "atleast_3d",
    "repeat", "tile", "resize", "pad", "take", "compress", "trim_zeros",
    # combining and splitting
    "concatenate", "stack", "hstack", "vstack", "dstack", "column_stack",
    "row_stack", "block", "split", "hsplit", "vsplit", "dsplit", "append",
    "insert", "delete", "unique",
    # element-wise or ordering helpers that only reshape the data
    "where", "clip", "round", "around", "sort", "argsort", "lexsort", "cumsum",
    "cumprod", "nan_to_num", "real", "imag", "conj", "angle",
    # dtype constructors (np.float32(x) and friends)
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_",
}

# The statements a kernel body may execute on the host: assignments above the magic loop
# whose right-hand side only builds data.
def _magic_loop_end(statements:List[ast.stmt])->int:
    """The index just past the statements that belong to the magic loop's body.

    A phase is `[header..., for-loop, after...]`, so everything from the loop onwards except
    its own body is the *gap* before the next phase -- the statements a body places between
    two magic loops.
    """
    for index, stmt in enumerate(statements):
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt):
            return index + 1
    return len(statements)


def _outside_loops(body:List[ast.stmt])->List[ast.stmt]:
    """The top-level statements a body runs on the host, outside its work-item dispatch.

    A single-loop body places all of them before the loop, but a body with several loops
    may also place one *between* two of them (`mid = np.zeros(4)` in a two-pass body), and
    the pass that reads such a value needs it bound on the host just the same.

    Only statements whose meaning is *host* work belong here. A serial `for` loop and an
    `if` block are per-work-item code that the device must run (every work item iterates
    its own slice, or tests its own lane), so they are left to the translator; returning
    them to the host would execute device-only builtins there -- `zeros_per_group` used
    under such a loop reported "only available inside a @kernel function body" for a body
    that plainly was one.
    """
    loops:List[int] = [
        index for index, stmt in enumerate(body)
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt)]
    # A statement is host work only if the host can *execute* it. The body's serial loops and
    # its `if` blocks are per-work-item code: every work item iterates its own slice or tests
    # its own lane, so handing them to the host would run device-only builtins there --
    # `zeros_per_group` under such a loop reported "only available inside a @kernel function
    # body" for a body that plainly was one. A bare `for` / `If` is therefore never host code,
    # with or without a dispatch loop; everything else (a creation, a plain assignment) is,
    # because its value has to exist before the launch.
    per_item = (ast.For, ast.Return, ast.If)
    if not loops:
        # A gid-style body: all of it is per-work-item code, except the creations the host
        # still has to build (see _creation_assignments), which stay in the list.
        return [stmt for stmt in body if not isinstance(stmt, per_item)]
    # Everything before the first loop, plus the gap between consecutive loops: those are
    # the statements a body places outside its work-item dispatch, and a value bound there
    # (a created array, a scalar) is read by a later pass. Statements *after* the last loop
    # are excluded -- `return out` lives there and must keep its device meaning.
    outside:List[ast.stmt] = [stmt for stmt in body[:loops[0]]
                              if not isinstance(stmt, per_item)]
    for start, end in zip(loops, loops[1:]):
        outside.extend(stmt for stmt in body[start + 1:end]
                       if not isinstance(stmt, per_item))
    return outside


def _creation_assignments(func:Callable)->List[ast.stmt]:
    """The statements the host runs before the launch, in body order.

    Two kinds are lifted out, both from **above** the magic loop:

    * a **creation** (`arr = np.zeros(n)`) -- the object has to exist with the shape and
      dtype numpy gives it, and a device has no way to make one;
    * a plain **assignment** that a later creation reads (`n = 5` feeding `np.zeros(n)`) --
      the host has to have the name in scope when it evaluates the creation.

    The second kind is *also* still emitted as device code, because a work item reads it
    too; the host evaluation is an extra, not a replacement. Everything else above the loop
    is device code only. A magic loop is what defines "above" here: an assignment below it
    would run once per work item, so it is not lifted out (the translator refuses a creation
    there).
    """
    source:str = get_dedented_source(func)
    tree:ast.Module = _parse_kernel_source(source)
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return []
    body:List[ast.stmt] = tree.body[0].body
    magic_index:Optional[int] = next(
        (index for index, stmt in enumerate(body) if _is_magic_loop_stmt(stmt)), None)
    # A gid-style body has no magic loop, so its whole body is "above the work item": it is
    # the per-work-item code, and a top-level creation statement there still runs once per
    # launch on the host, which is what python means.
    if magic_index is None:
        magic_index = len(body)

    import builtins as _builtins

    globalns:Dict[str, object] = getattr(func, "__globals__", {})

    def is_numpy(node:ast.expr)->bool:
        return (isinstance(node, ast.Name) and globalns.get(node.id) is np)

    def is_creation(node:ast.expr)->bool:
        return _is_creation_shape(node, is_numpy)

    del _builtins
    # Which plain assignments a creation needs: the names a creation statement reads that
    # are bound above the loop but are not themselves creations. Only those are lifted, so
    # the host does not run unrelated device code -- and everything lifted here is still
    # emitted for the work items as well.
    creations:List[ast.stmt] = [
        stmt for stmt in _outside_loops(body)
        if _creation_target_of(stmt) is not None and is_creation(_creation_target_of(stmt))]
    needed:set = set()
    for stmt in creations:
        for node in ast.walk(stmt.value):
            if isinstance(node, ast.Name):
                needed.add(node.id)

    bound_above:Dict[str, ast.stmt] = {}
    for stmt in _outside_loops(body):
        name:Optional[str] = _assigned_name_of(stmt)
        if name is not None:
            bound_above.setdefault(name, stmt)

    creation_names:set = {_assigned_name_of(stmt) for stmt in creations}
    # A fixed point, because a lifted assignment can depend on another lifted one:
    # `a = 2; b = a * 2; c = np.zeros(b)` needs all three, and one pass over the creation's
    # own names would stop at `b` and leave `a` undefined when the host runs it.
    fitting:set = set()
    changed:bool = True
    while changed:
        changed = False
        for name in list(needed):
            if name in fitting or name in creation_names:
                continue
            stmt:Optional[ast.stmt] = bound_above.get(name)
            if stmt is None or stmt not in _outside_loops(body):
                continue
            if not _is_host_evaluable(stmt.value if stmt.value is not None else stmt):
                continue
            fitting.add(name)
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and node.id not in fitting:
                    needed.add(node.id)
            changed = True

    chosen:List[ast.stmt] = []
    for stmt in _outside_loops(body):
        value:Optional[ast.expr] = _creation_target_of(stmt)
        name:Optional[str] = _assigned_name_of(stmt)
        if value is not None and is_creation(value):
            chosen.append(stmt)
        elif name is not None and name in fitting and name not in creation_names:
            chosen.append(stmt)
    return chosen


def _is_host_evaluable(node:ast.expr)->bool:
    """Whether a right-hand side can be computed by the interpreter, for a creation's need.

    Only the forms that carry no device-only meaning: a literal, a plain name, attribute
    access, and arithmetic over those. A call is excluded because it may name a device
    helper whose body the host cannot run -- lifting it would either fail or, worse, run
    something different from what the device computes.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, (ast.Name, ast.Attribute))
    if isinstance(node, ast.BinOp):
        return _is_host_evaluable(node.left) and _is_host_evaluable(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_host_evaluable(node.operand)
    if isinstance(node, ast.Subscript):
        return _is_host_evaluable(node.value) and _is_host_evaluable(node.slice)
    if isinstance(node, ast.Tuple):
        return all(_is_host_evaluable(element) for element in node.elts)
    return False


def _is_creation_shape(node:ast.expr, is_numpy:Callable[[ast.expr], bool])->bool:
    """True when this expression only *builds* data (and cannot run on the device).

    The device has no numpy, so a constructor that allocates is host-only by definition.
    A scalar-type call (`np.float32(...)`) is a creation only when its arguments are pure
    constants, because `np.float32(x[i])` is a device cast instead.
    """
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Name):
        return False
    if isinstance(node, ast.Attribute):
        return is_numpy(node.value)
    if isinstance(node, ast.Call):
        func_node:ast.expr = node.func
        if not (isinstance(func_node, ast.Attribute) and is_numpy(func_node.value)):
            return False
        name:str = func_node.attr
        # The class attribute is read through the facade, lazily: this module is below
        # the class in the package, and a top-level import would be a cycle.
        from . import FunctionTranslator
        if name in _NUMPY_DEVICE_TYPES or name in FunctionTranslator.RETURN_SLOT_TYPES:
            return all(isinstance(arg, ast.Constant) for arg in node.args)
        if name not in _HOST_CREATION_FUNCTIONS and not name.endswith("_like"):
            return False
        # The arguments may be ordinary values (a shape, a count, an input array); what
        # makes the statement host-only is the numpy constructor itself. Only a dtype
        # argument has to be a type name, which is why it is not evaluated on the device.
        return True
    return False


def _is_creation_expression(node:ast.expr, func:Callable)->bool:
    """Whether a right-hand side is a creation, answering for one expression alone.

    The launcher needs this to tell a creation from a plain assignment it lifted for a
    creation's sake: only the creation travels to the device as a parameter.
    """
    globalns:Dict[str, object] = getattr(func, "__globals__", {})
    return _is_creation_shape(
        node, lambda candidate: isinstance(candidate, ast.Name)
        and globalns.get(candidate.id) is np)


def _creation_target_of(stmt:ast.stmt)->Optional[ast.expr]:
    """The right-hand side of `name = expr` / `name: T = expr`, when the target is a plain name.

    An annotated assignment is the same statement to python -- the annotation is only a
    hint for readers and type checkers -- so `x: np.ndarray = np.zeros_like(y)` creates an
    array exactly like `x = np.zeros_like(y)`. Recognising only the unannotated form made
    the annotated one fall through to the device code, which has no numpy.
    """
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return None
        target:ast.expr = stmt.targets[0]
        return stmt.value if isinstance(target, ast.Name) else None
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return None  # `x: np.ndarray` declares nothing, so there is nothing to create
        return stmt.value if isinstance(stmt.target, ast.Name) else None
    return None


def _constant_int_of(node:ast.expr)->Optional[int]:
    """The integer an expression is a literal for, or None.

    `-1` reaches the translator as `UnaryOp(USub, Constant(1))`, not as a negative
    constant -- the parser does not fold it -- so a rule written against `ast.Constant`
    alone silently misses every negative literal. That is exactly where it matters: a
    negative index and a reversed slice are both written that way.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner:Optional[int] = _constant_int_of(node.operand)
        return None if inner is None else -inner
    return None


def _assigned_name_of(stmt:ast.stmt)->Optional[str]:
    """The plain name an assignment binds, for either assignment spelling."""
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return None
        target:ast.expr = stmt.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(stmt, ast.AnnAssign):
        return stmt.target.id if isinstance(stmt.target, ast.Name) else None
    return None


def _magic_loop_index(body:List[ast.stmt])->int:
    """The index of the outermost loop that becomes work-item dispatch, or len(body).

    The same detection the translator uses, reduced to "where does the body's host region
    end": statements before it run once, everything from it on runs per work item.
    """
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt):
            return index
    return len(body)


def _is_shape_access(node:ast.expr)->bool:
    """True for `<X>.shape[0]`, the only bound a work-item dispatch may take.

    This is the same shape `_shape0_range_param` insists on when it decides the loop
    is the dispatch: a plain count, a start/step, or any other expression describes a
    *serial* loop the work item runs itself, and is not the iteration domain.
    """
    return (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute) and node.value.attr == "shape"
            and isinstance(node.slice, ast.Constant) and node.slice.value == 0)


def _is_magic_loop_stmt(stmt:ast.stmt)->bool:
    """True for a loop *shaped* like the work-item dispatch.

    The test has to be as strict as the dispatch itself, because its other callers
    (`_outside_loops`, `magic_loop_phases`, `_translate_phase_kernels`) use it to decide
    which statements are host code. `for j in range(lane * 4, lane * 4 + 4)` is an ordinary
    serial loop over four elements, but it used to count as the dispatch -- which pushed
    every statement above it into the host region, where a device-only builtin such as
    `zeros_per_group` cannot run (it reported "only available inside a @kernel function
    body" for a body that plainly was one). `_is_secondary_magic_loop` already told a
    literal `range(8)` apart from a real domain; this predicate has to agree with it.
    """
    if not isinstance(stmt, ast.For):
        return False
    iterator:ast.expr = stmt.iter
    if not isinstance(iterator, ast.Call):
        return False
    func:ast.expr = iterator.func
    if isinstance(func, ast.Name) and func.id == "range":
        # One argument, and it addresses an array's first dimension: the dispatch form.
        return (len(iterator.args) == 1 and not iterator.keywords
                and _is_shape_access(iterator.args[0]))
    if isinstance(func, ast.Name) and func.id == "enumerate":
        # `enumerate([1.0, 2.0])` is a dispatch too: the domain is a literal list, whose
        # length and elements are compile-time constants.
        if len(iterator.args) == 1 and not iterator.keywords:
            argument:ast.expr = iterator.args[0]
            if isinstance(argument, (ast.List, ast.Tuple)):
                return all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                           and not isinstance(e.value, bool) for e in argument.elts)
        return len(iterator.args) == 1 and not iterator.keywords
    if (isinstance(func, ast.Attribute) and func.attr in ("ndindex", "ndenumerate")
            and isinstance(func.value, ast.Name)):
        return True
    return False


_UNRESOLVED:object = object()

# Python builtins are never kernel parameters: they resolve through the translator's own
# dispatch (print, len, range, ...) or are rejected with a message of their own.
_BUILTIN_NAMES:Set[str] = set(dir(builtins))


def _declared_scope_names(tree:ast.AST)->Set[str]:
    """Names the function declares with `nonlocal`/`global`.

    Python requires such a declaration before a nested function may assign to an outer
    name, so the declarations are exactly the writable outer variables. Both statements
    are otherwise unsupported in a kernel body, which is why this scan happens before the
    body is visited instead of in the statement dispatch.
    """
    names:Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Nonlocal):
            names.update(node.names)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


@functools.lru_cache(maxsize=512)
def _outer_scope_names_cached(func:object)->frozenset:
    """The declared outer names, cached: a launch asks for them on every call.

    The answer depends only on the function's source, and the walk behind it visits every
    node of the body -- which for a kernel with no `global`/`nonlocal` at all still cost a
    full tree walk per launch, on a path that has to stay close to the launch itself.

    A frozen set is cached and callers get a fresh `set` copy, because the callers treat the
    answer as their own working set.
    """
    try:
        tree:ast.Module = _parse_kernel_source(get_dedented_source(func))
    except SyntaxError:
        return frozenset()
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return frozenset()
    return frozenset(_declared_scope_names(tree.body[0]))


def _outer_scope_names(func:object)->Set[str]:
    """The names this function's body declares `nonlocal`/`global`.

    Those travel through the outer scope's own device slot: the *device* writes them, so
    the host must not run the statement that assigns them (see `_host_region_values`).
    """
    return set(_outer_scope_names_cached(func))


def type_sig(types:List[object])->str:
    parts:List[str] = []
    for t in types:
        parts.append((t[1] + "[]") if is_ptr(t) else str(t))
    return "_".join(parts) or "void"


def short_hash(text:str)->str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
