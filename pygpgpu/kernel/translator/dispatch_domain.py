"""Which loop is the launch: the one question every region split has to answer the same way.

A kernel body is divided into regions by its *dispatch* loop -- the statements above it run
once on the host, the loop's body runs once per work item, the statements after it are the
host tail. Four places ask where that loop is (the translator's own scan, `_outside_loops`,
`magic_loop_phases`, and the launcher's host-region split), so the question lives here once
and each of them calls in.

That is not a tidiness argument. When two of them disagree, the failure is silent: a
statement lands in the host region on one path and in device code on the other, and the
symptom is a name that is simply undefined at run time (or worse, a host side effect applied
twice) rather than an error naming the rule.

Two spellings state a domain, and the difference between them is *who decides the count*:

* a **shape** -- `range(arr.shape[0])`, `enumerate(arr)`, `np.ndindex(arr.shape)` -- is a
  property of the caller's data;
* a **runtime count** -- `range(n)` with `n` a kernel parameter -- is a value only the call
  knows, so one iteration per work item is the only reading that uses the device at all.
  Without it a body whose only bound was a scalar ran in a single work item: a hundred
  thousand samples on one thread, with nothing in the source looking wrong.

A **compile-time count** (`range(8)`, `range(MAX_ITER)`) is deliberately not a domain. It is
a fixed number of steps one work item walks -- a raymarcher's march, a matmul's inner
product -- and promoting it would turn every such walk into a grid, and a `range(3)` above a
real dispatch into a second pass.
"""
from __future__ import annotations
import ast
from typing import (Callable, Collection, List, Optional, Set, Tuple)


def _parameter_names_of(func:Callable)->Set[str]:
    """The python parameter names of a function, `self`/`cls` included.

    The kernel's own parameters are what may size a counted dispatch (see
    `_counted_range_param`), and every scan that splits a body needs them *before*
    translation: the translator's parameter list is built while the body is walked, so it
    does not exist yet when the split is decided.
    """
    code:object = getattr(func, "__code__", None)
    if code is None:
        return set()
    varnames:Tuple[str, ...] = tuple(getattr(code, "co_varnames", ()) or ())
    positional:int = int(getattr(code, "co_argcount", 0) or 0)
    keyword_only:int = int(getattr(code, "co_kwonlyargcount", 0) or 0)
    # A code object lays its parameters out first -- positional, then keyword-only -- and the
    # body's own locals after them, so the two counts are the whole answer.
    return set(varnames[:positional + keyword_only])


def _is_shape_access(node:ast.expr)->bool:
    """True for `<X>.shape[0]`, the first-dimension bound a dispatch may take.

    `<X>.shape[2]` and `<X>.shape` describe other domains, and `_shape0_range_param` reads
    them on the paths that support them; this predicate is only the 1-D answer.
    """
    return (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute) and node.value.attr == "shape"
            and isinstance(node.slice, ast.Constant) and node.slice.value == 0)


def _counted_range_param(stmt:ast.stmt, params:Collection[str])->Optional[str]:
    """The scalar parameter a `range(<count>)` loop takes its domain from, or None.

    The parameter itself has to be the bound: `n = other; range(n)` stays a serial loop,
    because the alias table records a *shape* alias (`rows = arr.shape[0]`) and not a
    scalar's provenance -- guessing there would silently promote a walk over a computed
    number.
    """
    if not isinstance(stmt, ast.For):
        return None
    iterator:ast.expr = stmt.iter
    if not (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range" and len(iterator.args) == 1
            and not iterator.keywords):
        return None
    bound:ast.expr = iterator.args[0]
    if isinstance(bound, ast.Name) and bound.id in params:
        return bound.id
    return None


def _is_magic_loop_stmt(stmt:ast.stmt, params:Collection[str])->bool:
    """True for a loop *shaped* like the work-item dispatch.

    The test has to be as strict as the dispatch itself, because its other callers
    (`_outside_loops`, `magic_loop_phases`, `_translate_phase_kernels`) use it to decide
    which statements are host code. `for j in range(lane * 4, lane * 4 + 4)` is an ordinary
    serial loop over four elements, but it once counted as the dispatch -- which pushed
    every statement above it into the host region, where a device-only builtin such as
    `local_array_per_group` cannot run (it reported "only available inside a @kernel function
    body" for a body that plainly was one).

    `params` is the kernel's own parameter names, and it is what makes a *counted* domain
    (`for i in range(n)`) recognisable -- see `_counted_range_param`. It is required rather
    than defaulted on purpose: a caller that could omit it would silently get the
    shape-only answer, and then the host-region boundary would disagree with the translator
    about which loop is the dispatch.
    """
    if not isinstance(stmt, ast.For):
        return False
    if _counted_range_param(stmt, params) is not None:
        return True
    iterator:ast.expr = stmt.iter
    if not isinstance(iterator, ast.Call):
        return False
    func:ast.expr = iterator.func
    if isinstance(func, ast.Name) and func.id == "range":
        # One argument, and it addresses an array's first dimension: the shape dispatch form.
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


def _magic_loop_index(body:List[ast.stmt], params:Collection[str])->int:
    """The index of the outermost loop that becomes work-item dispatch, or len(body).

    The same detection the translator uses, reduced to "where does the body's host region
    end": statements before it run once, everything from it on runs per work item.
    """
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt, params):
            return index
    return len(body)


def _magic_loop_end(statements:List[ast.stmt], params:Collection[str])->int:
    """The index just past the statements that belong to the magic loop's body.

    A phase is `[header..., for-loop, after...]`, so everything from the loop onwards except
    its own body is the *gap* before the next phase -- the statements a body places between
    two magic loops.
    """
    for index, stmt in enumerate(statements):
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt, params):
            return index + 1
    return len(statements)


def _outside_loops(body:List[ast.stmt], params:Collection[str])->List[ast.stmt]:
    """The top-level statements a body runs on the host, outside its work-item dispatch.

    A single-loop body places all of them before the loop, but a body with several loops
    may also place one *between* two of them (`mid = np.zeros(4)` in a two-pass body), and
    the pass that reads such a value needs it bound on the host just the same.

    Only statements whose meaning is *host* work belong here. A serial `for` loop and an
    `if` block are per-work-item code that the device must run (every work item iterates
    its own slice, or tests its own lane), so they are left to the translator; returning
    them to the host would execute device-only builtins there -- `local_array_per_group` used
    under such a loop reported "only available inside a @kernel function body" for a body
    that plainly was one.
    """
    loops:List[int] = [
        index for index, stmt in enumerate(body)
        if isinstance(stmt, ast.For) and _is_magic_loop_stmt(stmt, params)]
    # A statement is host work only if the host can *execute* it. The body's serial loops and
    # its `if` blocks are per-work-item code: every work item iterates its own slice or tests
    # its own lane, so handing them to the host would run device-only builtins there --
    # `local_array_per_group` under such a loop reported "only available inside a @kernel function
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


__all__ = [
    "_counted_range_param",
    "_is_magic_loop_stmt",
    "_is_shape_access",
    "_magic_loop_end",
    "_magic_loop_index",
    "_outside_loops",
    "_parameter_names_of",
]
