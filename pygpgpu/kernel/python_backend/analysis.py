# What the interpreter needs to know about a body before it runs it.
from __future__ import annotations
from .magic import (
    Holder,
    _MAGIC_ENUMERATE,
    _MAGIC_RANGE,
    _dedented_ast,
    _globals_of,
    _gpu_attr,
    _nd_magic,
    _shape0_holder,
    _enumerate_holder,
)
import ast
from typing import Callable, Dict, List, Optional
from pygf import wrap_vector_array
from ..device_builtins import INDEX_BUILTINS



class PythonAnalysis:
    """What the serial runner needs to know about a kernel body."""

    def __init__(self) -> None:
        # Arrays the decorator's `domain=` named: the gid-style iteration domain.
        self.declared: List[Holder] = []
        # Work size derivable from the magic loop (reported by
        # gpu.get_global_size/get_local_size/get_num_groups).
        self.magic_form: Optional[str] = None
        self.magic_holder: Optional[Holder] = None
        # The body calls a gpu indexing builtin, so it dispatches on the work item.
        self.uses_index: bool = False



def analyse(func: Callable, self_param: str = "", domain: str = "") -> PythonAnalysis:
    """Inspect a kernel body for the two launch styles (never translates anything).

    ``domain`` is the parameter name `@kernel(domain=...)` declared, if any, so it seeds the
    analysis with that declaration instead of requiring the body to carry one.
    """
    analysis = PythonAnalysis()
    body: Optional[ast.FunctionDef] = _dedented_ast(func)
    if body is None:
        return analysis
    globals_of: Dict[str, object] = _globals_of(func)
    if domain:
        # A parameter is held as ("arg", <name>), the same shape the statement produces.
        analysis.declared.append(("arg", domain))

    for stmt in body.body:
        if not isinstance(stmt, ast.For):
            continue
        holder = _shape0_holder(stmt.iter, self_param)
        form = _MAGIC_RANGE
        if holder is None:
            holder = _enumerate_holder(stmt.iter, self_param)
            form = _MAGIC_ENUMERATE
        if holder is None:
            nd = _nd_magic(stmt.iter, self_param, globals_of)
            if nd is not None:
                form, holder = nd
        if holder is not None:
            analysis.magic_form = form
            analysis.magic_holder = holder
            break

    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        attr: Optional[str] = _gpu_attr(node, globals_of)
        if attr is None:
            continue
        if attr in INDEX_BUILTINS:
            analysis.uses_index = True

    return analysis



def as_host_array(value: object) -> object:
    """The value as the body should see it: records and vectors, not their carriers.

    A `ctypes.Structure` instance is a structure *value* and is passed through unchanged: it
    already answers both spellings a kernel body may use -- `p["i"]` (installed by
    `prepare_structure_classes`) and `p.i` (the ctypes attribute) -- whereas converting it to
    a record would throw the attribute form away. An array of that dtype is left alone too,
    since its elements read back as records.

    A `ctypes.Union` instance is the exception on some platforms: CPython there gives a
    union subclass no mapping slot at all, so a `__getitem__` assigned to the class
    dictionary is never reached and `p["i"]` raises "'U' object is not subscriptable"
    (see `structs._add_subscript_access`). The instance is wrapped in a proxy that
    answers the subscript itself and forwards everything else -- reads and writes --
    to the real union, which keeps the attribute spelling working too.
    """
    from .. import structs

    if structs.union_needs_proxy(value):
        return structs.UnionProxy(value)
    return wrap_vector_array(value)



def prepare_structure_classes(func: Callable[..., object]) -> None:
    """Give every `ctypes` structure or union class this function names a member subscript.

    The interpreter runs the body unchanged, so `ray["origin"]` needs to work on the
    instance the body constructed -- but nothing on this path inspects argument types, which
    is where a class would otherwise be registered. The body's own names are the reliable
    source: `Ray` is a global of the function, and its own module is what a body usually
    refers to.

    A union is a sibling of `ctypes.Structure`, not a subclass, so it is asked for through
    `is_ctypes_record` -- otherwise a union body that spells `p["i"]` had no subscript to
    reach and the interpreter backend alone failed on a kernel every device backend ran.
    """
    import sys

    from .. import structs

    namespace: dict = getattr(func, "__globals__", {}) or {}
    for value in list(namespace.values()):
        if isinstance(value, type) and structs.is_ctypes_record(value):
            structs.register_class(value)
    for value in list(vars(sys.modules["builtins"]).values()):
        # `ctypes.Structure` itself, which a body may name -- harmless, already a class.
        if isinstance(value, type) and structs.is_ctypes_record(value):
            structs.register_class(value)
