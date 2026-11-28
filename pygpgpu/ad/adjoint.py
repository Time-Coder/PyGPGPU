# Assembling an adjoint kernel: the transformed tree becomes an ordinary PyKernel.
#
# The transform's output is a Python function like any other, and that is the point: the
# adjoint kernel reads gradient buffers and writes them, so the existing binding, copy
# policy, cache and launch machinery apply unchanged (the plan's "the adjoint kernel is an
# ordinary kernel" decision). What this module adds is only the wiring: the tree is
# compiled into a function object, its tree is registered for every source-level reader,
# and the kernel object is built with the parent's own configuration.
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from ..kernel.translator.source_text import register_adjoint_tree
from .ast_support import NameAllocator
from .rules import INJECTS

# The kinds an extra parameter can travel as, which is what the form allocates for each.
GRAD_PARAM = "grad"      # a gradient buffer this form allocated (zeros, accumulated into)
TANGENT_PARAM = "tangent"  # a direction array the caller supplied
COT_PARAM = "cot"        # a cotangent weight array the caller supplied


class AdjointSpec:
    """What a transform produced, and what the assembled kernel needs from its caller.

    ``extra_params`` is ordered: it is the tail of the adjoint function's signature, and
    the form appends the matching arrays in exactly that order at launch.
    """

    def __init__(self, funcdef:ast.FunctionDef, extra_params:List[Tuple[str, str]],
                 name:str, helpers:Optional[Dict[str, object]] = None):
        self.funcdef:ast.FunctionDef = funcdef
        self.extra_params:List[Tuple[str, str]] = list(extra_params)
        self.name:str = name
        # Registered helper adjoints the body calls, keyed by the mangled spelling their
        # call sites spell; the namespace carries the function objects under them.
        self.helpers:Dict[str, object] = dict(helpers or {})


def assemble(spec:AdjointSpec, parent:object, allocator:NameAllocator)->object:
    """Compile the transformed tree into a registered PyKernel over it.

    The adjoint function's globals are the parent kernel's own, so every name the body
    resolves -- the builtins, the helpers, the modules -- resolves to the same object the
    forward call resolved to. The rule-provided spellings are added under mangled names,
    so a user's own ``sqrt`` cannot shadow the package's.
    """
    funcdef:ast.FunctionDef = spec.funcdef
    funcdef.name = spec.name
    for param_name, _kind in spec.extra_params:
        funcdef.args.args.append(ast.arg(arg=param_name, annotation=None))
    rename_injects(funcdef, allocator)
    module:ast.Module = ast.Module(body=[funcdef], type_ignores=[])
    # fix_missing_locations fills a node's position from its parent, so the statement
    # lines the transforms placed (the user's own lines) anchor everything they contain;
    # the one statement it cannot place is the FunctionDef itself when the tree arrived
    # without positions, which is why lineno is set here on the def.
    funcdef.lineno = getattr(funcdef, "lineno", None) or 1
    funcdef.col_offset = getattr(funcdef, "col_offset", None) or 0
    ast.fix_missing_locations(module)
    origin:object = parent.raw_func
    filename:str = getattr(getattr(origin, "__code__", None), "co_filename",
                           "<pygpgpu-adjoint>")
    namespace:Dict[str, object] = dict(getattr(origin, "__globals__", {}))
    namespace["__name__"] = getattr(origin, "__module__", "") or "pygpgpu.ad"
    # A nested kernel's helpers resolve through the enclosing function's cells, and the
    # exec'd adjoint has no closure of its own -- so the origin's free variables are
    # copied into the namespace, which is where the translator's resolver looks next.
    origin_code:object = getattr(origin, "__code__", None)
    for cell_name, cell in zip(getattr(origin_code, "co_freevars", ()) or (),
                               getattr(origin, "__closure__", None) or ()):
        try:
            namespace[cell_name] = cell.cell_contents
        except ValueError:
            pass
    for spell, target in INJECTS.items():
        namespace[mangled_spell(spell)] = target
    for spell, target in spec.helpers.items():
        namespace[spell] = target
    code:object = compile(module, filename=filename, mode="exec")
    exec(code, namespace)
    func:object = namespace[spec.name]
    func.__qualname__ = f"{getattr(origin, '__qualname__', spec.name)}_adjoint"
    func.__doc__ = None
    register_adjoint_tree(func, module, origin)
    kernel:object = _build_kernel(spec, func, parent)
    return kernel


def rename_injects(funcdef:ast.FunctionDef, allocator:NameAllocator)->None:
    """Rename the rule-provided spellings to their mangled forms, in place.

    A partial builder spells ``sqrt(...)``; the adjoint namespace carries the package's
    own sqrt under ``_ad_sqrt`` (or the first free mangling), so the call resolves to the
    device builtin even when the user's module binds ``sqrt`` to something else.
    """
    renames:Dict[str, str] = {}
    for node in ast.walk(funcdef):
        if isinstance(node, ast.Name) and node.id in INJECTS:
            renames.setdefault(node.id, allocator.fresh(mangled_spell(node.id)))
    for node in ast.walk(funcdef):
        if isinstance(node, ast.Name) and node.id in renames:
            node.id = renames[node.id]


def mangled_spell(spell:str)->str:
    """The namespace spelling of a rule-provided name."""
    return f"_ad_{spell}"


def _build_kernel(spec:AdjointSpec, func:object, parent:object)->object:
    """The PyKernel, with the parent's own configuration and the rules version folded in.

    The kernel is built directly rather than through the ``@kernel`` factory so it shares
    the parent's backend snapshot and default devices; the ``type_map`` is copied after
    construction (the constructor resolves a *declared* mapping, and this one is already
    resolved), and the source signature is recomputed so the AD rules' version is part of
    what the adjoint kernel is.
    """
    from ..kernel.decorator import PyKernel
    kernel:object = PyKernel(func, backend=parent.backend,
                             default_devices=parent._default_devices,
                             domain=parent._domain)
    kernel._type_map = dict(parent._type_map)
    kernel._source_sig = kernel._source_signature()
    # The tape records launches through the pipeline hook; an adjoint launch must never
    # be recorded, or a backward inside a `with Tape():` would chase itself.
    kernel._pygpgpu_adjoint = True
    return kernel
