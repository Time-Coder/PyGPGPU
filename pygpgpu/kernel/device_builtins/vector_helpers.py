"""Device helpers for the vector shapes CUDA and HIP have no native form of.

CUDA's `cosf`/`floorf`/`fract`-template family is scalar-only, and this NVRTC
environment carries no vector arithmetic operators either, so every vector use was
expanded per component at the call site -- which duplicated the argument expression
once per component and compounded through nesting (`a + b*cos(c*t + d)` copied the
inner expression nine times into the generated source). The complex types solved the
same problem the way `kernel/include/complex.h` does: one device function per
operation, defined once, called by name -- so the argument expression appears exactly
once and the compiler inlines the helper back to the same instructions.

This module builds those helper definitions. A call the translator cannot pass
through natively (a math builtin, the clamp family, vector arithmetic) gets a
`pygpgpu_<name>` function overloaded per shape (`pygpgpu_mul(float3, float)` and
`pygpgpu_mul(float, float3)` coexist -- the CUDA/HIP compilers are C++), and the
definition travels in the translator's `device_helpers` channel, the same one the
HIP round-toward-zero helpers use.

The bodies spell exactly what the per-component expansion they replace spelled, so
a helper call computes the same values in the same width; only the duplication is
gone. OpenCL and Metal are untouched: they have native vector overloads for all of
this and keep the direct call.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

# The prefix every helper carries. One namespace, overloaded per shape: a generated
# source can use `pygpgpu_mul` for float2, float3 and the scalar-broadcast forms
# alike, and the C++ overload set picks by the argument types.
PREFIX:str = "pygpgpu"

# The helper name per infix symbol. A comparison returns the int vector of the
# operands (0/1 per component, the form the expansion unified to), so its helper
# needs its own name beside the arithmetic one.
ARITH_HELPERS:Dict[str, str] = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
COMPARE_HELPERS:Dict[str, str] = {"<": "less", "<=": "less_equal", ">": "greater",
                                  ">=": "greater_equal", "==": "equal",
                                  "!=": "not_equal"}



def _fmin_max(width:str)->Tuple[str, str]:
    """The min/max pair the width spells: `fminf`/`fmaxf` for float, `fmin`/`fmax`."""
    return (("fmin", "fmax") if width == "double" else ("fminf", "fmaxf"))


def _literal(value:str, width:str)->str:
    """A real constant in its width: `2.0` for double, the `f` suffix for float."""
    return value if width == "double" else f"{value}f"


def helper_name(name:str, param_types:Tuple[str, ...])->str:
    """The unique C name of one helper shape.

    This NVRTC environment compiles the input as C for these functions -- a second
    definition of one name fails with `more than one instance of overloaded function
    has C linkage`, which is why `complex.h` spells every function's width in its
    name (`c64_add`, `c128_add`) and never relies on C++ overloading. The vector
    helpers follow: the parameter types join the name, so `mul` exists as
    `pygpgpu_mul_float3_float`, `pygpgpu_mul_float_float3` and friends, one
    definition each.
    """
    return f"{PREFIX}_{name}_{'_'.join(param_types)}"


@lru_cache(maxsize=None)
def helper_source(name:str, spell:str, kind:str, ret:str, vec:str,
                  shape:Tuple[str, ...], param_types:Tuple[str, ...])->str:
    """The `__device__` definition one helper shape needs, built once and reused.

    `name` is the python-facing builtin (`cos`, `fract`, `add`); `spell` is the
    scalar device form this backend uses (`cosf`, the fract template, the `+`
    symbol); `kind` selects the body shape; `ret` is the return type (`vec` for
    arithmetic and the math family, the int vector for a comparison); `vec` is the
    vector whose components the body walks; `shape` records per argument whether it
    is a vector (`v`) or a scalar (`s`); `param_types` are the declared types, in
    order. The body names its parameters `a`, `b`, `c` in order.
    """
    param_names:List[str] = ["a", "b", "c"][:len(shape)]
    components:int = {"float2": 2, "float3": 3, "float4": 4,
                      "double2": 2, "double3": 3, "double4": 4,
                      "int2": 2, "int3": 3, "int4": 4,
                      "uint2": 2, "uint3": 3, "uint4": 4}[vec]
    width:str = "double" if vec.startswith("double") else "float"
    # The per-component access for each argument at one component: `a.x`, `a.y`,
    # ... for a vector argument, or the scalar's own text (a scalar parameter is one
    # value, shared by the components).
    def access(index:int, component:int)->str:
        if shape[index] != "v":
            return param_names[index]
        return f"{param_names[index]}.{chr(ord('x') + component)}"
    terms:List[str] = []
    for component in range(components):
        parts:List[str] = [access(index, component) for index in range(len(shape))]
        terms.append(_term(name, spell, kind, parts, width))
    signature:str = ", ".join(f"{ptype} {pname}"
                              for ptype, pname in zip(param_types, param_names))
    return (f"__device__ {ret} {helper_name(name, param_types)}({signature})\n"
            "{\n"
            f"    return make_{ret}({', '.join(terms)});\n"
            "}\n")


def _term(name:str, spell:str, kind:str, parts:List[str], width:str)->str:
    """One component of the helper body: what the expansion spelled, on `a.x` pieces."""
    if kind == "call":
        return f"{spell}({', '.join(parts)})"
    if kind == "template":
        return spell.format(code=parts[0])
    if kind == "arith":
        return f"({parts[0]} {spell} {parts[1]})"
    if kind == "compare":
        return f"(({parts[0]} {spell} {parts[1]}) ? 1 : 0)"
    if kind == "neg":
        return f"(-{parts[0]})"
    if name == "clamp":
        if len(parts) == 1:
            # `saturate(x)`: the clamp pair is the constants 0 and 1 in the width.
            parts = [parts[0], _literal("0.0", width), _literal("1.0", width)]
        minimum, maximum = _fmin_max(width)
        return f"{minimum}({maximum}({parts[0]}, {parts[1]}), {parts[2]})"
    if name == "mix":
        one = _literal("1.0", width)
        return (f"(({parts[0]}) * ({one} - ({parts[2]}))"
                f" + ({parts[1]}) * ({parts[2]}))")
    if name == "mad":
        return f"(({parts[0]}) * ({parts[1]}) + ({parts[2]}))"
    if name == "step":
        zero, one = _literal("0.0", width), _literal("1.0", width)
        return f"(({parts[1]}) < ({parts[0]})) ? {zero} : {one}"
    if name == "smoothstep":
        minimum, maximum = _fmin_max(width)
        zero, one = _literal("0.0", width), _literal("1.0", width)
        three, two = _literal("3.0", width), _literal("2.0", width)
        t:str = (f"{minimum}({maximum}((({parts[2]}) - ({parts[0]}))"
                 f" / (({parts[1]}) - ({parts[0]})), {zero}), {one})")
        return f"(({t}) * ({t}) * ({three} - ({two}) * ({t})))"
    raise ValueError(f"no device helper body for {name!r}")


def helper_call(call_name:str, arg_codes:List[str])->str:
    """The call text one site emits: every argument expression appears exactly once."""
    return f"{call_name}({', '.join(arg_codes)})"
