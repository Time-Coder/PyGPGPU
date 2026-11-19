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
through natively (a math builtin, the clamp family, vector arithmetic, the geometric
family, a unary sign) gets a `pygpgpu_<name>` function overloaded per shape
(`pygpgpu_mul(float3, float)` and `pygpgpu_mul(float, float3)` coexist -- the
CUDA/HIP compilers are C++), and the definition travels in the translator's
`device_helpers` channel, the same one the HIP round-toward-zero helpers use.

The geometric family -- `dot`, `length`, `distance`, `normalize`, `cross` -- does
not fit the per-component term form: its result is a scalar (the sum over all
components) or a formula that *mixes* components (cross's
`a.y * b.z - a.z * b.y`), so a one-component-at-a-time body cannot spell it.
`geometric_source` composes those bodies whole, from the same formulas the inline
expansion spelled. The reductions (`any`, `all`) fold across components the same
way. The bodies compute the same values in the same width the expansions did; only
the duplication is gone. OpenCL and Metal are untouched: they have native vector
overloads for all of this and keep the direct call.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

# The prefix every helper carries, in complex.h's style (`c64_add` names the type
# beside the operation): short, and not a name a kernel body is likely to define.
PREFIX:str = "vec"

# The helper name per infix symbol. A comparison returns the int vector of the
# operands (0/1 per component, the form the expansion unified to), so its helper
# needs its own name beside the arithmetic one.
ARITH_HELPERS:Dict[str, str] = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
COMPARE_HELPERS:Dict[str, str] = {"<": "less", "<=": "less_equal", ">": "greater",
                                  ">=": "greater_equal", "==": "equal",
                                  "!=": "not_equal"}


def _component_count(vec:str)->int:
    """How many components one device vector type declares."""
    return {"float2": 2, "float3": 3, "float4": 4,
            "double2": 2, "double3": 3, "double4": 4,
            "int2": 2, "int3": 3, "int4": 4,
            "uint2": 2, "uint3": 3, "uint4": 4}[vec]


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


def parameter_list(param_types:Tuple[str, ...], param_names:List[str])->str:
    """The parameter list one helper shape declares, spelled the way C reads it."""
    return ", ".join(f"{ptype} {pname}"
                     for ptype, pname in zip(param_types, param_names))


def _definition(ret:str, name:str, param_types:Tuple[str, ...],
                param_names:List[str], body:str)->str:
    """One whole `__device__` definition around the return expression given."""
    return (f"__device__ {ret} {helper_name(name, param_types)}("
            f"{parameter_list(param_types, param_names)})\n"
            "{\n"
            f"    return {body};\n"
            "}\n")


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
    components:int = _component_count(vec)
    width:str = "double" if vec.startswith("double") else "float"
    if kind in ("any", "all"):
        # A reduction folds ACROSS the components, so it is not a per-component
        # term: the whole vector's components join into one test.
        joiner:str = ("|" if kind == "any" else "&")
        expr:str = f" {joiner} ".join(f"({access(0, c, param_names, shape)})"
                                      for c in range(components))
        return _definition("int", name, param_types, param_names,
                           f"({expr}) != 0")
    # The per-component access for each argument at one component: `a.x`, `a.y`,
    # ... for a vector argument, or the scalar's own text (a scalar parameter is one
    # value, shared by the components). The component letters are the field names
    # the device structs actually declare -- spelled out, because `w` does not sit
    # beside `z` in ASCII.
    terms:List[str] = []
    for component in range(components):
        parts:List[str] = [access(index, component, param_names, shape)
                           for index in range(len(shape))]
        terms.append(_term(name, spell, kind, parts, width))
    return _definition(ret, name, param_types, param_names,
                       f"make_{ret}({', '.join(terms)})")


def access(index:int, component:int, param_names:List[str], shape:Tuple[str, ...])->str:
    """One argument's value at one component: the scalar itself, or `a.x`."""
    if shape[index] != "v":
        return param_names[index]
    return f"{param_names[index]}.{'xyzw'[component]}"


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
    if kind == "select":
        # The value form python's spelling means: the condition picks one operand
        # per component. (OpenCL's native `select` takes a mask and answers -1
        # bits for a true lane, which is why the OpenCL path keeps its own
        # emission instead of this helper.)
        return f"(({parts[2]}) ? ({parts[1]}) : ({parts[0]}))"
    if kind == "absdiff":
        return f"abs(({parts[0]}) - ({parts[1]}))"
    if kind == "hadd":
        return f"((({parts[0]}) & ({parts[1]})) + ((({parts[0]}) ^ ({parts[1]})) >> 1))"
    if kind == "rhadd":
        return f"((({parts[0]}) | ({parts[1]})) - ((({parts[0]}) ^ ({parts[1]})) >> 1))"
    if kind == "mad":
        return f"(({parts[0]}) * ({parts[1]}) + ({parts[2]}))"
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


@lru_cache(maxsize=None)
def geometric_source(name:str, vec:str, ret:str,
                     param_types:Tuple[str, ...])->str:
    """The `__device__` definition one geometric helper needs.

    The family (`dot`, `length`, `distance`, `normalize`, `cross`) cannot be built
    one component at a time -- the answer is a scalar summed over every component,
    or a formula that reads two components of one argument in the same term (the
    cross product) -- so the body is composed whole here, out of the same formulas
    the inline expansion spelled at the call site. `ret` is the declared return
    type: the element type for `dot` (an `int` vector's product stays an `int`), a
    `float` for the two `sqrt` reductions, and the vector itself for `normalize`
    and `cross`. The parameters are named `a`, `b` in order.
    """
    param_names:List[str] = ["a", "b"][:len(param_types)]
    components:int = _component_count(vec)
    width:str = "double" if vec.startswith("double") else "float"
    shape:Tuple[str, ...] = tuple("v" for _ in param_types)
    def component(parameter:str, index:int)->str:
        return access(0 if parameter == param_names[0] else 1, index,
                      param_names, shape)
    def dot_sum(first:str, second:str)->str:
        """The dot-product sum of two arguments: one product term per component."""
        return "(" + " + ".join(
            f"(({component(first, c)}) * ({component(second, c)}))"
            for c in range(components)) + ")"
    body:str
    if name == "dot":
        body = dot_sum(param_names[0], param_names[1])
    elif name == "length":
        body = f"sqrtf({dot_sum(param_names[0], param_names[0])})"
    elif name == "distance":
        body = ("sqrtf((" + " + ".join(
            f"((({component(param_names[0], c)}) - ({component(param_names[1], c)}))"
            f" * (({component(param_names[0], c)}) - ({component(param_names[1], c)})))"
            for c in range(components)) + "))")
    elif name == "normalize":
        def scaled(c:int)->str:
            return (f"(({component(param_names[0], c)})"
                    f" * rsqrtf({dot_sum(param_names[0], param_names[0])}))")
        body = f"make_{vec}({', '.join(scaled(c) for c in range(components))})"
    elif name == "cross":
        # The formula reads two components of one argument in the same term, so it
        # is spelled here per component, exactly as the inline expansion wrote it:
        # (a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x).
        terms:List[str] = [f"(({component(param_names[0], 1)}) * ({component(param_names[1], 2)})"
                           f" - ({component(param_names[0], 2)}) * ({component(param_names[1], 1)}))",
                           f"(({component(param_names[0], 2)}) * ({component(param_names[1], 0)})"
                           f" - ({component(param_names[0], 0)}) * ({component(param_names[1], 2)}))",
                           f"(({component(param_names[0], 0)}) * ({component(param_names[1], 1)})"
                           f" - ({component(param_names[0], 1)}) * ({component(param_names[1], 0)}))"]
        if components == 4:
            # A 4-component cross leaves the last slot at zero, the way the
            # expansion did; the constant takes the width.
            terms.append(_literal("0.0", width))
        body = f"make_{vec}({', '.join(terms)})"
    else:
        raise ValueError(f"no geometric helper body for {name!r}")
    return _definition(ret, name, param_types, param_names, body)


def helper_call(call_name:str, arg_codes:List[str])->str:
    """The call text one site emits: every argument expression appears exactly once."""
    return f"{call_name}({', '.join(arg_codes)})"
