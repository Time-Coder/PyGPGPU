# The scalar entries of the AD rule table: arithmetic, the math vocabulary, and the
# names the device carries no partial for.
#
# Every entry is keyed by the *object* a kernel body resolves to, exactly like
# ``builtin_registry`` -- ``from pygpgpu import sqrt``, ``math.sqrt(x)`` and ``np.sqrt(x)``
# are three objects and each gets its own entry, because a call is differentiable when the
# callee resolves to an object that has a rule, never because its spelling matches.
from __future__ import annotations

import ast
import builtins as _py
import math as _math

import numpy as _np

from ..ast_support import (
    call,
    constant,
    ifexp,
    name,
    negate,
    wrap,
)
from ...kernel import builtins as _gpu
from ...kernel import sci_math as _sci
from ...vec_types import funcs as _vf
from . import DiffRule, refuse, register

# The names the partial expressions may spell, mapped to the objects they must resolve to.
# The adjoint namespace gets all of them (under mangled spellings, so a user's own `sqrt`
# helper cannot shadow one), so a rule works whether or not the user's module imported it.
INJECTS:dict = {
    "sqrt": _vf.sqrt, "exp": _vf.exp, "exp2": _vf.exp2, "log": _vf.log,
    "sin": _vf.sin, "cos": _vf.cos, "tanh": _vf.tanh, "sign": _vf.sign,
    "rint": _sci.rint, "trunc": _math.trunc,
    "erf_inverse": _sci.erf_inverse, "normal_cdf_inverse": _sci.normal_cdf_inverse,
}


class _F:
    """A rule-provided callee: calling it builds the call and names the injected spelling.

    The adjoint namespace carries every INJECTS name under a mangled spelling, so a rule's
    ``sqrt(u)`` resolves to the package's own device builtin even when the user's module
    has its own ``sqrt``.
    """

    def __init__(self, spell:str):
        if spell not in INJECTS:
            raise KeyError(f"AD rule spells {spell!r}, which INJECTS does not carry")
        self.spell:str = spell

    def __call__(self, *args:object)->ast.expr:
        return call(name(self.spell), [wrap(a) for a in args])


def _c(value:float)->"constant":
    return constant(value)


def _u(args:list)->"name":
    return args[0]


def _v(args:list)->"name":
    return args[1]


def _w(args:list)->"name":
    return args[2]


def _rule(obj:object, partials, arity:"int | None" = None)->None:
    """Register one differentiable object with its partial builder."""
    register(obj, DiffRule(partials, arity=arity))


def _one(obj:object, partial)->None:
    """A one-argument rule; the builder takes the argument LIST, like every other."""
    _rule(obj, lambda args: [partial(args)], arity=1)


def _two(obj:object, first, second)->None:
    """A two-argument rule with both partials given as builders."""
    _rule(obj, lambda args: [first(args), second(args)], arity=2)


def _refuse(module:object, names:list, reason:str)->None:
    """Register a refusal for every name a module publishes under it."""
    for spell in names:
        obj:object = getattr(module, spell, None)
        if obj is not None:
            refuse(obj, reason)


# --------------------------------------------------------------- partial builders

def _smooth_t(args:list)->"name":
    t:"name" = (_w(args) - _u(args)) / (_v(args) - _u(args))
    return ifexp(t < _c(0.0), _c(0.0), ifexp(t > _c(1.0), _c(1.0), t))


def _hypot_partials(args:list)->list:
    root:"name" = _F("sqrt")(_u(args) * _u(args) + _v(args) * _v(args))
    return [_u(args) / root, _v(args) / root]


def _recip_hypot_partials(args:list)->list:
    root:"name" = _F("sqrt")(_u(args) * _u(args) + _v(args) * _v(args))
    cube:"name" = root * root * root
    return [negate(_u(args)) / cube, negate(_v(args)) / cube]


def _clip_inside(args:list)->"name":
    lo:object = args[1]
    hi:object = args[2]
    inside:"name" = _c(1.0)
    if not _is_none(lo):
        inside = inside * (lo <= _u(args))
    if not _is_none(hi):
        inside = inside * (_u(args) <= hi)
    return inside


def _clip_low(args:list)->"name":
    if _is_none(args[1]):
        return None
    return (_u(args) < args[1])


def _clip_high(args:list)->"name":
    if _is_none(args[2]):
        return None
    return (_u(args) > args[2])


def _is_none(node:object)->bool:
    return isinstance(node, ast.Constant) and node.value is None


def _erf_partials(args:list)->list:
    return [_c(1.1283791670955126) * _F("exp")(negate(_u(args) * _u(args)))]


def _pow_partials(args:list)->list:
    """`u ** c` differentiates; a variable exponent is refused for its `ln u` domain."""
    base:"name" = _u(args)
    exponent:object = _v(args)
    if isinstance(exponent, ast.Constant) and isinstance(exponent.value, (int, float)) \
            and not isinstance(exponent.value, bool):
        return [constant(exponent.value) * (base ** constant(exponent.value - 1)), None]
    return [None, None]


def _powr_partials(args:list)->list:
    """`u ** v` on positive `u`: both partials are closed forms over `u` and `v`."""
    return [_v(args) * (_u(args) ** (_v(args) - _c(1.0))),
            (_u(args) ** _v(args)) * _F("log")(_u(args))]


# --------------------------------------------------------------- square roots

def _sqrt_half(a):
    return _c(1.0) / (_c(2.0) * _F("sqrt")(_u(a)))
_one(_math.sqrt, _sqrt_half)
_one(_vf.sqrt, _sqrt_half)
_one(_np.sqrt, _sqrt_half)
_one(_sci.native_sqrt, _sqrt_half)
_one(_sci.exact_sqrt, _sqrt_half)
_one(_sci.exact_sqrt_rz, _sqrt_half)
_one(_vf.rsqrt, lambda a: negate(_c(0.5) / (_F("sqrt")(_u(a)) * _u(a))))
_one(_sci.native_rsqrt, lambda a: negate(_c(0.5) / (_F("sqrt")(_u(a)) * _u(a))))
def _cbrt_partial(a):
    return _c(1.0) / (_c(3.0) * (_u(a) ** _c(2.0 / 3.0)))
_one(_vf.cbrt, _cbrt_partial)
_one(_math.cbrt, _cbrt_partial)

# --------------------------------------------------------------- exp / log

_one(_math.exp, lambda a: _F("exp")(_u(a)))
_one(_vf.exp, lambda a: _F("exp")(_u(a)))
_one(_np.exp, lambda a: _F("exp")(_u(a)))
_one(_sci.native_exp, lambda a: _F("exp")(_u(a)))
_one(_sci.fast_exp, lambda a: _F("exp")(_u(a)))
_one(_vf.exp2, lambda a: _c(_math.log(2.0)) * _F("exp2")(_u(a)))
_one(_sci.native_exp2, lambda a: _c(_math.log(2.0)) * _F("exp2")(_u(a)))
def _exp10(a):
    return _c(_math.log(10.0)) * _F("exp2")(_u(a) * _c(_math.log2(10.0)))
_one(_sci.native_exp10, _exp10)
_one(_sci.fast_exp10, _exp10)
_one(_math.expm1, lambda a: _F("exp")(_u(a)))
_one(_math.log, lambda a: _c(1.0) / _u(a))
_one(_vf.log, lambda a: _c(1.0) / _u(a))
_one(_np.log, lambda a: _c(1.0) / _u(a))
_one(_sci.native_log, lambda a: _c(1.0) / _u(a))
def _log2(a):
    return negate(_c(1.0) / (_u(a) * _c(_math.log(2.0))))
_one(_math.log2, _log2)
_one(_vf.log2, _log2)
_one(_np.log2, _log2)
_one(_sci.native_log2, _log2)
_one(_sci.fast_log2, _log2)
def _log10(a):
    return negate(_c(1.0) / (_u(a) * _c(_math.log(10.0))))
_one(_math.log10, _log10)
_one(_vf.log10, _log10)
_one(_np.log10, _log10)
_one(_sci.fast_log10, _log10)
_one(_math.log1p, lambda a: _c(1.0) / (_u(a) + _c(1.0)))
_one(_sci.fast_log, lambda a: _c(1.0) / _u(a))

# --------------------------------------------------------------- trigonometry

_one(_math.sin, lambda a: _F("cos")(_u(a)))
_one(_vf.sin, lambda a: _F("cos")(_u(a)))
_one(_np.sin, lambda a: _F("cos")(_u(a)))
_one(_sci.native_sin, lambda a: _F("cos")(_u(a)))
_one(_sci.fast_sin, lambda a: _F("cos")(_u(a)))
_one(_math.cos, lambda a: negate(_F("sin")(_u(a))))
_one(_vf.cos, lambda a: negate(_F("sin")(_u(a))))
_one(_np.cos, lambda a: negate(_F("sin")(_u(a))))
_one(_sci.native_cos, lambda a: negate(_F("sin")(_u(a))))
_one(_sci.fast_cos, lambda a: negate(_F("sin")(_u(a))))
def _tan(a):
    return _c(1.0) / (_F("cos")(_u(a)) * _F("cos")(_u(a)))
_one(_math.tan, _tan)
_one(_vf.tan, _tan)
_one(_np.tan, _tan)
_one(_sci.native_tan, _tan)
_one(_sci.fast_tan, _tan)
def _asin(a):
    return _c(1.0) / _F("sqrt")(_c(1.0) - _u(a) * _u(a))
_one(_math.asin, _asin)
_one(_vf.asin, _asin)
_one(_np.arcsin, _asin)
def _acos(a):
    return negate(_c(1.0) / _F("sqrt")(_c(1.0) - _u(a) * _u(a)))
_one(_math.acos, _acos)
_one(_vf.acos, _acos)
_one(_np.arccos, _acos)
def _atan(a):
    return _c(1.0) / (_c(1.0) + _u(a) * _u(a))
_one(_math.atan, _atan)
_one(_vf.atan, _atan)
_one(_np.arctan, _atan)
def _atan2(a):
    return [
    _v(a) / (_u(a) * _u(a) + _v(a) * _v(a)),
    negate(_u(a) / (_u(a) * _u(a) + _v(a) * _v(a)))]
_two(_math.atan2, lambda a: _atan2(a)[0], lambda a: _atan2(a)[1])
_two(_vf.atan2, lambda a: _atan2(a)[0], lambda a: _atan2(a)[1])
_two(_np.arctan2, lambda a: _atan2(a)[0], lambda a: _atan2(a)[1])

# sinh/cosh/tanh: cosh and sinh are each other's derivatives, so both names are injected.
_one(_math.sinh, lambda a: _F("cosh")(_u(a)))
_one(_vf.sinh, lambda a: _F("cosh")(_u(a)))
_one(_np.sinh, lambda a: _F("cosh")(_u(a)))
_one(_math.cosh, lambda a: _F("sinh")(_u(a)))
_one(_vf.cosh, lambda a: _F("sinh")(_u(a)))
_one(_np.cosh, lambda a: _F("sinh")(_u(a)))
def _tanh(a):
    return _c(1.0) - _F("tanh")(_u(a)) * _F("tanh")(_u(a))
_one(_math.tanh, _tanh)
_one(_vf.tanh, _tanh)
_one(_np.tanh, _tanh)
_one(_sci.sin_pi, lambda a: _c(_math.pi) * _F("cos")(_u(a) * _c(_math.pi)))
_one(_sci.cos_pi, lambda a: negate(_c(_math.pi) * _F("sin")(_u(a) * _c(_math.pi))))

# --------------------------------------------------------------- abs / min / max / clamp

_one(_math.fabs, lambda a: _F("sign")(_u(a)))
_one(_vf.fabs, lambda a: _F("sign")(_u(a)))
_one(_np.abs, lambda a: _F("sign")(_u(a)))
_one(_py.abs, lambda a: _F("sign")(_u(a)))
_one(_vf.sign, lambda a: _c(0.0))
def _minmax(a):
    return [(_u(a) <= _v(a)), (_v(a) < _u(a))]
_two(_sci.fmin, lambda a: _minmax(a)[0], lambda a: _minmax(a)[1])
_two(_py.min, lambda a: _minmax(a)[0], lambda a: _minmax(a)[1])
_two(_gpu.min, lambda a: _minmax(a)[0], lambda a: _minmax(a)[1])
_two(_np.minimum, lambda a: _minmax(a)[0], lambda a: _minmax(a)[1])
def _maxmin(a):
    return [(_u(a) >= _v(a)), (_v(a) > _u(a))]
_two(_sci.fmax, lambda a: _maxmin(a)[0], lambda a: _maxmin(a)[1])
_two(_py.max, lambda a: _maxmin(a)[0], lambda a: _maxmin(a)[1])
_two(_gpu.max, lambda a: _maxmin(a)[0], lambda a: _maxmin(a)[1])
_two(_np.maximum, lambda a: _maxmin(a)[0], lambda a: _maxmin(a)[1])
_two(_vf.saturate, lambda a: (_c(0.0) < _u(a)) * (_u(a) < _c(1.0)), lambda a: None)
def _three(obj:object, first, second, third)->None:
    """A three-argument rule with all three partials given as builders."""
    _rule(obj, lambda a: [first(a), second(a), third(a)], arity=3)
_three(_vf.clamp, lambda a: ((_v(a) <= _u(a)) * (_u(a) <= _w(a))),
       lambda a: (_u(a) < _v(a)), lambda a: (_u(a) > _w(a)))
_three(_np.clip, lambda a: _clip_inside(a), lambda a: _clip_low(a), lambda a: _clip_high(a))
_three(_vf.mix, lambda a: (_c(1.0) - _w(a)), lambda a: _w(a), lambda a: (_v(a) - _u(a)))
_three(_vf.mad, lambda a: _v(a), lambda a: _u(a), lambda a: _c(1.0))
_three(_sci.exact_fma, lambda a: _v(a), lambda a: _u(a), lambda a: _c(1.0))
_three(_vf.smoothstep, lambda a: None, lambda a: None,
       lambda a: (_c(6.0) * _smooth_t(a) * (_c(1.0) - _smooth_t(a))) / (_w(a) - _u(a)))
_three(_vf.select, lambda a: None, lambda a: ifexp(_u(a), _c(1.0), _c(0.0)),
       lambda a: ifexp(_u(a), _c(0.0), _c(1.0)))
_two(_sci.fdim, lambda a: (_u(a) > _v(a)),
     lambda a: (_u(a) > _v(a)) * negate(_c(1.0)))
_two(_sci.remainder, lambda a: _c(1.0),
     lambda a: negate(_F("rint")(_u(a) / _v(a))))
_two(_math.fmod, lambda a: _c(1.0), lambda a: negate(_F("trunc")(_u(a) / _v(a))))
_two(_vf.fmod, lambda a: _c(1.0), lambda a: negate(_F("trunc")(_u(a) / _v(a))))
_two(_sci.copysign, lambda a: _F("sign")(_v(a)) * _F("sign")(_u(a)),
     lambda a: None)
_two(_math.copysign, lambda a: _F("sign")(_v(a)) * _F("sign")(_u(a)),
     lambda a: None)
_two(_math.hypot, lambda a: _hypot_partials(a)[0], lambda a: _hypot_partials(a)[1])
_two(_vf.hypot, lambda a: _hypot_partials(a)[0], lambda a: _hypot_partials(a)[1])
_two(_sci.reciprocal_hypot,
     lambda a: _recip_hypot_partials(a)[0], lambda a: _recip_hypot_partials(a)[1])
def _divide(a):
    return [_c(1.0) / _v(a), negate(_u(a)) / (_v(a) * _v(a))]
_two(_sci.native_divide, lambda a: _divide(a)[0], lambda a: _divide(a)[1])
_two(_sci.fast_divide, lambda a: _divide(a)[0], lambda a: _divide(a)[1])
_two(_sci.exact_divide, lambda a: _divide(a)[0], lambda a: _divide(a)[1])
_two(_sci.exact_divide_rz, lambda a: _divide(a)[0], lambda a: _divide(a)[1])
def _recip(a):
    return negate(_c(1.0)) / (_u(a) * _u(a))
_one(_sci.native_recip, _recip)
_one(_sci.exact_recip, _recip)
_one(_sci.exact_recip_rz, _recip)
_two(_sci.exact_add, lambda a: _c(1.0), lambda a: _c(1.0))
_two(_sci.exact_sub, lambda a: _c(1.0), lambda a: negate(_c(1.0)))
_two(_sci.exact_mul, lambda a: _v(a), lambda a: _u(a))
_two(_vf.ldexp, lambda a: _F("exp2")(_v(a)), lambda a: None)
_two(_math.ldexp, lambda a: _F("exp2")(_v(a)), lambda a: None)
_one(_math.degrees, lambda a: _c(180.0 / _math.pi))
_one(_math.radians, lambda a: _c(_math.pi / 180.0))
_one(_vf.fract, lambda a: _c(1.0))
_two(_py.pow, lambda a: _pow_partials(a)[0], lambda a: _pow_partials(a)[1])
_two(_math.pow, lambda a: _pow_partials(a)[0], lambda a: _pow_partials(a)[1])
_two(_vf.pow, lambda a: _pow_partials(a)[0], lambda a: _pow_partials(a)[1])
_two(_np.power, lambda a: _pow_partials(a)[0], lambda a: _pow_partials(a)[1])
_two(_sci.native_powr, lambda a: _powr_partials(a)[0], lambda a: _powr_partials(a)[1])
_two(_sci.fast_pow, lambda a: _powr_partials(a)[0], lambda a: _powr_partials(a)[1])
_one(_py.float, lambda a: _c(1.0))
_one(_math.erf, _erf_partials)
_one(_sci.erf, _erf_partials)
_one(_math.erfc, lambda a: negate(_erf_partials([_u(a)])))
_one(_sci.erfc, lambda a: negate(_erf_partials([_u(a)])))
_one(_sci.erf_inverse,
     lambda a: _c(_math.sqrt(_math.pi) / 2.0) * _F("exp")(_F("erf_inverse")(_u(a)) ** _c(2.0)))
_one(_sci.normal_cdf,
     lambda a: _c(0.3989422804014327) * _F("exp")(negate(_c(0.5) * _u(a) * _u(a))))
_one(_sci.normal_cdf_inverse,
     lambda a: _c(_math.sqrt(2.0 * _math.pi)) * _F("exp")(
         _F("normal_cdf_inverse")(_u(a)) ** _c(2.0) / _c(2.0)))

# --------------------------------------------------------------- refusals, by name

_PIECEWISE = ("is piecewise-constant: its derivative is zero almost everywhere and "
              "undefined at the breaks, which is a plausible wrong answer, not a gradient")
_INTEGER = "works on integers: a gradient through it is not representable"
_BOOL = ("answers a bool: a comparison propagates no tangent, so there is nothing to "
         "differentiate")
_VECTOR = ("is a vector operation: its adjoint is a component rule this scalar table does "
           "not carry (the vector-rules milestone)")
_NO_SPECIAL = ("does not have a closed-form partial the device carries; the AD stage "
               "refuses it by name instead of answering a plausible number")
_SIDE_EFFECT = ("writes launch-wide state that is not the declared loss, so "
                "differentiating it would double or race the side effect")

_refuse(_math, ["floor", "ceil", "trunc"], _PIECEWISE)
_refuse(_py, ["round", "divmod"], _PIECEWISE)
_refuse(_vf, ["floor", "ceil", "trunc"], _PIECEWISE)
_refuse(_sci, ["rint"], _PIECEWISE)
_refuse(_sci, ["float_to_int_nearest", "float_to_int_zero", "float_to_int_down",
               "float_to_int_up", "double_to_int_nearest", "double_to_int_zero",
               "double_to_int_down", "double_to_int_up"], _PIECEWISE)
_refuse(_py, ["int", "bool"], _INTEGER)
_refuse(_sci, ["int_to_float", "uint_to_float"], _INTEGER)
_refuse(_vf, ["step"], _PIECEWISE)
_refuse(_vf, ["isequal", "isgreater", "isgreaterequal", "isless", "islessequal",
              "isnotequal", "isordered", "isunordered", "any", "all"], _BOOL)
_refuse(_math, ["isnan", "isinf", "isfinite"], _BOOL)
_refuse(_np, ["isnan", "isinf", "isfinite", "signbit"], _BOOL)
_refuse(_vf, ["dot", "cross", "length", "distance", "normalize", "abs_diff",
              "hadd", "rhadd"], _VECTOR)
_refuse(_sci, ["tgamma", "lgamma", "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1",
               "bessel_i0", "bessel_i1", "nextafter"], _NO_SPECIAL)
_refuse(_gpu, ["atomic_add", "atomic_sub", "atomic_and", "atomic_or", "atomic_xor",
               "atomic_min", "atomic_max", "atomic_xchg", "atomic_cas"], _SIDE_EFFECT)
_refuse(_gpu, ["local_array", "local_array_per_group", "shared_array",
               "shared_array_per_block"],
        ("builds work-group storage: its gradient slots would need a shared-memory story "
         "per backend, which is a separate feature"))
_refuse(_gpu, ["barrier", "get_global_id", "get_local_id", "get_group_id",
               "get_local_size", "get_global_size", "get_num_groups"],
        ("is a work-item or dispatch fact, not a differentiable quantity: it has no "
         "partial to take"))
_refuse(_gpu, ["mul24", "mad24"],
        ("is named after a hardware instruction and OpenCL leaves it undefined outside "
         "24-bit inputs; the AD stage refuses it by name"))
_refuse(_gpu, ["print"], "writes from the device; the translator refuses it too")
