"""The dual-sided device builtins: one function per name a kernel body may call.

This module holds the builtins and nothing else. On a device backend the translator
recognizes a call by *which function it resolves to* (see ``builtin_registry.py``) and
emits the matching OpenCL C or CUDA call, never looking at the body below. On the
``python`` backend the body is what runs, so it implements the same operation on host numpy
values -- that is why each one is written out rather than left as a placeholder.

The two sides must agree, which is why the implementations live here next to the names:
the signatures are what an editor shows, and the bodies are what the interpreter does.
Anything a body needs but a user should not see lives in ``host_runtime.py``; the device
side lives in ``device_builtins.py``.

Outside a kernel, the functions that depend on a work item (``get_global_id`` and friends)
raise a ``SyntaxError`` explaining that; the ones that act on a value (``atomic_add``,
``dot``, ...) simply operate on it.

Usage::

    from pygpgpu import atomic_add, get_global_id

    @kernel
    def k(out: np.ndarray):
        i = get_global_id(0)
        atomic_add(out, i, 1)"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..host_runtime import get_host_frame

from ...vec_types.funcs import abs_diff as abs_diff
from ...vec_types.funcs import all as all
from ...vec_types.funcs import any as any
from ...vec_types.funcs import clamp as clamp
from ...vec_types.funcs import cross as cross
from ...vec_types.funcs import distance as distance
from ...vec_types.funcs import dot as dot
from ...vec_types.funcs import exp2 as exp2
from ...vec_types.funcs import hadd as hadd
from ...vec_types.funcs import isequal as isequal
from ...vec_types.funcs import isgreater as isgreater
from ...vec_types.funcs import isgreaterequal as isgreaterequal
from ...vec_types.funcs import isless as isless
from ...vec_types.funcs import islessequal as islessequal
from ...vec_types.funcs import isnotequal as isnotequal
from ...vec_types.funcs import isordered as isordered
from ...vec_types.funcs import isunordered as isunordered
from ...vec_types.funcs import length as length
from ...vec_types.funcs import mad as mad
from ...vec_types.funcs import mad24 as mad24
from ...vec_types.funcs import mix as mix
from ...vec_types.funcs import mul24 as mul24
from ...vec_types.funcs import normalize as normalize
from ...vec_types.funcs import rhadd as rhadd
from ...vec_types.funcs import rsqrt as rsqrt
from ...vec_types.funcs import saturate as saturate
from ...vec_types.funcs import select as select
from ...vec_types.funcs import step as step
from ...vec_types.funcs import sqrt as sqrt
from ...vec_types.funcs import sin as sin
from ...vec_types.funcs import cos as cos
from ...vec_types.funcs import tan as tan
from ...vec_types.funcs import asin as asin
from ...vec_types.funcs import acos as acos
from ...vec_types.funcs import atan as atan
from ...vec_types.funcs import atan2 as atan2
from ...vec_types.funcs import sinh as sinh
from ...vec_types.funcs import cosh as cosh
from ...vec_types.funcs import tanh as tanh
from ...vec_types.funcs import exp as exp
from ...vec_types.funcs import log as log
from ...vec_types.funcs import log2 as log2
from ...vec_types.funcs import log10 as log10
from ...vec_types.funcs import pow as pow
from ...vec_types.funcs import floor as floor
from ...vec_types.funcs import ceil as ceil
from ...vec_types.funcs import trunc as trunc
from ...vec_types.funcs import fabs as fabs
from ...vec_types.funcs import fmod as fmod
from ...vec_types.funcs import hypot as hypot
from ...vec_types.funcs import cbrt as cbrt
from ...vec_types.funcs import ldexp as ldexp
from ...vec_types.funcs import fract as fract
from ...vec_types.funcs import sign as sign

from ..sci_math import erf as erf
from ..sci_math import erfc as erfc
from ..sci_math import tgamma as tgamma
from ..sci_math import lgamma as lgamma
from ..sci_math import rint as rint
from ..sci_math import remainder as remainder
from ..sci_math import fmin as fmin
from ..sci_math import fmax as fmax
from ..sci_math import fdim as fdim
from ..sci_math import copysign as copysign
from ..sci_math import nextafter as nextafter
from ..sci_math import native_recip as native_recip
from ..sci_math import native_divide as native_divide
from ..sci_math import native_sqrt as native_sqrt
from ..sci_math import native_rsqrt as native_rsqrt
from ..sci_math import native_exp as native_exp
from ..sci_math import native_log as native_log
from ..sci_math import native_sin as native_sin
from ..sci_math import native_cos as native_cos
from ..sci_math import native_tan as native_tan
from ..sci_math import native_exp2 as native_exp2
from ..sci_math import native_log2 as native_log2
from ..sci_math import native_exp10 as native_exp10
from ..sci_math import native_powr as native_powr
from ..sci_math import fast_exp as fast_exp
from ..sci_math import fast_log as fast_log
from ..sci_math import fast_log2 as fast_log2
from ..sci_math import fast_log10 as fast_log10
from ..sci_math import fast_sin as fast_sin
from ..sci_math import fast_cos as fast_cos
from ..sci_math import fast_tan as fast_tan
from ..sci_math import fast_exp10 as fast_exp10
from ..sci_math import fast_pow as fast_pow
from ..sci_math import fast_divide as fast_divide
from ..sci_math import exact_recip as exact_recip
from ..sci_math import exact_recip_rz as exact_recip_rz
from ..sci_math import exact_divide as exact_divide
from ..sci_math import exact_divide_rz as exact_divide_rz
from ..sci_math import exact_sqrt as exact_sqrt
from ..sci_math import exact_sqrt_rz as exact_sqrt_rz
from ..sci_math import exact_add as exact_add
from ..sci_math import exact_sub as exact_sub
from ..sci_math import exact_mul as exact_mul
from ..sci_math import exact_fma as exact_fma
from ..sci_math import erf_inverse as erf_inverse
from ..sci_math import normal_cdf as normal_cdf
from ..sci_math import normal_cdf_inverse as normal_cdf_inverse
from ..sci_math import reciprocal_hypot as reciprocal_hypot
from ..sci_math import sin_pi as sin_pi
from ..sci_math import cos_pi as cos_pi
from ..sci_math import bessel_j0 as bessel_j0
from ..sci_math import bessel_j1 as bessel_j1
from ..sci_math import bessel_y0 as bessel_y0
from ..sci_math import bessel_y1 as bessel_y1
from ..sci_math import bessel_i0 as bessel_i0
from ..sci_math import bessel_i1 as bessel_i1
from ..sci_math import float_to_int_nearest as float_to_int_nearest
from ..sci_math import float_to_int_zero as float_to_int_zero
from ..sci_math import float_to_int_down as float_to_int_down
from ..sci_math import float_to_int_up as float_to_int_up
from ..sci_math import int_to_float as int_to_float
from ..sci_math import uint_to_float as uint_to_float
from ..sci_math import double_to_int_nearest as double_to_int_nearest
from ..sci_math import double_to_int_zero as double_to_int_zero
from ..sci_math import double_to_int_down as double_to_int_down
from ..sci_math import double_to_int_up as double_to_int_up

from .work_item import (
    KERNEL_TRANSLATING,
    get_global_id,
    get_local_id,
    get_group_id,
    get_global_size,
    get_local_size,
    get_num_groups,
)
from .group_memory import barrier, zeros_per_group, zeros_per_block
from .atomic import (
    atomic_add,
    atomic_sub,
    atomic_min,
    atomic_max,
    atomic_and,
    atomic_or,
    atomic_xor,
    atomic_xchg,
    atomic_cas,
)

__all__ = [
    # compile-time switch: False in python, True while a kernel is being translated
    "KERNEL_TRANSLATING",
    # work-item indexing
    "get_global_id", "get_local_id", "get_group_id",
    "get_global_size", "get_local_size", "get_num_groups",
    # CUDA's index vectors (three axes each)
    "blockIdx", "threadIdx", "blockDim", "gridDim",
    # synchronisation and launch shape
    "barrier", "zeros_per_group", "zeros_per_block",
    # atomics
    "atomic_add", "atomic_sub", "atomic_min", "atomic_max",
    "atomic_and", "atomic_or", "atomic_xor", "atomic_xchg", "atomic_cas",
    # math and vector builtins (defined in pygpgpu.vec_types.funcs)
    "rsqrt", "exp2", "clamp", "mix", "mad", "saturate", "step", "select",
    "any", "all",
    "isequal", "isnotequal", "isgreater", "isgreaterequal",
    "isless", "islessequal", "isordered", "isunordered",
    "abs_diff", "hadd", "rhadd", "mad24", "mul24",
    "dot", "cross", "normalize", "length", "distance",
    # scalar math intrinsics (also from pygpgpu.vec_types.funcs)
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log2",
    "log10",
    "pow",
    "floor",
    "ceil",
    "trunc",
    "fabs",
    "fmod",
    "hypot",
    "cbrt",
    # the scientific-computing vocabulary (see sci_math.py): the functions both backends
    # have, and the ones only CUDA has
    "erf", "erfc", "tgamma", "lgamma", "rint", "remainder", "fmin", "fmax",
    "fdim", "copysign", "nextafter", "ldexp", "fract", "sign",
    "native_recip", "native_divide", "native_sqrt", "native_rsqrt", "native_exp",
    "native_log", "native_sin", "native_cos", "native_tan", "native_exp2", "native_log2",
    "native_exp10", "native_powr",
    "fast_exp", "fast_log", "fast_log2", "fast_log10", "fast_sin", "fast_cos",
    "fast_tan", "fast_exp10", "fast_pow", "fast_divide",
    "exact_recip", "exact_recip_rz", "exact_divide", "exact_divide_rz",
    "exact_sqrt", "exact_sqrt_rz", "exact_add", "exact_sub", "exact_mul", "exact_fma",
    "erf_inverse", "normal_cdf", "normal_cdf_inverse", "reciprocal_hypot",
    "sin_pi", "cos_pi",
    "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1", "bessel_i0", "bessel_i1",
    "float_to_int_nearest", "float_to_int_zero", "float_to_int_down", "float_to_int_up",
    "int_to_float", "uint_to_float",
    "double_to_int_nearest", "double_to_int_zero", "double_to_int_down", "double_to_int_up",
]


class _IndexVector:
    """An `int3` whose components are read from the current launch frame.

    Subscripting it (`blockIdx[0]`) and taking a component (`blockIdx.x`) both give the
    axis's number, so a kernel can use it however it used to spell that axis. The device
    side does not see this class at all: the translator recognizes the *name* and emits the
    vector directly (see `INDEX_VECTOR_BUILTINS`), and this only has to be right on the
    interpreter backend, where the body runs unchanged.

    The three components are declared for the editor and the type checker only: at runtime
    they come from `__getattr__`, because they read the launch frame and cannot be stored.
    Publishing them here is what makes `threadIdx.x` read as an `int` instead of `Any` -- a
    name a checker cannot see is a name every reader has to guess at.
    """

    if TYPE_CHECKING:
        x:int
        y:int
        z:int

    __slots__ = ("_readers",)

    def __init__(self, **readers:Callable[[], int])->None:
        object.__setattr__(self, "_readers", readers)

    def _value(self)->object:
        from pygf import int3

        return int3(*[reader() for reader in self._readers.values()])

    def __getitem__(self, axis:int)->int:
        readers:Dict[str, Callable[[], int]] = object.__getattribute__(self, "_readers")
        if not isinstance(axis, int) or axis < 0 or axis >= len(readers):
            raise IndexError(
                f"an index vector has three axes; {axis} is out of range")
        return list(readers.values())[axis]()

    def __getattr__(self, name:str)->int:
        readers:Dict[str, Callable[[], int]] = object.__getattribute__(self, "_readers")
        if name not in readers:
            raise AttributeError(
                f"an index vector has .x, .y and .z; it has no '{name}'")
        return readers[name]()

    def __setattr__(self, name:str, value:object)->None:
        raise TypeError(
            "an index vector is read-only: it reports the launch, and cannot be assigned to")

    def __repr__(self)->str:
        # Never evaluates the readers: this object is a registry key (see
        # `builtin_registry.BUILTIN_NAMES_BY_FUNCTION`), and hashing or printing it during
        # import must not need a launch frame.
        readers:object = object.__getattribute__(self, "_readers")
        axes:str = ", ".join(readers)
        return f"<index vector: {axes}>"

    def __hash__(self)->int:
        return id(self)


blockIdx:_IndexVector = _IndexVector(x=lambda: get_group_id(0), y=lambda: get_group_id(1),
                                     z=lambda: get_group_id(2))


threadIdx:_IndexVector = _IndexVector(x=lambda: get_local_id(0), y=lambda: get_local_id(1),
                                      z=lambda: get_local_id(2))


blockDim:_IndexVector = _IndexVector(x=lambda: get_local_size(0), y=lambda: get_local_size(1),
                                     z=lambda: get_local_size(2))


gridDim:_IndexVector = _IndexVector(x=lambda: get_num_groups(0), y=lambda: get_num_groups(1),
                                    z=lambda: get_num_groups(2))
