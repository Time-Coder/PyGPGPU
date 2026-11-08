# The call vocabulary: the backend spellings and the KERNEL_TRANSLATING switch.
from __future__ import annotations
import ast
from typing import (Dict, List, Optional)



def _math_device_name(attr:str, backend:str, result_type:object=None)->str:
    """The device spelling of a `math.*` function.

    OpenCL's C dialect has float overloads under the plain name. CUDA has both the
    double and the `f`-suffixed float form, and asking for the plain name with a float
    argument is ambiguous whenever the argument's type is not exactly `float` -- which is
    how `floor` came to fail with "more than one instance of overloaded function".

    The caller's *result* type decides which of the two CUDA forms to spell: `ret` is
    already computed from the argument types (`_emit_math_call` widens it to `double`
    when any argument is one), so a double-width call gets the plain name and the
    compiler resolves the double overload. Spelling `sinf` with a double argument
    silently cut a `float=np.float64` kernel's transcendentals back to float32.
    """
    if backend != "cuda":
        return attr
    # The three predicates are already type-generic in CUDA and take no `f` suffix.
    if attr in ("isnan", "isinf", "isfinite"):
        return attr
    if result_type == "double":
        return attr
    return attr if attr.endswith("f") else attr + "f"


# The name of the compile-time switch (see `builtins.KERNEL_TRANSLATING`), and the value a
# translation sees. Python resolves the same name to False, which is what lets one body run
# differently on each side without the translator guessing at it.
KERNEL_TRANSLATING_NAME:str = "KERNEL_TRANSLATING"
KERNEL_TRANSLATING_VALUE:bool = True


def _translation_branch(node:ast.expr, body:List[ast.stmt],
                        orelse:List[ast.stmt])->Optional[List[ast.stmt]]:
    """The branch a translation keeps for an ``if`` about ``KERNEL_TRANSLATING``.

    Returns None when the condition says nothing about translation (an ordinary device
    ``if``), and otherwise the statements the device side should contain -- an empty list
    when the condition is false, so the whole ``if`` disappears.
    """
    truth:Optional[bool] = _translation_truth(node)
    if truth is None:
        return None
    return list(body) if truth else list(orelse)


def _translation_truth(node:ast.expr)->Optional[bool]:
    """The compile-time truth of a condition about `KERNEL_TRANSLATING`, or None.

    None means the condition says nothing about translation, so it is left to the device.

    **The trigger is the name, not the object it resolves to.** A body that declares its own
    `KERNEL_TRANSLATING` -- a module global, a local, anything that reads as that name --
    gets the same conditional translation, because a kernel author writing that name is
    asking for exactly this and requiring an import would only make the spelling longer. It
    is also why this works on source text rather than on a resolved value: the translator
    never has to import anything to decide the branch.

    The constant is False in python and True while translating, so a condition is decidable
    when it is built from that name with `not`, `and`, `or` and boolean literals --
    each of which python evaluates with short-circuiting. `if KERNEL_TRANSLATING:` is
    therefore the device branch and `if not KERNEL_TRANSLATING:` is python's.
    """
    if isinstance(node, ast.Name) and node.id == KERNEL_TRANSLATING_NAME:
        return KERNEL_TRANSLATING_VALUE
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        # A boolean literal says the same thing on both sides, so it is decided here and a
        # branch around it collapses (KERNEL_TRANSLATING and True is just the constant).
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner:Optional[bool] = _translation_truth(node.operand)
        return None if inner is None else not inner
    if isinstance(node, ast.BoolOp):
        values:List[Optional[bool]] = [_translation_truth(value) for value in node.values]
        if any(value is None for value in values):
            return None
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        return None
    return None


_MATH_FUNCS:Dict[str, Optional[str]] = {
    "sqrt": None, "exp": None, "log": None, "log2": None, "log10": None,
    "sin": None, "cos": None, "tan": None, "asin": None, "acos": None,
    "atan": None, "atan2": None, "sinh": None, "cosh": None, "tanh": None,
    "fabs": None, "pow": None, "hypot": None, "erf": None,
    "log1p": None, "expm1": None, "fmod": None, "cbrt": None,
    "copysign": None, "ldexp": None, "degrees": None, "radians": None,
    # python returns an int here; the *argument* stays a float (see _emit_math_call, where
    # casting the argument made `ceil(2.1)` answer 2).
    "floor": "int", "ceil": "int", "trunc": "int",
    # python returns a bool; the device answers 0/1, which is the same number.
    "isnan": "int", "isinf": "int", "isfinite": "int",
}

_MATH_CONSTS:Dict[str, str] = {
    "pi": "3.14159265358979323846f",
    "e": "2.71828182845904523536f",
    "tau": "6.28318530717958647692f",
    # A literal rather than the INFINITY macro: NVRTC's default headers do not define it,
    # so the CUDA translation unit failed with "identifier INFINITY is undefined". A float
    # literal is the same value on every backend and needs no header.
    "inf": "(1.0f / 0.0f)",
}


# Which of the math vocabulary's names map to which of the complex header's functions, which
# answer a *real* (the magnitude and the phase), and which have no complex meaning at all.
# numpy is the authority here: it defines exactly this split, the interpreter backend runs
# numpy, and the device has to agree with it rather than inventing an answer.
COMPLEX_DEVICE_FUNCTIONS:Dict[str, str] = {
    "sqrt": "sqrt", "exp": "exp", "log": "log", "log2": "log2", "log10": "log10",
    "sin": "sin", "cos": "cos", "tan": "tan",
    # numpy spells the inverse trig `arcsin`/`arccos`/`arctan`, and its `np.sin`-shaped
    # aliases (`np.asin`, ...) exist too; both spellings mean the same function here.
    "asin": "asin", "acos": "acos", "atan": "atan",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "square": "square", "reciprocal": "reciprocal", "conjugate": "conjugate",
    "conj": "conjugate", "exp2": "exp2",
    # `rint` answers a complex value in numpy (each part rounded to even), which is the same
    # rule the real path spells `rint`; `round` is a builtin and is refused for a complex
    # value (see COMPLEX_BUILTIN_REFUSALS), so a kernel writes `np.round(z)` for one.
    "rint": "rint",
}
# A complex argument answering a *real*: numpy's own rule (`abs` is the magnitude, `np.angle`
# the phase). An empty spelling means the plain real function of the magnitude.
COMPLEX_ABS_FUNCTIONS:Dict[str, str] = {
    "absolute": "abs", "abs": "abs", "angle": "angle", "sign": "",
}
# The real-only names, which numpy refuses for a complex input -- with the complex spelling
# that is the right counterpart, or the fact that there is none.
COMPLEX_UNSUPPORTED_FUNCTIONS:Dict[str, str] = {
    "floor": "there is none: a complex value has no ordering",
    "ceil": "there is none: a complex value has no ordering",
    "trunc": "there is none: a complex value has no ordering",
    "fmod": "use `a - b * floor(a / b)`, or np.mod for a real pair",
    "hypot": "use `np.abs(a - b)` on the difference",
    "cbrt": "use `np.pow(z, 1 / 3)`",
    "ldexp": "use `z * 2 ** exponent`",
    "atan2": "use `np.angle(a / b)`, or `np.atan(a / b)`",
}

# The *builtin* spellings -- pygf's functions, re-exported from the package top level -- that
# are a complex operation, and the header function each one calls. The interpreter answers
# these for a complex value (they are pygf's functions, which route a complex scalar through
# `cmath`), so the device answers them too: a kernel and its serial run have to agree. The
# table exists because the builtin tables are keyed by *type* and a complex value has no entry
# in them, which is how `sqrtf(z)` and `powf(z, w)` used to be emitted.
COMPLEX_BUILTIN_FUNCTIONS:Dict[str, str] = {
    "sqrt": "sqrt", "exp": "exp", "exp2": "exp2", "log": "log", "log2": "log2",
    "log10": "log10", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "conjugate": "conjugate", "conj": "conjugate", "pow": "pow",
    # `sign` answers a *complex* value (the direction on the unit circle), unlike `abs`,
    # which answers the magnitude -- so it belongs here and not beside `abs`.
    "sign": "sign",
    # `rint` rounds each part to even. This package's own host body answers that now
    # (sci_math rounds the two parts with python's `round`), which is what makes the device
    # allowed to: python's builtin `round` still refuses a complex value, and that is a
    # different name.
    "rint": "rint",
}

# The builtin names that answer a *real* for a complex argument: the magnitude, which is
# numpy's own rule for it.
COMPLEX_BUILTIN_REAL_RESULTS:Dict[str, str] = {"abs": "abs"}

# The builtin names numpy refuses for a complex value, and what to write instead. The
# interpreter refuses them as well -- they reach `math.floor` and friends through pygf -- so
# saying so is the only honest answer, and it keeps the device from emitting a call the
# compiler cannot type. `round` is here rather than in the table above because *python's*
# builtin refuses a complex value; numpy's `np.round` is the spelling that answers one.
COMPLEX_BUILTIN_REFUSALS:Dict[str, str] = {
    "floor": "there is none: a complex value has no ordering",
    "ceil": "there is none: a complex value has no ordering",
    "trunc": "there is none: a complex value has no ordering",
    "round": "use `np.round(z)`: python's round() refuses a complex value",
    "fract": "use `z - np.floor(z)`, which is a real-pair operation",
    "fmod": "use `a - b * np.floor(a / b)`, or np.mod for a real pair",
    "remainder": "there is none: numpy has no complex form for it",
    "rsqrt": "use `1 / np.sqrt(z)`",
    "cbrt": "use `np.power(z, 1 / 3)`",
    "ldexp": "use `z * 2 ** exponent`",
    "step": "there is none: a complex value has no ordering",
    "smoothstep": "there is none: a complex value has no ordering",
    "fdim": "there is none: numpy has no complex form for it",
    "copysign": "there is none: numpy has no complex form for it",
    "nextafter": "there is none: numpy has no complex form for it",
    "degrees": "there is none: numpy has no complex form for it",
    "radians": "there is none: numpy has no complex form for it",
    "hypot": "use `np.abs(a - b)` on the difference",
}


# printf specifier and argument cast per integer type. Two things differ between the
# backends: what a `long` is (64-bit in OpenCL C, but only 32-bit in CUDA/HIP on Windows)
# and which 64-bit conversions the device printf implements (AMD's OpenCL printf refuses
# "%lld", so OpenCL uses "%ld" with a long cast, which is exactly as wide there).
PRINT_INT_FORMATS:Dict[str, Dict[str, str]] = {
    "opencl": {
        "char": "d", "short": "d", "int": "d",
        "uchar": "u", "ushort": "u", "uint": "u",
        "long": "ld", "ulong": "lu",
    },
    "cuda": {
        "char": "d", "short": "d", "int": "d",
        "uchar": "u", "ushort": "u", "uint": "u",
        "long": "lld", "ulong": "llu",
    },
}

PRINT_INT_CASTS:Dict[str, Dict[str, str]] = {
    "opencl": {
        "char": "int", "short": "int", "int": "int",
        "uchar": "unsigned int", "ushort": "unsigned int", "uint": "unsigned int",
        "long": "long", "ulong": "unsigned long",
    },
    "cuda": {
        "char": "int", "short": "int", "int": "int",
        "uchar": "unsigned int", "ushort": "unsigned int", "uint": "unsigned int",
        "long": "long long", "ulong": "unsigned long long",
    },
}

# Python prints a float with the fewest significant digits that still read back as the same
# value. numpy's str() of a float32 switches to the exponent form from 1e6 up, and python's
# repr of a double from 1e16 up; both switch down to it below 1e-4 (0.0001 stays fixed,
# 0.00001 becomes 1e-05, i.e. from decimal exponent -5).
PRINT_REAL_SCI_HI:Dict[bool, int] = {True: 6, False: 16}
PRINT_REAL_SCI_LO:Dict[bool, int] = {True: -5, False: -5}

# Significant digits to try for a real: 9 round-trips every float32, 17 every double.
PRINT_REAL_DIGITS:Dict[bool, int] = {True: 9, False: 17}

# The largest magnitude the exponent form is needed for, per precision: a float32 that big
# is already close to its range, a double's is the top of its range.
PRINT_REAL_BIG:Dict[bool, str] = {True: "3.4028235e38", False: "1.7976931348623157e308"}

# Whole values below this magnitude print with a trailing ".0" rather than in the exponent
# form, matching python: numpy's str() of a float32 switches to the exponent form at 1e6, so
# a whole value from there up is written "1e+06" rather than "1000000.0".
PRINT_REAL_PLAIN_MAX:Dict[bool, str] = {True: "1e6", False: "1e16"}

# The five values python spells out rather than through a conversion: nan, inf, -inf, 0.0
# and -0.0. They own the first leaves of a real's layout (see _print_real_conversion).
PRINT_REAL_SPECIAL_COUNT:int = 5

# How many printf leaves one print() may expand into. Each bool argument doubles the count
# and each real multiplies it by its digit budget, so the cap keeps a single line from
# growing without bound. Ordinary lines stay far below it; one that trips it should be
# split into several print() calls. Keep this small: a huge switch of printf call sites is
# also a lot of device code for a debug facility, and some drivers handle it badly.
PRINT_MAX_LINES:int = 512
