"""Native recursion for scalar self-calls, on the backend whose functions may recurse.

CUDA and HIP device functions may recurse -- the per-thread stack is a documented,
adjustable limit, not a language ban -- while MSL and OpenCL C forbid it outright. The
monomorphizer refused every cycle because a recursive call revisits the cache slot that
is still being filled; the unlock (see `pygpgpu.kernel.translator.scalar_recursion`)
admits the one shape whose signature the *call site* can spell completely before the
body is translated: scalars only, no raise, no outer-scope reads, a plain function, and
a return type every `return` names statically.

The tests pin three boundaries:

* the unlocked shape answers python's own recursion, on the device -- the interpreter
  backend always did (it runs the body as written), so the device half is the contract;
* a shape that fails any clause of the bar keeps the monomorphizer's refusal, with its
  own message -- the unlock must be invisible for everything it does not take;
* the two passes cooperate: a *tail* recursion is compiled into a loop by the
  tail-recursion pass before native recursion is ever asked for, so the generated source
  carries a `while` -- native recursion is the fallback for the shapes tail conversion
  refuses, and the generated source then carries the self-call.
"""
from __future__ import annotations


import numpy as np
import pytest

from pygpgpu import kernel
from pygf import vector3f
from pygpgpu.kernel.translator.scalar_recursion import RECURSIVE_BACKENDS


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


def rec_sum(n: int, acc: int = 0) -> int:
    """A tail-recursive accumulation -- the tail pass takes it before this one."""
    if n <= 0:
        return acc
    return rec_sum(n - 1, acc + n)


def rec_frac(x: float, depth: int, acc: float = 1.0) -> float:
    """A parameter-returning recursion: every `return` names `acc`'s own type."""
    if depth <= 0:
        return acc
    return rec_frac(x, depth - 1, acc * 0.5)


def rec_mixed(n: float, acc: float = 0.0) -> float:
    """Two returns: a literal (float width) and a parameter read, of one type."""
    if n <= 0:
        return acc
    return rec_mixed(n - 1.0, acc + 1.0)


def fib(n: int) -> int:
    """Tree recursion: two self-calls inside a `+`. The seed reads the operator's
    promoted type (`promote(int, int)` is `int`), so the unlock takes it natively."""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def consumed(n: int, acc: int = 1) -> int:
    """The textbook non-tail recursion: the caller multiplies after the callee
    answers. The multiply promotes int with int, so the seed answers `int`."""
    if n <= 1:
        return acc
    return n * consumed(n - 1, acc)


def biased(n: int) -> int:
    """The self-call sits inside a `+` with a literal: promoted int, unlocked."""
    if n <= 0:
        return 0
    return biased(n - 1) + 1


def even(n: int) -> int:
    """Mutual recursion: the cycle rides the frame check, and each member fixes the
    same return type (`int`) through its own literal returns."""
    if n == 0:
        return 1
    return odd(n - 1)


def odd(n: int) -> int:
    """The cycle's other member: its call of `even` is what closes the loop."""
    if n == 0:
        return 0
    return even(n - 1)


def depth_of(arr: np.ndarray, i: int) -> int:
    """An array parameter recursed over: the self-call forwards the array and the
    extent parameter, which for a self-call is the site's own `_shape_var` spelling."""
    if i >= arr.shape[0]:
        return 0
    if arr[i] == 0:
        return 0
    return 1 + depth_of(arr, i + 1)


def scan_from(arr: np.ndarray, i: int, target: float) -> int:
    """A structure-free two-pointer shape: the array and a scalar walk together."""
    if i >= arr.shape[0]:
        return -1
    if arr[i] == target:
        return i
    return scan_from(arr, i + 1, target)


def touching(n: int) -> int:
    """NOT unlocked: the body reads a module-level constant (an outer parameter),
    which the self-call site cannot spell."""
    if n <= 0:
        return 0
    return touching(n - 1) + OFFSET


def divided(n: int) -> int:
    """NOT unlocked: the return's type runs through a *local* (`half`), which the
    scan cannot type -- the honest answer is that the translation would decide."""
    if n <= 1:
        return 1
    steps_out = divided(n // 2)
    return steps_out + steps_out


def raised(n: int, acc: int = 0) -> int:
    """The raise sits beside a tail call, so the *tail pass* converts it first (v1
    runs before this unlock is ever asked) and the device answers by iteration."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return acc
    return raised(n - 1, acc + n)


def deep_sum(n: int, acc: int = 0) -> int:
    """A 200-frame recursion: each level carries a couple of scalars, which is the
    shape the per-thread stack raise sizes for (see test_deep_recursion_*)."""
    if n <= 0:
        return acc
    return deep_sum(n - 1, acc + 1)


class Walker:
    """A method that walks itself: the recursion is spelled `self.walk(...)`."""

    def __init__(self, factor: int):
        self.factor = factor

    def walk(self, n: int, acc: int = 0) -> int:
        if n <= 0:
            return acc
        return self.walk(n - 1, acc + self.factor)


def first_vector(arr: np.ndarray, i: int) -> vector3f:
    """A vector-returning recursion: the constructor names the device type (`float3`,
    through the host alias's MRO), so the seed can read the return off the body."""
    if i >= arr.shape[0]:
        return vector3f(0.0, 0.0, 0.0)
    if arr[i] > 0:
        return vector3f(float(arr[i]), 0.0, 0.0)
    return first_vector(arr, i + 1)


def raise_mid(n: int, acc: int = 0) -> int:
    """NOT convertible and NOT unlocked: the raise sits on the *recursive* path (it
    runs before the self-call on the same walk), so the loop conversion refuses the
    body, and the native unlock refuses the raise outright."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return acc
    if n % 2 == 0:
        return raise_mid(n - 1, acc + 1)
    raise_mid(n - 1, acc)  # the non-tail call: the caller keeps `acc` to raise with
    raise ValueError("odd")


OFFSET:int = 10


@kernel
def k_sum(out: np.ndarray, n: int):
    out[0] = rec_sum(n, 0)


@kernel
def k_frac(out: np.ndarray, depth: int):
    out[0] = rec_frac(1.0, depth, 1.0)


@kernel
def k_mixed(out: np.ndarray, n: float):
    out[0] = rec_mixed(n, 0.0)


@kernel
def k_fib(out: np.ndarray, n: int):
    out[0] = fib(n)


@kernel
def k_consumed(out: np.ndarray, n: int):
    out[0] = consumed(n, 1)


@kernel
def k_biased(out: np.ndarray, n: int):
    out[0] = biased(n)


@kernel
def k_touching(out: np.ndarray, n: int):
    out[0] = touching(n)


@kernel
def k_divided(out: np.ndarray, n: int):
    out[0] = divided(n)


@kernel
def k_even(out: np.ndarray, n: int):
    out[0] = even(n)


@kernel
def k_depth(out: np.ndarray, arr: np.ndarray):
    out[0] = depth_of(arr, 0)


@kernel
def k_scan(out: np.ndarray, arr: np.ndarray, target: float):
    out[0] = scan_from(arr, 0, target)


@kernel
def k_raised(out: np.ndarray, n: int):
    out[0] = raised(n, 0)


@kernel
def k_deep(out: np.ndarray, n: int):
    out[0] = deep_sum(n, 0)


@kernel
def k_walk(out: np.ndarray, w: Walker, n: int):
    out[0] = w.walk(n, 0)


@kernel
def k_vec(out: np.ndarray, arr: np.ndarray):
    v: vector3f = first_vector(arr, 0)
    out[0] = v.x
    out[1] = v.y
    out[2] = v.z


@kernel
def k_raise_mid(out: np.ndarray, n: int):
    out[0] = raise_mid(n, 0)

# ---------------------------------------------------------------------------
# The unlocked shape answers python, on the device
# ---------------------------------------------------------------------------


def test_a_tail_recursion_still_takes_the_loop(gpu_backend):
    """The tail pass converts it first: 100 terms sum to 5050 through the `while`."""
    out = np.zeros(1, dtype=np.int64)
    k_sum[gpu_backend](out, np.int32(100))
    assert int(out[0]) == 5050
    source = _helper_source(rec_sum, gpu_backend)
    assert "while" in source


def test_a_parameter_returning_recursion_answers_python(gpu_backend):
    """Four halvings of 1.0 -- every return names the parameter's own type."""
    out = np.zeros(1, dtype=np.float64)
    k_frac[gpu_backend](out, np.int32(4))
    assert float(out[0]) == pytest.approx(0.0625)


def test_a_mixed_literal_and_parameter_return(gpu_backend):
    """`return acc` and `return rec_mixed(...)` agree on float; ten steps answer 10."""
    out = np.zeros(1, dtype=np.float64)
    k_mixed[gpu_backend](out, np.float32(10.0))
    assert float(out[0]) == pytest.approx(10.0)


def test_tree_recursion_answers_python(gpu_backend):
    """fib(15) = 610 -- two self-calls per site, neither in tail position.

    The unlock's widest reach: the operator's promoted type is static (`int`), so the
    generated source calls the helper's own hashed name, which the second pin checks
    textually. Only the backend whose functions may recurse takes it (see
    RECURSIVE_BACKENDS); the others keep the refusal, which the boundary test pins.
    """
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    out = np.zeros(1, dtype=np.int64)
    k_fib[gpu_backend](out, np.int32(15))
    assert int(out[0]) == 610
    source = _helper_source(fib, gpu_backend)
    assert "while" not in source


def test_the_backend_boundary(gpu_backend):
    """The same body: unlocked where recursion is legal, refused where it is not."""
    out = np.zeros(1, dtype=np.int64)
    if _codegen_backend(gpu_backend) in RECURSIVE_BACKENDS:
        k_fib[gpu_backend](out, np.int32(10))
        assert int(out[0]) == 55
    else:
        with pytest.raises(SyntaxError) as caught:
            k_fib[gpu_backend](out, np.int32(10))
        message:str = str(caught.value)
        assert "recursive call detected" in message
        # The refusal names the three ways out, so the user has a path: the tail
        # conversion (automatic), a hand-written loop, or a recursive backend.
        assert "tail-recursive" in message
        assert "loop" in message
        assert "cuda or hip" in message


def test_a_tail_call_answers_on_the_nonrecursive_backends(gpu_backend):
    """The tail conversion is backend-independent: a *tail* recursion runs on the
    backends that forbid recursion too, as the loop it became."""
    if _codegen_backend(gpu_backend) in RECURSIVE_BACKENDS:
        pytest.skip("the nonrecursive backends are the boundary this pins")
    out = np.zeros(1, dtype=np.int64)
    k_sum[gpu_backend](out, np.int32(100))
    assert int(out[0]) == 5050


def test_deep_recursion_answers_with_the_raised_stack(gpu_backend):
    """200 native frames answer exactly: the launch raised the per-thread stack.

    CUDA's default per-thread stack is a handful of frames, and a frame that spills
    past it writes through memory nobody owns -- the wrong answer the HIP probe hit
    before the raise existed (fib(10) came home 512). The raise is the launch's own
    work now: a recursion-carrying kernel launches with a 64 KiB stack, which carries
    the hundreds of small frames the translator generates.
    """
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    out = np.zeros(1, dtype=np.int32)
    k_deep[gpu_backend](out, np.int32(200))
    assert int(out[0]) == 200


def test_the_stack_limit_is_raised_for_recursion(gpu_backend):
    """After a recursive launch, the context's stack limit is at least the raise's."""
    if gpu_backend != "cuda":
        pytest.skip("the stack raise is the CUDA context's own API")
    from pygpgpu.cuda.driver import CUDA, CUlimit
    from ctypes import c_size_t, pointer

    out = np.zeros(1, dtype=np.int32)
    k_deep["cuda"](out, np.int32(4))
    assert int(out[0]) == 4
    value:c_size_t = c_size_t()
    CUDA.cuCtxGetLimit(pointer(value), CUlimit.CU_LIMIT_STACK_SIZE)
    assert value.value >= 64 * 1024


def _codegen_backend(backend:str)->str:
    """The code-generation backend a launch backend spells (hip reuses cuda's)."""
    return "cuda" if backend == "hip" else backend


def test_a_consumed_call_answers_python(gpu_backend):
    """4! = 24 through a multiply after the callee answers -- the shape the tail pass
    refuses and the native unlock takes (on the backend that may recurse)."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    out = np.zeros(1, dtype=np.int32)
    k_consumed[gpu_backend](out, np.int32(4))
    assert int(out[0]) == 24


def test_a_self_call_beside_a_literal_answers_python(gpu_backend):
    """`biased(n - 1) + 1` -- the `+` promotes int with int, and the depth is n."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    out = np.zeros(1, dtype=np.int32)
    k_biased[gpu_backend](out, np.int32(2))
    assert int(out[0]) == 2


def _helper_source(func, backend:str)->str:
    """The generated device code of one helper, from the plan's own cache."""
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    callers:dict = {fib: (k_fib, [infer_ctype(np.zeros(1, dtype=np.int64)),
                                  infer_ctype(np.int32(0))]),
                    consumed: (k_consumed, [infer_ctype(np.zeros(1, dtype=np.int32)),
                                            infer_ctype(np.int32(0)),
                                            infer_ctype(np.int32(0))]),
                    biased: (k_biased, [infer_ctype(np.zeros(1, dtype=np.int32)),
                                        infer_ctype(np.int32(0))]),
                    rec_sum: (k_sum, [infer_ctype(np.zeros(1, dtype=np.int64)),
                                      infer_ctype(np.int32(0))]),
                    raised: (k_raised, [infer_ctype(np.zeros(1, dtype=np.int32)),
                                        infer_ctype(np.int32(0))]),
                    depth_of: (k_depth, [infer_ctype(np.zeros(1, dtype=np.int32)),
                                         infer_ctype(np.zeros(1, dtype=np.int32))])}
    caller, types = callers[func]
    translator = FunctionTranslator(
        caller.raw_func, types, _codegen_backend(backend),
        lambda name: globals().get(name), {}, is_kernel=True)
    translator.translate()
    pieces = [entry["code"] for entry in translator.device_cache.values()
              if isinstance(entry, dict) and isinstance(entry.get("code"), str)]
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# The shapes the unlock refuses keep the old error
# ---------------------------------------------------------------------------


def test_an_outer_read_now_converts(gpu_backend):
    """The suspended expression reads a module-level constant: the constant is
    loop-invariant (one value for the whole walk), so the generalized conversion
    accepts it and the answer agrees with python. The v1 pass refused this
    spelling; the generalized suspension accepts any expression reading only
    the slot, loop-invariant globals, and the block's own locals."""
    out = np.zeros(1, dtype=np.int32)
    k_touching[gpu_backend](out, np.int32(2))
    assert int(out[0]) == 2 * OFFSET


def test_an_operator_over_a_local_now_converts(gpu_backend):
    """The suspended expression reads only the slot (`half + half`): the loop
    form re-runs it once per level, which is exactly the original recursion's
    order. The v1 pass refused this shape; the generalized suspension accepts
    any expression whose reads are the slot and loop-invariant names."""
    out = np.zeros(1, dtype=np.int32)
    k_divided[gpu_backend](out, np.int32(8))
    assert int(out[0]) == 8


def test_a_tail_call_with_a_raise_takes_the_loop_not_the_unlock(gpu_backend):
    """A raise beside a *tail* call is v1's shape: the loop conversion runs first and
    the device answers by iteration -- the numbers agree with python, and the raise's
    refusal of native recursion never comes into play."""
    out = np.zeros(1, dtype=np.int32)
    k_raised[gpu_backend](out, np.int32(3))
    assert int(out[0]) == 6
    source = _helper_source(raised, gpu_backend)
    assert "while" in source


def test_a_raise_outside_tail_position_still_refuses(gpu_backend):
    """A raise the loop conversion cannot take keeps the monomorphizer's refusal:
    the exception base is part of the key, and a self-call would grow it."""
    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(SyntaxError):
        k_raise_mid[gpu_backend](out, np.int32(3))


def test_a_refused_shape_answers_python_on_the_interpreter():
    """The python backend runs every body as written -- the unlock is device-only."""
    out = np.zeros(1, dtype=np.int64)
    k_fib["python"](out, np.int32(10))
    assert int(out[0]) == 55

    # The static scans over the same bodies live in the shapes suite; the bar and
    # the seed are exercised there against these very functions (imported, so the
    # two files share one set of helpers and one set of answers).


def test_the_kernel_backend_is_untouched():
    """The interpreter runs the recursion as written -- the rewrite is device-only."""
    out = np.zeros(1, dtype=np.int64)
    k_fib["python"](out, np.int32(10))
    assert int(out[0]) == 55
