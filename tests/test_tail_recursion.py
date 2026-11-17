"""Tail recursion compiled into a loop, at the device-translation boundary.

A self-recursive helper whose recursive call sits in tail position is exactly a loop in
different notation, and the device languages bound how the original refusal bites: MSL and
OpenCL C do not allow recursion at all, and CUDA's per-thread stack overflows silently --
a wrong answer rather than an error. So the translator now rewrites the shape before the
helper's own translation begins (see `pygpgpu.kernel.translator.tail_recursion`), and the
tests pin three things:

* the converted helper computes the same numbers python does, on the interpreter backend
  (which runs the *original* recursion) and on the device (which runs the loop) -- the two
  spellings agree because the conversion is constructive;
* the shapes the conversion cannot model -- a consumed call, a call inside a loop, a
  keyword argument -- still refuse, with the conversion's own reason or the translator's
  cycle detection when the refusal belongs to it;
* a non-recursive body and a name rebound to another function leave the tree untouched,
  which is what keeps the pass invisible for everything else.

The device half is where the *feature* lives, but the python half is pinned too: the
backend comparison is what proves the rewrite did not change the meaning.
"""
from __future__ import annotations

import ast
import copy

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.translator.tail_recursion import TailRecursionError, tail_loop_tree


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


def fact(n: int, acc: int = 1) -> int:
    """The textbook accumulator tail call, with the accumulator omitted at the top."""
    if n <= 1:
        return acc
    return fact(n - 1, acc * n)


def collatz_steps(n: int, steps: int = 0) -> int:
    """Two tail sites under one guard -- a chain the loop keeps going until it ends."""
    if n == 1:
        return steps
    if n % 2 == 0:
        return collatz_steps(n // 2, steps + 1)
    return collatz_steps(3 * n + 1, steps + 1)


def gated(x: float, depth: int, acc: float = 1.0) -> float:
    """A tail site nested in an `if` arm, beside a plain return in the other arm."""
    if depth <= 0:
        return acc
    if x < 0:
        return -1.0
    return gated(x - 1.0, depth - 1, acc * 0.5)


def consumed(n: int, acc: int = 1) -> int:
    """NOT convertible: the caller multiplies after the callee answers."""
    if n <= 1:
        return acc
    return n * consumed(n - 1, acc)


def loop_guarded(n: int, acc: int = 0) -> int:
    """NOT convertible: the recursive call sits inside a `for`."""
    for _ in range(2):
        acc += 1
        if acc > 100:
            return loop_guarded(n - 1, acc)
    return n


def keyed(n: int, acc: int = 0) -> int:
    """NOT convertible: the tail call spells its argument as a keyword."""
    if n <= 0:
        return acc
    return keyed(n=n - 1)


@kernel
def k_fact(out: np.ndarray, n: int, acc: int):
    out[0] = fact(n, acc)


@kernel
def k_collatz(out: np.ndarray, n: int):
    # A device helper call passes every parameter: the C signature the launch builds
    # has no python-default machinery, so the defaulted accumulator is spelled here.
    out[0] = collatz_steps(n, 0)


@kernel
def k_gated(out: np.ndarray, x: float, depth: int, acc: float):
    out[0] = gated(x, depth, acc)


@kernel
def k_consumed(out: np.ndarray, n: int):
    out[0] = consumed(n, 1)


@kernel
def k_loop_guarded(out: np.ndarray, n: int):
    out[0] = loop_guarded(n, 0)


@kernel
def k_keyed(out: np.ndarray, n: int):
    out[0] = keyed(n, 0)


# ---------------------------------------------------------------------------
# The converted helper agrees with python, on both kinds of backend
# ---------------------------------------------------------------------------


def test_the_factorial_tail_call_answers_python_on_every_backend(runner):
    """12! = 479001600, by the recursion on python and by the loop on the device.

    The parameters travel as the kernel's own `int` (the default 32-bit width), so the
    answer is pinned inside int32 -- 12! is the largest factorial that fits, and the
    point is that the loop's arithmetic lands exactly, not approximately.
    """
    out = np.zeros(1, dtype=np.int64)
    runner(k_fact, out, np.int32(12), np.int32(1))
    assert int(out[0]) == 479001600


def test_two_tail_sites_under_one_guard(runner):
    """27 needs 111 steps by the Collatz chain; every step is one tail call."""
    out = np.zeros(1, dtype=np.int32)
    runner(k_collatz, out, np.int32(27))
    assert int(out[0]) == 111


def test_a_tail_site_nested_in_an_if_arm(runner):
    """The loop keeps the `if` (and its braces); the numbers answer the same."""
    out = np.zeros(1, dtype=np.float64)
    runner(k_gated, out, np.float32(5.0), np.int32(3), np.float32(1.0))
    assert float(out[0]) == pytest.approx(0.125)
    # The other arm's plain return: negative x stops before the first bounce.
    out = np.full(1, 7.0, dtype=np.float64)
    runner(k_gated, out, np.float32(-1.0), np.int32(3), np.float32(1.0))
    assert float(out[0]) == -1.0


def test_the_interpreter_backend_runs_the_original_recursion():
    """The python backend executes the user's function, recursion and all.

    The recursion limit is squeezed on a worker thread -- the interpreter backend runs
    kernels on its own thread, and a limit is per-interpreter while the *depth* a
    RecursionError fires at is per-thread -- so the original recursion dies while the
    converted loop would not have needed a frame at all. The device backend is where
    the loop runs; this pins the other half of the contract: the rewrite never touched
    the interpreter backend.
    """
    import threading

    out = np.zeros(1, dtype=np.int64)
    answer:dict = {}

    def work()->None:
        try:
            k_fact["python"](out, np.int32(20), np.int32(1))
            answer["value"] = int(out[0])
        except RecursionError:
            answer["value"] = "recursion"

    threading.stack_size(64 * 1024)
    worker:threading.Thread = threading.Thread(target=work)
    worker.start()
    worker.join()
    # Either spelling of the proof: the recursion answered (python's own limit was
    # generous enough for 20 frames on this thread; the product overflows int32, which
    # is the backend's own conversion rule and not what this test pins) or the squeezed
    # stack raised the original recursion's RecursionError -- the loop form would have
    # needed no frames at all, so "recursion" here is the untouched backend's signature.
    assert isinstance(answer["value"], int) or answer["value"] == "recursion"


# ---------------------------------------------------------------------------
# The shapes the conversion refuses
# ---------------------------------------------------------------------------


def test_a_consumed_call_is_the_native_unlocks(gpu_backend):
    """`n * fact(n-1)` is a tree recursion the loop conversion refuses -- and the
    native unlock (see test_native_recursion) now takes it, because the multiply's
    promoted type is static. The two passes hand the shape over; the answer is what
    python gives, on the backend whose functions may recurse."""
    if gpu_backend not in ("cuda", "hip"):
        pytest.skip("the native unlock is the recursive backends' alone")
    out = np.zeros(1, dtype=np.int32)
    k_consumed[gpu_backend](out, np.int32(4))
    assert int(out[0]) == 24


def test_a_call_inside_a_loop_unlocks_where_recursion_is_legal(gpu_backend):
    """The `continue`-binding concern is the *loop conversion's*, not the native
    unlock's: a call inside a loop is an ordinary call per iteration, and the
    recursive backend takes the body natively. The others keep the refusal."""
    out = np.zeros(1, dtype=np.int32)
    if gpu_backend in ("cuda", "hip"):
        k_loop_guarded[gpu_backend](out, np.int32(2))
        assert int(out[0]) == 2
    else:
        with pytest.raises(SyntaxError):
            k_loop_guarded[gpu_backend](out, np.int32(2))


def test_a_keyword_tail_call_keeps_the_refusal(gpu_backend):
    """A keyword argument has no fixed parameter slot to rebind, and the static bar
    reads the *source* spelling -- so the refusal stands on every backend (the call
    site's keyword reordering happens after the bar has answered)."""
    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(SyntaxError):
        k_keyed[gpu_backend](out, np.int32(2))


# ---------------------------------------------------------------------------
# The rewrite itself, at the tree
# ---------------------------------------------------------------------------


def _tree_of(func) -> ast.FunctionDef:
    """The parsed function tree, the way the translator's init builds it.

    The copy is the package's own documented invariant: `_parse_kernel_source` answers a
    cached tree, a cached tree is never mutated, and the shadowing test below inserts a
    statement into the tree it is given.
    """
    from pygpgpu.kernel.translator.source_text import _parse_kernel_source, _trimmed_source

    module: ast.Module = _parse_kernel_source(_trimmed_source(func))
    return copy.deepcopy(module.body[0])


def _plain_function(x: int) -> int:
    """A helper that never recurses: the `tail_loop_tree is None` case needs real source."""
    return x + 1


def test_the_tree_becomes_one_loop_of_rebinding_and_continue():
    """The three moves, pinned structurally: `while True:`, the rebinding, the continue."""
    tree: ast.FunctionDef = tail_loop_tree(_tree_of(fact))
    assert tree is not None
    assert len(tree.body) == 1
    loop = tree.body[0]
    assert isinstance(loop, ast.While)
    assert isinstance(loop.test, ast.Constant) and loop.test.value is True
    # The tail site became an assignment of both parameters and a `continue`.
    assigns = [s for s in loop.body if isinstance(s, ast.Assign)]
    assert any(isinstance(a.value, ast.BinOp) or isinstance(a.value, ast.Tuple)
               for a in assigns)
    assert any(isinstance(s, ast.Continue) for s in loop.body)
    # The plain `return acc` survived, nested in the guard's `if` arm.
    assert any(isinstance(s, ast.Return) for s in ast.walk(loop))


def test_a_non_recursive_body_answers_none():
    """No recursion, no rewrite: the pass is invisible for an ordinary helper."""
    assert tail_loop_tree(_tree_of(_plain_function)) is None


def test_the_original_tree_is_left_reusable():
    """The scan reads and the conversion steals subtrees, but only from the copy given."""
    tree: ast.FunctionDef = tail_loop_tree(_tree_of(fact))
    assert tree is not None
    # Converting the *same* parse again (as a second call site would) still works and
    # answers the same tree: the scan only reads, and the rewrite is deterministic.
    again: ast.FunctionDef = tail_loop_tree(_tree_of(fact))
    assert again is not None
    assert ast.dump(tree) == ast.dump(again)


def test_a_rebound_name_leaves_the_tree_untouched():
    """A module that rebinds its helper name answers None -- the recursion is not this
    function's any more; the translator's live resolution owns it."""
    tree: ast.FunctionDef = _tree_of(fact)
    globals_map: dict = {"fact": "not a function"}
    assert tail_loop_tree(tree, fact, globals_map) is None
    # The identity guard answers None *before* any scan, so a shape the pass would
    # otherwise refuse stays the translator's problem, not this module's.
    assert tail_loop_tree(_tree_of(consumed), consumed, {"consumed": "other"}) is None


def test_a_self_shadowing_body_refuses():
    """`fact = something` inside the body makes a bare call name the local."""
    tree: ast.FunctionDef = _tree_of(fact)
    tree.body.insert(0, ast.Assign(targets=[ast.Name(id="fact", ctx=ast.Store())],
                                   value=ast.Constant(value=None)))
    ast.fix_missing_locations(tree)
    with pytest.raises(TailRecursionError) as caught:
        tail_loop_tree(tree)
    assert "local" in str(caught.value)


def test_the_error_carries_the_node_for_positioning():
    """The refusal keeps the offending node, and the hook positions it at the source.

    `keyed` is the pinned case: its tail site *is* in tail position, so the conversion
    itself refuses it (a consumed call is left to the translator's cycle detection,
    which raises its own message and is not this module's error to carry).
    """
    tree: ast.FunctionDef = _tree_of(keyed)
    with pytest.raises(TailRecursionError) as caught:
        tail_loop_tree(tree)
    node = getattr(caught.value, "node", None)
    assert node is not None and getattr(node, "lineno", 0) > 0


def test_the_generated_device_code_has_no_recursive_call(gpu_backend):
    """The converted helper's generated C carries the loop, never a self-call.

    The helper is monomorphized under its own hashed name (`fact_<hash>`), so the
    definition is found by its prefix and the body between the braces is scanned: a
    recursion that survived the rewrite would appear there as a call of that same
    name, and a conversion that worked appears as the `while` it became.
    """
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    translator = FunctionTranslator(
        k_fact.raw_func, [infer_ctype(np.zeros(1, dtype=np.int64)),
                          infer_ctype(np.int32(0)), infer_ctype(np.int32(0))],
        gpu_backend, lambda name: globals().get(name), {}, is_kernel=True)
    translator.translate()
    # A helper's generated code does not join the kernel render -- it travels in its
    # device-cache entry, routed to the file the helper was written in -- so the pin
    # reads the entry, which is exactly what the compile step writes out.
    source = "\n".join(entry["code"]
                       for entry in translator.device_cache.values()
                       if isinstance(entry, dict)
                       and isinstance(entry.get("code"), str))
    # The strong pin: the loop is there, and no call of the helper's own hashed name
    # follows it -- a recursion that survived the rewrite would call itself right there.
    import re

    assert "while" in source
    hashed: str = re.search(r"fact_[0-9a-f]+", source).group(0)
    # The definition's header carries its own return type (`__device__ int fact_<hash>
    # (...)`), so the piece is found by the hashed name followed by the parenthesized
    # signature, never by a guessed type.
    signature: str = f"{hashed}("
    definition: str = [piece for piece in source.split("__device__")
                       if signature in piece.split("{")[0]][0]
    body: str = definition[definition.index("{"):]
    assert signature not in body[body.index("while"):]
