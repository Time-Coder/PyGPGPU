"""The unlock's wider shapes: methods, vectors, mutual cycles and array parameters.

The first suite (`test_native_recursion.py`) pins the *core* unlock and the refusals;
this one pins the shapes the extended verdict admits, each of which rides the same
frame check with a wider forwarding rule:

* a **method's** recursion spells `self.<name>` and forwards the receiver struct plus
  its `_self_<attr>` slots back to the same function -- the identity a self-call gets;
* a **vector-returning** recursion is typable because the constructor names the device
  spelling (through the host alias's MRO), and the value travels in the same slots any
  helper's vector return uses;
* a **mutual cycle** (A calls B, B calls A) rides the frame check; both members fix
  the same scalar type, and the generated unit needs the forward declarations any
  mutual C unit does;
* an **array parameter** forwards the shape extents a self-call indexes through --
  the identity the same-function forwarding gives -- and the token rewrite spells the
  call with the site's own shape variable.

Every device answer is compared against the same body run on the interpreter backend,
which executes the user's recursion as written: the two spellings agree because the
conversion is the call machinery's, not the body's.
"""
from __future__ import annotations

import pytest

import numpy as np

from pygpgpu import kernel

from test_native_recursion import (
    Walker,
    _codegen_backend,
    biased,
    consumed,
    depth_of,
    divided,
    even,
    fib,
    k_depth,
    k_even,
    k_fib,
    k_vec,
    k_walk,
    odd,
    rec_frac,
    rec_mixed,
    raised,
    scan_from,
    touching,
)
from pygpgpu.kernel.translator.scalar_recursion import RECURSIVE_BACKENDS, scalar_bar_met
from pygpgpu.kernel.translator.return_type_scan import seed_return_type
from pygpgpu.kernel.translator.source_text import _parse_kernel_source, _trimmed_source


def scan_from_local(arr: np.ndarray, i: int, target: float) -> int:
    """The module-level `scan_from` is imported; this one backs its own kernel."""
    if i >= arr.shape[0]:
        return -1
    if arr[i] == target:
        return i
    return scan_from_local(arr, i + 1, target)


@kernel
def k_scan(out: np.ndarray, arr: np.ndarray, target: float):
    out[0] = scan_from_local(arr, 0, target)


def test_a_method_recursion_answers_python(gpu_backend):
    """`w.walk(5)` accumulates the factor five times: 15 -- the recursion is spelled
    `self.walk(...)` inside the method, and the receiver struct plus its `_self_`
    slots forward to the same function (the identity a self-call gives)."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    w = Walker(3)
    out = np.zeros(1, dtype=np.int32)
    k_walk[gpu_backend](out, w, np.int32(5))
    assert int(out[0]) == 15


def test_a_vector_returning_recursion_answers_python(gpu_backend):
    """The first positive element comes home as a `float3`: the constructor names the
    device type through the host alias's MRO, and the recursion carries the vector
    by value -- the same slot machinery any helper's vector return uses."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    arr = np.array([0.0, -1.0, 5.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    k_vec[gpu_backend](out, arr)
    assert out.tolist() == [5.0, 0.0, 0.0]


def test_mutual_recursion_answers_python(gpu_backend):
    """even(7) = 0 through the A-to-B-to-A cycle -- each member fixes `int`, and the
    generated source carries the forward declarations a mutual unit needs."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    out = np.zeros(1, dtype=np.int32)
    k_even[gpu_backend](out, np.int32(7))
    assert int(out[0]) == 0
    out = np.full(1, -1, dtype=np.int32)
    k_even[gpu_backend](out, np.int32(8))
    assert int(out[0]) == 1


def test_an_array_parameter_recursion_forwards_the_shape(gpu_backend):
    """Three nonzero elements before the zero -- the self-call forwards the array and
    the extent, which the token rewrite spells with the site's own shape variable."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    arr = np.array([3, 2, 1, 0, 9], dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    k_depth[gpu_backend](out, arr)
    assert int(out[0]) == 3


def test_an_array_parameter_recursion_answers_the_search(gpu_backend):
    """The first index holding 2.0 -- the walk passes the array, the index and the
    target scalar; the answer travels back through four native frames."""
    if _codegen_backend(gpu_backend) not in RECURSIVE_BACKENDS:
        pytest.skip("this backend's device functions may not recurse")
    arr = np.array([1.0, 4.0, 2.0, 8.0], dtype=np.float32)
    out = np.zeros(1, dtype=np.int32)
    k_scan[gpu_backend](out, arr, np.float32(2.0))
    assert int(out[0]) == 2
    out = np.zeros(1, dtype=np.int32)
    k_scan[gpu_backend](out, arr, np.float32(7.0))
    assert int(out[0]) == -1


def test_the_scan_helpers_agree_on_the_interpreter():
    """The imported bodies answer python's own recursion -- the rewrite is device-only."""
    arr = np.array([1.0, 4.0, 2.0, 8.0], dtype=np.float32)
    assert even(7) == 0   # 7 is odd
    assert odd(7) == 1
    assert even(8) == 1
    assert odd(8) == 0
    assert depth_of(np.array([3, 2, 1, 0, 9], dtype=np.int32), 0) == 3
    assert scan_from(arr, 0, 2.0) == 2


# ---------------------------------------------------------------------------
# The static scans, at the tree
# ---------------------------------------------------------------------------


def _tree_of(func) -> object:
    """The parsed function tree, the way the translator's init builds it.

    The copy is the package's own documented invariant: `_parse_kernel_source` answers
    a cached tree, and a cached tree is never mutated.
    """
    import copy  # noqa: PLC0415

    module: object = _parse_kernel_source(_trimmed_source(func))
    return copy.deepcopy(module.body[0])


def test_the_bar_accepts_only_the_unlocked_shape():
    """The bar is the *signature* clause: the unlocked probes pass it, the ones with
    a raise or an outer read fail it. The return-type clause is the seed's, beside it."""
    assert scalar_bar_met(_tree_of(fib), "fib", ["int"])
    assert scalar_bar_met(_tree_of(consumed), "consumed", ["int", "int"])
    assert not scalar_bar_met(_tree_of(raised), "raised", ["int", "int"])
    assert not scalar_bar_met(_tree_of(touching), "touching", ["int"])


def test_the_seed_names_one_scalar():
    """The seed reads the operator rules off the AST: parameter returns, promoted
    arithmetic, the bool of a comparison, the induction of a bare self-call."""
    assert seed_return_type(_tree_of(rec_frac),
                            ["float", "int", "float"], "float") == "float"
    assert seed_return_type(_tree_of(rec_mixed), ["float", "float"], "float") == "float"
    assert seed_return_type(_tree_of(fib), ["int"], "float") == "int"
    assert seed_return_type(_tree_of(consumed), ["int", "int"], "float") == "int"
    assert seed_return_type(_tree_of(biased), ["int"], "float") == "int"
    # An unknown operand type (`half` runs through a local) leaves the seed undecided.
    assert seed_return_type(_tree_of(divided), ["int"], "float") is None


def test_the_kernel_backend_is_untouched():
    """The interpreter runs the recursion as written -- the rewrite is device-only."""
    out = np.zeros(1, dtype=np.int64)
    k_fib["python"](out, np.int32(10))
    assert int(out[0]) == 55
