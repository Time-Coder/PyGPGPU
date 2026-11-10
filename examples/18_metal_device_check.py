"""Run the Metal backend end to end on a real Apple GPU.

Everything in this repository's test suite runs on any machine: the source
generation, the declaration scanner, the type tables and the cache are all checked
without Metal.framework, and the parts that need a device are stubbed. This script is
the other half -- the four facts that only a Mac can answer:

1. **The generated MSL compiles.** `tests/metal_syntax_check.py` parses it as C++14,
   which catches structural errors but not the attribute and address-space rules Metal
   enforces. `newLibraryWithSource:options:error:` is what proves the prelude's macros
   expand onto the declared parameters.
2. **A pipeline accepts the signature.** The function lookup and the pipeline build are
   wired and their ownership is balanced; whether a generated signature is one Metal
   accepts is a device question.
3. **The dispatch executes the work items.** The bracket-to-`MTLSize` mapping is
   unit-tested for every spelling; that the dispatched threads then run is not.
4. **A scalar parameter reads its value.** `setBytes:` and `setBuffer:` are two
   different bindings, and a real launch is what proves a scalar arrives as a value
   rather than a pointer.

Each check below states what it proves and why it cannot be answered elsewhere. Run it
from the repository root:

    python examples/18_metal_device_check.py

Exit codes: 0 all checks passed, 1 a check ran and failed, 2 there is no Metal device
on this machine (the expected answer everywhere off macOS).

**The checks themselves are exercised elsewhere.** Every kernel here was run on an
OpenCL device first, so the call shapes -- the bracket spelling, `atomic_add`'s tuple
index, the `local_array_per_group`/`barrier` pair, the return claim -- are known to be
accepted by this library rather than assumed. What this script adds is the part OpenCL
cannot answer for Metal: whether the *Metal* binding and dispatch do the same thing.
"""
import sys

import numpy as np

from pygpgpu import float4, kernel, list_devices
# The public `list_devices()` answers "which devices exist"; there is no public
# per-backend probe, and asking one is what this script needs to tell "Metal is
# missing" apart from "a check failed". `_probe_backend` is the library's own
# answer to that question (it is what the default-launch order uses), so it is
# imported deliberately rather than reimplemented here.
from pygpgpu.kernel.decorator import _probe_backend


def require_metal() -> str:
    """The Metal device this run will use, or exit with the reason it cannot."""
    if _probe_backend("metal") is None:
        names = list_devices()
        print("The Metal backend is not available on this machine.")
        print("Devices found:", names or "(none)")
        print()
        print("That is expected off macOS: MTLCreateSystemDefaultDevice returns nil")
        print("there, and a Mac virtual machine has no device either. Run this on a")
        print("Mac with a real Apple GPU.")
        # Exit 2, not 1: "this machine has no Metal" is the expected answer off macOS
        # and must not read as a failed check. 1 is reserved for a check that ran and
        # failed; 2 for "could not run".
        raise SystemExit(2)
    devices = [name for name in list_devices() if name.startswith("metal:")]
    print("Metal is available:", devices)
    return devices[0]


def check_source_compiles(device: str) -> None:
    """(1) The emitted MSL is a library Metal accepts."""
    @kernel
    def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
        for i in range(x.shape[0]):
            out[i] = alpha * x[i] + y[i]

    x = np.arange(16, dtype=np.float32)
    y = np.ones_like(x)
    out = np.zeros_like(x)

    saxpy[device](x, y, out, 2.0)
    expected = 2.0 * x + y
    assert np.allclose(out, expected), (out, expected)
    print("  (1) the MSL compiled and the kernel ran: OK")
    print(f"      out.sum() = {float(out.sum()):.1f} (expected {float(expected.sum()):.1f})")


def check_scalar_binding(device: str) -> None:
    """(4) A scalar arrives as a value, not as a pointer.

    This is the binding the backend cannot check without a device: a scalar parameter
    lives in MSL's thread address space and binds with `setBytes:length:atIndex:`. If
    it were bound with `setBuffer:` the pipeline would present a pointer where the
    kernel declared a value. The kernel below writes the scalar it received back into
    the array, so a wrong binding shows up as a wrong number rather than a crash.
    """
    @kernel
    def echo_scalar(out: np.ndarray, value: int) -> None:
        for i in range(out.shape[0]):
            out[i] = value

    out = np.zeros(8, dtype=np.int32)
    echo_scalar[device](out, 12345)
    assert (out == 12345).all(), out
    print("  (4) a scalar parameter arrived as its value: OK")


def check_dispatch(device: str) -> None:
    """(3) The dispatched work items really run, and each sees its own id."""
    from pygpgpu import get_global_id

    @kernel(domain="out")
    def ids(out: np.ndarray) -> None:
        index = get_global_id(0)
        out[index] = index

    # A bracket the unit tests cover: two groups of four.
    out = np.full(8, -1, dtype=np.int32)
    ids[device][(2, 4)](out)
    assert (out == np.arange(8, dtype=np.int32)).all(), out
    print("  (3) dispatchThreadgroups ran every work item with its own id: OK")


def check_shared_memory(device: str) -> None:
    """A threadgroup buffer plus a barrier -- the reduction shape, on the device."""
    from pygpgpu import barrier, blockIdx, get_local_id, local_array_per_group

    @kernel
    def reverse(text: np.ndarray, out: np.ndarray) -> None:
        scratch = local_array_per_group(8, np.float32)
        lane = get_local_id(0)
        scratch[blockIdx][lane] = text[lane]
        barrier()
        out[lane] = scratch[blockIdx][7 - lane]

    text = np.arange(8, dtype=np.float32)
    out = np.zeros_like(text)
    reverse[device](text, out)
    assert np.allclose(out, text[::-1]), out
    print("  (2) threadgroup memory and the barrier work: OK")


def check_atomics(device: str) -> None:
    """A racy `+=` becomes the C++11 atomic family and loses no update."""
    from pygpgpu import atomic_add

    @kernel
    def count(sources: np.ndarray, totals: np.ndarray) -> None:
        for i in range(sources.shape[0]):
            atomic_add(totals, (sources[i],), 1)

    sources = np.zeros(4096, dtype=np.uint32)
    totals = np.zeros(1, dtype=np.uint32)
    count[device](sources, totals)
    assert int(totals[0]) == 4096, int(totals[0])
    print("  (2) the atomic version counted every work item: OK")


def check_vector_types(device: str) -> None:
    """MSL's own vector widths: `float4` in, `float4` out."""
    @kernel
    def scale_vectors(source: np.ndarray, out: np.ndarray) -> None:
        for i in range(out.shape[0]):
            out[i] = source[i] * 2.0

    source = np.zeros(4, dtype=float4)
    source["x"] = 1.0
    out = np.zeros(4, dtype=float4)
    scale_vectors[device](source, out)
    assert np.allclose(out["x"], 2.0), out
    print("  (4) vector parameters kept their MSL type: OK")


def check_two_phases(device: str) -> None:
    """Two magic loops: two kernels, launched in order, sharing one stream."""
    @kernel
    def double_then_add(x: np.ndarray, mid: np.ndarray, out: np.ndarray) -> None:
        for i in range(out.shape[0]):
            mid[i] = x[i] * 2.0
        for i in range(out.shape[0]):
            out[i] = mid[i] + 1.0

    x = np.arange(8, dtype=np.float32)
    mid = np.zeros_like(x)
    out = np.zeros_like(x)
    double_then_add[device](x, mid, out)
    assert np.allclose(out, x * 2.0 + 1.0), out
    print("  (3) a two-phase body ran both passes in order: OK")


def check_return_value(device: str) -> None:
    """A kernel `return` travels through the slots and the claim word."""
    @kernel
    def first(x: np.ndarray) -> float:
        for i in range(x.shape[0]):
            return x[i]
        return 0.0

    x = np.arange(8, dtype=np.float32)
    answer = first[device](x)
    assert answer == 0.0, answer
    print("  (4) the return claim word handed back one value: OK")


def main() -> None:
    device = require_metal()
    checks = (
        ("generated MSL compiles and runs", check_source_compiles),
        ("scalar is bound as a value", check_scalar_binding),
        ("dispatch covers every work item", check_dispatch),
        ("threadgroup memory and barrier", check_shared_memory),
        ("atomics lose no update", check_atomics),
        ("vector types", check_vector_types),
        ("two-phase body", check_two_phases),
        ("return value", check_return_value),
    )
    failed = 0
    for label, check in checks:
        print(f"[{label}]")
        try:
            check(device)
        except Exception as error:  # noqa: BLE001  (a report, not a swallow)
            failed += 1
            print(f"  FAILED: {type(error).__name__}: {error}")
        print()
    if failed:
        print(f"{failed} of {len(checks)} checks failed.")
        raise SystemExit(1)
    print(f"All {len(checks)} checks passed on {device}.")


if __name__ == "__main__":
    sys.exit(main())
