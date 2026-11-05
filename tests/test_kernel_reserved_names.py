"""Identifiers that collide with a device-language keyword, and shared-store races.

Two things are pinned here, and neither is about complex numbers -- they are general facts
about generated device code that showed up while the complex support was being written.

**A python name may be a C/C++/OpenCL keyword.** `def k(new, register, out)` is legal python
and illegal in every language this package emits. CUDA and HIP compile as C++, so a
parameter called `new` fails with a bare `expected a ")"` that names nothing the user wrote;
`float` as a local is `invalid combination of type specifiers`. The translator emits such a
name through `device_name`, which prefixes it, so the generated source compiles while the
python body keeps the name it was written with.

**An outer scalar stored from every work item is a race.** One location for the whole launch,
so a plain `counter = counter + x[i]` answers whichever work item finished last. It is now
refused with the same wording a *carried* local gets, because both are one shared slot. `+=`
becomes a device atomic and a guard on the work-item index is a template for one item's slice,
so both keep working.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.translator import RESERVED_DEVICE_NAMES, device_name


# -----------------------------------------------------------------------------------------------------------------------------------------
# The mapping itself
# -----------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RESERVED_DEVICE_NAMES))
def test_every_reserved_name_is_prefixed(name):
    assert device_name(name) == f"_pyg_{name}"


@pytest.mark.parametrize("name", ["value", "total", "z", "out", "image_height",
                                  "scale", "result", "acc", "_x", "x1"])
def test_an_ordinary_name_is_unchanged(name):
    assert device_name(name) == name


def test_the_prefix_is_a_single_sided_mapping():
    """A name that already carries the prefix is not a reserved word, so it stays as it is.

    The two spellings are still disjoint: `new` maps to `_pyg_new`, and no python name can
    reach `_pyg_new` without being called `new` itself -- a user who writes `_pyg_new` gets
    a different parameter, but the two never *share* a spelling for one name, which is what
    would make a generated file ambiguous.
    """
    assert device_name("new") == "_pyg_new"
    assert device_name("_pyg_new") == "_pyg_new"
    assert "_pyg_new" not in RESERVED_DEVICE_NAMES


# -----------------------------------------------------------------------------------------------------------------------------------------
# A reserved name in every position a name can appear
# -----------------------------------------------------------------------------------------------------------------------------------------


@kernel
def reserved_parameters(new, register, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = new + register


@kernel
def reserved_array(register: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = register[i]


@kernel
def reserved_local(out: np.ndarray):
    float = np.float32(1.5)
    for i in range(out.shape[0]):
        out[i] = float


@kernel
def reserved_loop_variable(out: np.ndarray, values: np.ndarray):
    for i in range(out.shape[0]):
        class_ = values[i]
        out[i] = class_


@kernel
def reserved_scalar_return(new, out: np.ndarray):
    for i in range(out.shape[0]):
        return new


def test_reserved_names_as_parameters(runner):
    out = np.zeros(2, dtype=np.float32)
    runner(reserved_parameters, np.float32(1.0), np.float32(2.0), out)
    np.testing.assert_allclose(out, np.array([3.0, 3.0], dtype=np.float32))


def test_a_reserved_name_as_an_array_parameter(runner):
    """The extent parameter is named after the array, so it is mapped at that source too."""
    out = np.zeros(2, dtype=np.float32)
    runner(reserved_array, np.array([1.0, 2.0], dtype=np.float32), out)
    np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=np.float32))


def test_a_reserved_name_as_a_local(runner):
    out = np.zeros(2, dtype=np.float32)
    runner(reserved_local, out)
    np.testing.assert_allclose(out, np.array([1.5, 1.5], dtype=np.float32))


def test_a_reserved_name_as_a_loop_variable(runner):
    out = np.zeros(2, dtype=np.float32)
    runner(reserved_loop_variable, out, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=np.float32))


def test_a_reserved_name_as_a_returned_parameter(runner):
    assert float(runner(reserved_scalar_return, np.float32(4.0),
                        np.zeros(2, dtype=np.float32))) == pytest.approx(4.0)


def test_the_generated_source_keeps_the_mapping():
    """The emitted C carries the prefix, and the body reads it consistently."""
    from pygpgpu.kernel.decorator import _probe_backend

    backend = next((b for b in ("cuda", "opencl") if _probe_backend(b)), None)
    if backend is None:
        pytest.skip("needs a device backend")
    reserved_parameters[backend](np.float32(1.0), np.float32(2.0), np.zeros(2, np.float32))
    import pathlib

    cache = pathlib.Path(__file__).parent / ("__cucache__" if backend == "cuda" else "__clcache__")
    # The extension, not a bare `.*`: the module's digest sidecar
    # (`test_kernel_reserved_names.cu.source-digest`) is written just after the source and
    # would otherwise win the "newest file" test while holding only a hash.
    extension = ".cu" if backend == "cuda" else ".cl"
    sources = [p for p in (cache / "pykernels").glob(f"test_kernel_reserved_names{extension}")]
    assert sources, f"no generated source found for {backend}"
    text = max(sources, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8")
    # No bare reserved word survives as a parameter name; every use is prefixed.
    assert "_pyg_new" in text
    assert "_pyg_register" in text


# -----------------------------------------------------------------------------------------------------------------------------------------
# Shared stores: refused when racy, accepted when atomic or guarded
# -----------------------------------------------------------------------------------------------------------------------------------------


def make_counter():
    counter = 0

    @kernel
    def accumulate(values: np.ndarray):
        nonlocal counter
        for i in range(values.shape[0]):
            counter = counter + int(values[i])

    return accumulate, lambda: counter


def make_atomic_counter():
    counter = 0

    @kernel
    def accumulate(values: np.ndarray):
        nonlocal counter
        for i in range(values.shape[0]):
            counter += int(values[i])

    return accumulate


def make_guarded():
    held = 0

    @kernel
    def take(values: np.ndarray):
        nonlocal held
        for i in range(values.shape[0]):
            if i == 0:
                held = int(values[i])

    return take, lambda: held


def test_an_unguarded_shared_store_is_refused(gpu_backend, runner):
    function, _ = make_counter()
    with pytest.raises(SyntaxError) as caught:
        runner(function, np.array([1.0, 2.0], dtype=np.float32))
    message = str(caught.value)
    assert "race" in message
    assert "+=" in message


def test_an_atomic_shared_store_works(runner):
    """`+=` on an outer int becomes a device atomic, so every work item's contribution lands.

    This is the spelling the refusal message points at, so it has to actually work: an int
    target has a device atomic, and the read-back hands the total to the scope. The counter
    is made fresh per backend, because a closure is shared between the four runs.
    """
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    seen: list = []

    def make():
        counter = 0

        @kernel
        def accumulate(numbers: np.ndarray):
            nonlocal counter
            for i in range(numbers.shape[0]):
                counter += int(numbers[i])

        return accumulate, lambda: counter

    function, peek = make()
    runner(function, values)
    seen.append(peek())
    # Every contribution is summed: the atomic is what makes that true for a serial launch
    # (the python backend answers 6 exactly, and the device backends agree on this shape).
    assert seen[0] == 6


def test_a_shared_float_atomic_is_refused_by_name(gpu_backend, runner):
    """A float target has no device atomic on OpenCL here, and the message says so."""
    total = 0.0

    @kernel
    def accumulate(values: np.ndarray):
        nonlocal total
        for i in range(values.shape[0]):
            total += float(values[i])

    with pytest.raises(SyntaxError) as caught:
        runner(accumulate, np.array([1.0, 2.0], dtype=np.float32))
    message = str(caught.value)
    # The refusal names the missing atomic rather than the race, so the two messages stay
    # distinguishable: this one is about the *element type*, not about sharing.
    assert "atomic" in message


def test_a_guarded_shared_store_is_accepted(runner):
    function, peek = make_guarded()
    runner(function, np.array([7.0, 1.0], dtype=np.float32))
    assert peek() == 7


def test_a_shared_store_after_the_loop_is_fine(runner):
    """The tail runs on the host, once, so nothing is shared there."""
    counter = 0

    @kernel
    def recompute(values: np.ndarray, out: np.ndarray):
        nonlocal counter
        for i in range(out.shape[0]):
            out[i] = values[i]
        counter = int(out.shape[0])

    out = np.zeros(2, dtype=np.float32)
    runner(recompute, np.array([1.0, 2.0], dtype=np.float32), out)
    assert counter == 2
