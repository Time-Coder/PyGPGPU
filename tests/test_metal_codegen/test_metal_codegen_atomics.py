"""The atomics tests, split out of `test_metal_codegen.py`.

The subject is atomics.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("call,expected", [
    ("atomic_add", "atomic_fetch_add_explicit"),
    ("atomic_sub", "atomic_fetch_sub_explicit"),
    ("atomic_min", "atomic_fetch_min_explicit"),
    ("atomic_max", "atomic_fetch_max_explicit"),
    ("atomic_and", "atomic_fetch_and_explicit"),
    ("atomic_or", "atomic_fetch_or_explicit"),
    ("atomic_xor", "atomic_fetch_xor_explicit"),
    ("atomic_xchg", "atomic_exchange_explicit"),
])
def test_atomics_use_the_metal_c_11_spelling(call: str, expected: str) -> None:
    """MSL has no `atomicAdd` family; the C++11 `atomic_fetch_*` functions are it."""
    from pygpgpu.kernel.device_builtins import _METAL_ATOMICS

    assert _METAL_ATOMICS[call] == expected
    # The CUDA spelling must not survive into a metal source.
    from pygpgpu.kernel.device_builtins import ATOMIC_BUILTINS

    emitted = ATOMIC_BUILTINS["atomic_add"][1]
    assert emitted == "atomicAdd", "the CUDA table is unchanged"
    assert not expected.startswith("atomicAdd")


def test_a_metal_atomic_names_its_address_space() -> None:
    """A `threadgroup` buffer and a `device` array take different atomics.

    MSL overloads the atomics on the reference's address space, and casting one
    space to the other does not compile -- the same rule OpenCL enforces with
    `__local` and `__global`.
    """
    from pygpgpu.kernel.device_builtins import _emit_atomic_metal

    for is_local, space in ((True, "threadgroup"), (False, "device")):
        emitted = _emit_atomic_metal("atomic_add", "&(buf)[i]", "1", None, "uint", is_local)
        assert f"({space} atomic_uint*)" in emitted
        assert "memory_order_relaxed" in emitted
        assert "atomicAdd" not in emitted
    # The two address spaces produce different text: they are not interchangeable.
    assert _emit_atomic_metal("atomic_add", "&(b)[i]", "1", None, "uint", True) != \
        _emit_atomic_metal("atomic_add", "&(b)[i]", "1", None, "uint", False)


def test_cas_is_the_compare_exchange_form() -> None:
    """MSL's compare-and-swap answers a bool and takes the expected value by pointer."""
    from pygpgpu.kernel.device_builtins import _emit_atomic_metal

    emitted = _emit_atomic_metal("atomic_cas", "&(a)[i]", "1", "0", "int", False)
    assert "atomic_compare_exchange_weak_explicit" in emitted
    # It needs a local to hold the expected value, which the call takes the address of.
    assert "_cas_expected" in emitted
    assert "atomicCAS" not in emitted
