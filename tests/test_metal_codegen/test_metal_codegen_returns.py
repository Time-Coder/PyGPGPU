"""The returns tests, split out of `test_metal_codegen.py`.

The subject is returns.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    generate,
    returning,
)



def test_a_returning_kernel_uses_device_slots_and_a_metal_claim() -> None:
    """The return slots are `device` buffers and the claim word is a compare-exchange.

    Two things would be wrong without this. A `__global` slot would not compile in
    MSL, and `atomicCAS` does not exist there -- MSL's compare-and-swap takes the
    expected value by pointer and answers a bool, so the claim is read back from
    the word rather than taken as the call's result.
    """
    source = generate(returning, 2)

    # The slots the translator appends are device pointers, and no OpenCL qualifier
    # survives anywhere in the signature.
    assert "device float* _ret0" in source
    assert "device int* _ret_claim" in source
    assert "device int* _ret_site" in source
    assert "__global" not in source
    assert "atomicCAS" not in source

    # The claim is a compare-exchange plus a read-back of the word.
    assert "atomic_compare_exchange_weak_explicit" in source
    assert "_ret_claim[0] == 1" in source

    # The other backends keep their own spelling.
    opencl_source = generate(returning, 2, "opencl")
    assert "__global float* _ret0" in opencl_source
    assert "atomic_cmpxchg" in opencl_source
    cuda_source = generate(returning, 2, "cuda")
    assert "atomicCAS" in cuda_source
