"""The tables tests, split out of `test_hip_libraries.py`.

The subject is tables.
"""
from __future__ import annotations

from pygpgpu.hip.libraries import (
    HIPBLAS,
    driver_for,
    library_names,
)
from pygpgpu.hip.libraries._symbols import hip_symbol


def test_all_five_libraries_are_bound():
    """Every library whose entry points were checked against ROCm's headers."""
    assert library_names() == ["hipblas", "hipblaslt", "hipfft", "hipsparse",
                               "hipsolver"]


def test_the_library_tables_declare_the_entry_points_they_name():
    """Every entry is spelled with its library's own prefix and declares a return type.

    The argument list may legitimately be empty (`GetVersion` in some libraries), so only
    the prefix and the return type are asserted -- those are what a mistyped entry would
    get wrong.
    """
    prefixes = {"hipblas": "hipblas", "hipblaslt": "hipblasLt", "hipfft": "hipfft",
                "hipsparse": "hipsparse", "hipsolver": "hipsolver"}
    for name in library_names():
        driver = driver_for(name)
        assert driver._signatures, f"{name} declares no entry points"
        for entry, signature in driver._signatures.items():
            assert "restype" in signature, f"{name}.{entry} declares no return type"
            assert entry.startswith(prefixes[name]), (
                f"{name}.{entry} is not spelled with the library's own prefix")


def test_the_symbol_mapping_drops_the_abi_suffix():
    """hipBLAS has no second ABI, so CUDA 12's `_v2` is not part of the name."""
    assert hip_symbol("hipblas", "cublasSgemm_v2") == "hipblasSgemm"
    assert hip_symbol("hipblas", "cublasCreate_v2") == "hipblasCreate"
    assert hip_symbol("hipblas", "cublasDgemm_v2") == "hipblasDgemm"


def test_the_symbol_mapping_uses_the_camel_case_prefix_for_hipblaslt():
    """hipBLASLt's DLL is `hipblaslt`, but its entry points are `hipblasLt*`."""
    assert hip_symbol("hipblaslt", "cublasLtMatmul") == "hipblasLtMatmul"
    assert hip_symbol("hipblaslt", "cublasLtCreate") == "hipblasLtCreate"


def test_a_symbol_the_library_lacks_has_no_mapping():
    """`GetStatusString` is absent from every HIP library."""
    assert hip_symbol("hipblas", "cublasGetStatusString") is None
    assert hip_symbol("hipblaslt", "cublasLtGetStatusString") is None


def test_the_signature_table_is_derived_from_the_cuda_one():
    """The API is cuBLAS's, so the entries are the ones the CUDA table has."""
    from pygpgpu.cuda.libraries import cublas

    hip_names = set(HIPBLAS._signatures)
    # every CUDA entry whose HIP counterpart exists is present, under its HIP name
    for cuda_name in cublas.CUBLASInfo.func_signatures:
        translated = hip_symbol("hipblas", cuda_name)
        if translated is not None:
            assert translated in hip_names
    assert "hipblasSgemm" in hip_names
    assert "cublasSgemm_v2" not in hip_names


def test_the_hip_table_does_not_share_entries_with_the_cuda_one():
    """A shared entry would carry one library's bound function into the other.

    The binding machinery caches the bound function in the entry it was called
    through, so sharing would make a cuBLASLt call land in hipBLASLt (or the reverse).
    """
    from pygpgpu.cuda.libraries import cublas

    for name, entry in HIPBLAS._signatures.items():
        cuda_name = "cublas" + name[len("hipblas"):]
        cuda_entry = cublas.CUBLASInfo.func_signatures.get(cuda_name)
        if cuda_entry is not None:
            assert entry is not cuda_entry, (
                f"{name} shares its entry with {cuda_name}; a bound function cached in "
                f"one would be used by the other")
