"""The device side of the ``gpu`` namespace: how each builtin is emitted per backend.

This module is translation machinery, not API. The names a user writes live in
``builtins.py`` as dual-sided functions with real signatures; here are the tables the
translator turns those names into (OpenCL C or CUDA), plus the helpers that render one
call. Nothing here inspects a Python body, and nothing here is imported by user code.

The ``python`` backend does not use these tables at all: it runs the bodies from
``builtins.py`` instead."""
# This module is the facade of a split: every name it used to define is re-exported
# from the part module that now owns it, so `device_builtins.<name>` and
# `from pygpgpu.kernel.device_builtins import <name>` keep working unchanged.
#
# One object per builtin name, and the host and the device must agree on what it
# accepts: the tables here are keyed by backend and are read by the translator and by
# the tests that check the vocabulary, so their names are the contract.
from __future__ import annotations


from .index import (
    INDEX_BUILTINS,
    INDEX_VECTOR_AXES_CUDA,
    INDEX_VECTOR_AXES_OPENCL,
    INDEX_VECTOR_BUILTINS,
    INDEX_VECTOR_OPENCL,
    dim_component,
    emit_barrier,
    emit_index,
)
from .atomics import (
    ATOMIC_BUILTINS,
    FLOAT_ATOMIC_ADD_SOURCES,
    _ATOMIC_64_AS_UNSIGNED,
    _METAL_ATOMICS,
    _METAL_ATOMIC_TYPES,
    _emit_atomic_cuda_64,
    _emit_atomic_metal,
    emit_atomic,
    float_atomic_add_source,
)
from .math_tables import (
    CUDA_ONLY_BUILTINS,
    CUDA_ONLY_CONVERSIONS,
    GENERIC_BUILTINS,
    MATH_BUILTINS,
    SHARED_MATH_BUILTINS,
    SHARED_MATH_TEMPLATES,
)
from .floor import (
    HIP_ROUND_TOWARD_ZERO_HELPERS,
    HIP_ROUND_TOWARD_ZERO_SOURCES,
    PYTHON_FLOOR_QUALIFIERS,
    PYTHON_FLOOR_SOURCES,
    UNSIGNED_C_TYPES,
    c_identifier_suffix,
    python_floor_name,
    python_floor_source,
)
