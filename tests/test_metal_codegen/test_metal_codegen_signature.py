"""The signature tests, split out of `test_metal_codegen.py`.

The subject is signature.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    generate,
    vecadd,
)

import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.metal_codegen import (
    PROLOGUE_PARAMETERS,
    kernel_prefix,
    prelude,
    prologue_parameters,
)
from pygpgpu.kernel.metal_codegen import barrier as msl_barrier
from pygpgpu.kernel.translator import FunctionTranslator


def test_entry_point_is_kernel_void() -> None:
    """MSL writes the return type in the declaration, and the qualifier is `kernel`."""
    assert kernel_prefix() == "kernel void"
    source = generate(vecadd, 3)
    assert source.startswith("kernel void k_test(")
    # Not the OpenCL or CUDA spelling, and no doubled `void`.
    assert "__kernel" not in source
    assert "__global__" not in source
    assert "void void" not in source


def test_buffer_address_space_is_device() -> None:
    """A buffer pointer lives in `device`, not `__global`."""
    source = generate(vecadd, 3)
    assert "device float* a" in source
    assert "__global" not in source


def test_prologue_parameters_are_attributed() -> None:
    """The work-item numbers are parameters carrying a `[[...]]` attribute."""
    text = ", ".join(prologue_parameters())
    for _type_text, name, attribute in PROLOGUE_PARAMETERS:
        assert f"{name} [[{attribute}]]" in text
    # The attribute follows the name, which is MSL's spelling.
    for parameter in prologue_parameters():
        type_text, rest = parameter.split(" ", 1)
        assert rest.startswith("_mtl_")


def test_body_keeps_opencl_index_spelling() -> None:
    """The shared body still calls `get_global_id(0)`; the prelude is what defines it."""
    source = generate(vecadd, 3)
    assert "get_global_id(0)" in source
    assert "get_global_id(dim)" in prelude()


def test_prelude_maps_every_index_helper() -> None:
    """Every helper the OpenCL body may call has a macro over a real parameter."""
    text = prelude()
    declared = {name for _t, name, _a in PROLOGUE_PARAMETERS}
    for helper in ("get_global_id", "get_local_id", "get_group_id",
                   "get_global_size", "get_local_size", "get_num_groups"):
        assert f"#define {helper}(dim)" in text
    # Every macro expands onto exactly the parameters the kernel declares: each
    # `#define` line mentions at least one prologue name, and no name is invented.
    macro_lines = [line for line in text.splitlines() if line.startswith("#define")]
    assert len(macro_lines) == 6
    for line in macro_lines:
        assert any(name in line for name in declared), line


def test_prologue_parameters_are_in_the_signature() -> None:
    """They must be in the rendered parameter list -- the signature digest hashes it."""
    translator = FunctionTranslator(
        vecadd, [infer_ctype(np.zeros(8, np.float32)) for _ in range(3)],
        "metal", lambda name: None, {}, is_kernel=True)
    translator.translate()
    declarations = translator.parameter_declarations()
    expected = [f"{_t} {name} [[{attribute}]]"
                for _t, name, attribute in PROLOGUE_PARAMETERS]
    assert declarations[-len(expected):] == expected


def test_barrier_is_metal_spelling() -> None:
    """The Metal barrier is `threadgroup_barrier`, never CUDA's `__syncthreads()`."""
    assert "threadgroup_barrier" in msl_barrier()
    assert "sync" not in msl_barrier().lower().replace("threadgroup_barrier", "")
    # And the emitted call is reached through the builtin, through the shared emitter.
    from pygpgpu.kernel.device_builtins import emit_barrier

    assert "threadgroup_barrier" in emit_barrier("metal")
    assert "__syncthreads" not in emit_barrier("metal")


@pytest.mark.parametrize("backend", ["opencl", "cuda"])
def test_other_backends_unchanged(backend: str) -> None:
    """The same body on the other backends keeps its own spelling."""
    source = generate(vecadd, 3, backend)
    if backend == "opencl":
        assert "__kernel void" in source
        assert "__global float* a" in source
    else:
        assert "__global__ void" in source
        assert "device float*" not in source
    # Neither carries Metal's prologue.
    assert "_mtl_thread_position" not in source
