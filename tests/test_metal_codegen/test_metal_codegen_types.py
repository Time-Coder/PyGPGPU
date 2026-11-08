"""The types tests, split out of `test_metal_codegen.py`.

The subject is types.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    read_image,
    structured_add,
    vecadd,
    vector_scale,
    write_image,
)

import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.translator import FunctionTranslator


def test_an_image_parameter_is_refused_on_metal() -> None:
    """An image is refused by the translator, not left to fail in an MSL compiler.

    Metal *has* texture objects, but the whole image path is Metal's own -- the
    parameter type, the sampling call with its sampler state, the store, and the
    host-side `MTLTexture` binding. Until that exists the OpenCL formulation would
    reach the source as `__read_only image2d_t`, which MSL has no notion of, so the
    translator is where the refusal belongs: it can name the backends that work.
    """
    from pygpgpu.image import Image, ImageAddress, ImageFilter

    image = Image(np.zeros((8, 8), np.float32), address=ImageAddress.CLAMP,
                  filter=ImageFilter.POINT)
    types = [infer_ctype(image), infer_ctype(np.zeros(8, np.float32))]

    for func, action in ((read_image, "read"), (write_image, "write")):
        translator = FunctionTranslator(func, types, "metal", lambda name: None, {},
                                        is_kernel=True, launch_backend="metal")
        with pytest.raises(SyntaxError) as refused:
            translator.translate()
        message = str(refused.value)
        assert "metal" in message
        assert action in message
        # The message points at the backends that do support images.
        assert "opencl" in message and "cuda" in message

    # The backends that implement the image path are unaffected.
    for backend in ("opencl", "cuda"):
        translator = FunctionTranslator(read_image, types, backend, lambda name: None,
                                        {}, is_kernel=True, launch_backend=backend)
        translator.translate()
        source = "\n".join(s for _, s in translator.render_kernels("k"))
        assert "image2d_t" in source or "cudaTextureObject_t" in source


def test_a_structure_argument_is_a_device_const_pointer() -> None:
    """A structured array travels as `device const T*`, with the typedef ahead.

    MSL needs the record type before the pointer that names it, and it needs the
    macro prelude before the kernel body. The write order is therefore prelude ->
    typedef block -> kernel, which is what the decorator produces; a `__global`
    qualifier or a missing `typedef` would both fail the Metal compiler.
    """
    record = np.zeros(4, dtype=[("pos", np.float32), ("vel", np.float32)])
    types = [infer_ctype(record), infer_ctype(np.zeros(4, np.float32))]
    translator = FunctionTranslator(structured_add, types, "metal",
                                    lambda name: None, {}, is_kernel=True)
    translator.translate()
    source = "\n".join(piece for _, piece in translator.render_kernels("k"))
    assert "device const _st_" in source
    assert "__global" not in source

    # The record's own fields are the backend's scalar spellings.
    typedefs = translator.render_structure_typedefs()
    assert "float pos;" in typedefs
    assert "float vel;" in typedefs


def test_vector_parameters_and_arithmetic_are_msl() -> None:
    """A vector parameter is `device float4*`, and vector maths stays native.

    MSL has the same vector types OpenCL does, and the scalar type names it uses
    (`uchar`, `ushort`, `uint`, `ulong`, `half`) are exactly the translator's
    internal spellings -- which is why `_device_type` passes them through for
    Metal instead of applying the CUDA renaming. A vector times a scalar is one
    instruction there, so the body keeps `v * 2.0f` rather than expanding per
    component the way CUDA must.
    """
    from pygpgpu import float4

    vectors = np.zeros(4, dtype=float4)
    types = [infer_ctype(vectors) for _ in range(2)]
    translator = FunctionTranslator(vector_scale, types, "metal",
                                    lambda name: None, {}, is_kernel=True)
    translator.translate()
    source = "\n".join(piece for _, piece in translator.render_kernels("k"))
    assert "device float4* src" in source
    assert "device float4* out" in source
    # The multiply is not expanded into make_float4(...) components.
    assert "make_float4" not in source


def test_metal_scalar_type_spellings_are_msl_ones() -> None:
    """The scalar/vector names pass through: MSL spells them as OpenCL does."""
    translator = FunctionTranslator(
        vecadd, [infer_ctype(np.zeros(8, np.float32)) for _ in range(3)],
        "metal", lambda name: None, {}, is_kernel=True)
    expected = {
        "uchar": "uchar", "ushort": "ushort", "uint": "uint",
        "ulong": "ulong", "long": "long", "half": "half", "float": "float",
        "uchar4": "uchar4", "ushort4": "ushort4", "uint4": "uint4",
        "long4": "long4",
    }
    for internal, msl in expected.items():
        assert translator._device_type(internal) == msl, internal
