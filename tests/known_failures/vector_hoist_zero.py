"""Bug: a pygf vector local whose first assignment sits in a block is hoisted as ``(T)0``.

A python local belongs to the function, so the translator hoists its C declaration to the top
whenever the first assignment sits inside a braced block -- and spells the hoisted zero the
scalar way, ``(T)0``. For a pygf vector that is not valid C++: CUDA's ``float2``/``float3`` are
structures, and ``(float3)0`` is refused with

    error: no suitable constructor exists to convert from "int" to "float3"

The repo already knows this for records -- a hoisted record is initialised ``{0}`` for exactly
this reason (see the hoist tests) -- but pygf vectors fall between the two paths: they are not
translator records and not scalars, so they get the scalar spelling. The shape is ordinary GLSL
(``vec3 col; ... col = ...`` inside a loop) and this is what blocks the Light Cycle port, whose
ray-march loop assigns ``p``, ``res`` and friends inside the loop body.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygf import color3f, float2, float3


class HoistedVector:
    """A shader whose colour is assembled inside a branch, then read after it."""

    def __init__(self) -> None:
        self.image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.i_resolution = float2(3.0, 2.0)

    def mainImage(self, frag_coord: float2) -> color3f:
        uv = frag_coord / self.i_resolution
        if uv.x > 0.25:
            c = float3(1.0, 0.0, 0.0)
        else:
            c = float3(0.0, 1.0, 0.0)
        return color3f(c.x, c.y, c.z)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            color = self.mainImage(float2(float(j) + 0.5, float(i) + 0.5))
            self.image[i, j, 0] = 255 * color.r
            self.image[i, j, 1] = 255 * color.g
            self.image[i, j, 2] = 255 * color.b
