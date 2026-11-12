"""Bug: a pygf vector's components cannot be assigned in device code.

`ComponentStore.draw` builds a colour and then sets one component -- the shape every GLSL
shader writes (`color.r = ...`). The translator used to refuse it:

    SyntaxError: cannot assign to 'c.x': member stores need a structure value, and float3 is
    not one

The same refusal answers a swizzle store (`c.xy = uv`). The host side mutates the very same
object freely (`v.x = 9.0` on a `float2` works), and the error's own wording says what is
missing: a *structure value* is what member stores accept, and the pygf vectors are not
recognised as one. The right answer is for the store to spell the component write the record
layout already has -- not to send the shader author through "assign a whole new vector", which
is what the shadertoy viewer has to document.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygf import color3f, float2


class ComponentStore:
    """A shader that assembles its colour one component at a time."""

    def __init__(self) -> None:
        self.image = np.zeros((2, 3, 3), dtype=np.uint8)

    def mainImage(self, frag_coord: float2) -> color3f:
        uv = frag_coord / float2(3.0, 2.0)
        c = color3f(0.0, 0.0, 0.0)
        c.x = uv.x
        c.y = uv.y
        return c

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            color = self.mainImage(float2(float(j) + 0.5, float(i) + 0.5))
            self.image[i, j, 0] = 255 * color.r
            self.image[i, j, 1] = 255 * color.g
            self.image[i, j, 2] = 255 * color.b
