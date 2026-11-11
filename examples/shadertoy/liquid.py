"""A second shader: a cheap liquid, mostly to prove the uniforms are per-launch live.

The picture moves by itself (``i_time`` drives the bands) and reacts to the pointer
(``i_mouse`` bends them), which is the two-uniform contract of an animated Shadertoy. Resize the
window and the bands re-space through the new ``i_resolution``.
"""

import math

from pygf import color3f, float2

from shadertoy_viewer import ShadertoyViewer


class Liquid(ShadertoyViewer):

    def mainImage(self, frag_coord: float2) -> color3f:
        """Bands that flow with time and bend around the pointer."""
        uv = frag_coord / self.i_resolution
        p = uv - float2(0.5, 0.5)

        mouse = self.i_mouse.xy / self.i_resolution
        to_mouse = mouse - uv
        pull = 0.15 * math.sin(12.0 * to_mouse.x * to_mouse.y + 3.0 * self.i_time)

        wave = math.sin(9.0 * p.y + 4.0 * p.x + self.i_time + pull)
        shape = 0.5 + 0.5 * wave
        deep = 0.2 + 0.6 * shape

        shade = min(shape * deep, 1.0)
        return color3f(0.1 * deep, 0.35 * shade, deep)
