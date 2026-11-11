"""A first shader: the unit square as a colour wheel, and the pointer as a lamp.

The shader reads ``frag_coord``, ``i_resolution``, ``i_time`` and ``i_mouse`` -- the mouse in
Shadertoy's own float4 packing, xy the pointer and w the button state. The lamp follows the
pointer while the button is held and stays where it was released, because the viewer freezes
the uniform on release.
"""

import math

from pygf import color3f, float2

from shadertoy_viewer import ShadertoyViewer


class ColorWheel(ShadertoyViewer):

    def mainImage(self, frag_coord: float2) -> color3f:
        """Hue around the centre, brightness by radius, dimmed by the pointer's lamp."""
        uv = frag_coord / self.i_resolution
        p = uv - float2(0.5, 0.5)
        angle = math.atan2(p.y, p.x) / 6.28318530718 + 0.5
        radius = 2.0 * math.sqrt(p.x * p.x + p.y * p.y)

        mouse_uv = self.i_mouse.xy / self.i_resolution
        d = uv - mouse_uv
        lamp = 0.4 / (0.04 + 12.0 * (d.x * d.x + d.y * d.y))

        pulse = 0.75 + 0.25 * math.sin(6.28318530718 * (radius - self.i_time))
        glow = 0.35 + lamp if self.i_mouse.w > 0.5 else 0.35
        return color3f(angle * glow, radius * glow, (1.0 - angle) * pulse * glow)


if __name__ == "__main__":
    ColorWheel().show()
