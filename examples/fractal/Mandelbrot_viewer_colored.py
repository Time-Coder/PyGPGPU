"""The Mandelbrot viewer with the two flat regions coloured, on top of :class:`FractalViewer`.

`Mandelbrot_viewer.py` is deliberately minimal and pays for it: the escape count is divided by
`max_iter`, and at any ordinary zoom almost every pixel escapes in a handful of steps, so the
whole exterior lands on one dark tone, while every pixel of the set lands on another one. Two
large areas of the picture are therefore flat colour -- the far field, where one or two steps is
all the escape there is, and the set itself, which has no escape count at all.

This file keeps that viewer's structure and only changes what a pixel's colour is made of:

* outside the set, the palette is a white-to-blue gradient riding on a cyclic smooth escape
  value that repeats every `band` steps, so the bands keep their spacing however deep the zoom
  goes; the phase also carries the logarithmic distance from the origin, which is what gives the
  far field its gradient -- out there the escape value is almost constant, and no function of it
  alone can be anything but one tone;
* inside the set there is no escape count, so the orbit itself does the shading: the mean orbit
  radius grows from the attracting centre out to the rim, and the angle of the last orbit point
  draws the fine filaments.

Everything else -- the camera, the window, panning and zooming -- is inherited.
"""

import math

from pygf import color3f

from fractal_viewer import FractalViewer


class MandelbrotViewer(FractalViewer):

    def color_of_complex(self, c: complex) -> color3f:
        """Exterior shades from the escape count, interior from the orbit itself.

        The orbit is iterated right here rather than through a shared escape-count helper,
        because the interior needs two things a count does not carry: the sum of the orbit radii
        and the last orbit point.
        """
        zn = 0j
        abs_zn = abs(zn)
        n: int = 0
        sum_r: float = 0.0
        while abs_zn <= 2 and n < self.max_iter:
            zn = zn ** 2 + c
            abs_zn = abs(zn)
            n += 1
            sum_r += abs_zn

        if n < self.max_iter:
            return self.exterior_color(n, abs_zn, c)
        return self.interior_color(sum_r, zn, self.max_iter)

    @staticmethod
    def exterior_color(n: int, abs_zn: float, c: complex) -> color3f:
        # Smooth escape value: the step count with the last, partial step resolved from how far
        # past the escape radius the orbit went.
        t: float = float(n) + 1 - math.log2(math.log2(abs_zn))
        # The palette repeats every `band` steps, so the bands keep their spacing at any zoom
        # depth instead of collapsing into the end of a once-per-render ramp. The logarithmic
        # distance from the origin is part of the phase for the far field, where `t` is nearly
        # constant and a colour read from it alone is one flat tone.
        band: float = 16.0
        x: float = t / band + 0.6 * math.log2(abs(c))
        u: float = x - math.floor(x)
        # White-to-blue: the gradient rides on five white ridges per cycle (`|sin|^0.6` gives
        # sharp white crests over wide blue valleys), and the crest width is what stops a
        # two-colour ramp from flattening into broad monotone bands.
        val: float = abs(math.sin(3.14159265 * u * 5.0)) ** 0.6
        # Faded by how far out the pixel escaped, so the neighbourhood of the set carries the
        # light and the picture has depth.
        fade: float = 0.55 + 0.45 * (1 - math.exp(-t / 6))
        return color3f(
            fade * (0.05 + 0.95 * val),
            fade * (0.20 + 0.80 * val),
            fade * (0.95 + 0.05 * val)
        )

    @staticmethod
    def interior_color(sum_r: float, zn: complex, max_iter: int) -> color3f:
        # Mean orbit radius: small at the attracting centre, larger at the rim, so it is the
        # height of the bulb rather than a uniform fill.
        mean_r: float = sum_r / float(max_iter)
        glow: float = min(1.0, 2.0 * mean_r)
        # The last orbit point of a bounded orbit still circles the attracting cycle, and its
        # angle is what draws the filaments.
        ang: float = math.atan2(zn.imag, zn.real)
        fil: float = 0.5 + 0.5 * math.sin(6.0 * ang + 3.0 * glow)
        # Deep navy, the same family as the exterior's blue, so the set reads as a darker
        # silhouette inside it and its outline stays legible.
        return color3f(
            0.01 + 0.30 * glow * fil,
            0.03 + 0.12 * glow,
            0.12 + 0.50 * glow
        )


if __name__ == "__main__":
    MandelbrotViewer().show()
