"""Mandelbrot viewer without perturbation theory, on top of :class:`FractalViewer`.

Every pixel iterates its own `c = left_top + j*value_per_pixel` in float64, so the picture stops
resolving once one pixel is finer than a double's 1e-16, which at 200 pixels per unit is a scale
of about 1e14. `Mandelbrot_viewer_high_precision.py` is the same picture with a reference orbit
and float64 deltas, which goes far deeper; this file is what that one is compared against.

Everything else -- the camera, the window, the colouring, panning and zooming -- is inherited.
All this subclass says is how long a point's orbit survives.
"""

import math

from fractal_viewer import FractalViewer


class MandelbrotViewer(FractalViewer):

    def escape_steps(self, c: complex) -> float:
        """The smooth escape count of ``c``: steps survived, with the last step fractioned."""
        zn = 0j
        abs_zn = abs(zn)
        n: int = 0
        while abs_zn <= 2 and n < self.max_iter:
            zn = zn ** 2 + c
            abs_zn = abs(zn)
            n += 1
        if n == self.max_iter or abs_zn <= 1.0:
            return float(n)
        return n + 1 - math.log2(math.log2(abs_zn))


if __name__ == "__main__":
    MandelbrotViewer().show()
