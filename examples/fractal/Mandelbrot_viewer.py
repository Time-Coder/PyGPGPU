"""Mandelbrot viewer without perturbation theory, on top of :class:`FractalViewer`.

Every pixel iterates its own `c = left_top + j*value_per_pixel` in float64, so the picture stops
resolving once one pixel is finer than a double's 1e-16, which at 200 pixels per unit is a scale
of about 1e14. `Mandelbrot_viewer_high_precision.py` is the same picture with a reference orbit
and float64 deltas, which goes far deeper; this file is what that one is compared against.

The colouring is the one the promo film's fractal station renders with, and it needs the whole
frame before any pixel is painted, so a frame is two launches with a host step between them:

* ``measure`` iterates each pixel and records the log of its smooth escape count
  (``it + 2 - log2(log2|z|)``, continuous in the escape step, so no banded rings) and, from the
  orbit derivative `dz/dc`, the analytic distance from the pixel to the set boundary in pixels;
* the host normalizes the logged counts against this frame's own 3rd/99.7th percentiles -- a
  deep view and a wide view then fill the same palette, and neither saturates nor goes flat;
* ``paint`` maps the normalized value through a navy-blue-gold-white ramp, adds the distance
  estimate as a thin bright rim on the boundary, and leaves the interior black.

Unlike the film, the range is recomputed every frame with no smoothing in between: the picture
only changes when the camera does, and each frame is self-consistent on its own.
"""

import numpy as np

from pygpgpu import clamp, kernel, log, max, mix, smoothstep, sqrt
from pygpgpu import numpy as gnp
from pygf import float3

from fractal_viewer import FractalViewer

# Escape radius |z| > 256: the larger radius makes the smooth count smoother than 2 does.
BAILOUT = 65536.0


class MandelbrotViewer(FractalViewer):

    def __init__(self, init_width: int = 800, init_height: int = 600, title: str = "fractal") -> None:
        super().__init__(init_width, init_height, title)
        # The measured frame: channel 0 the log of the smooth escape count (-1 inside the
        # set), channel 1 the boundary-rim strength. Reallocated in draw() on a resize.
        self.field = gnp.zeros((init_height, init_width, 2), dtype=np.float32)

    @kernel(float=np.float64)
    def measure(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            c = self.point_of_pixel(i, j)
            z = complex(0.0, 0.0)
            der = complex(0.0, 0.0)
            it = 0
            escaped = 0
            mag = 0.0
            while it < self.max_iter:
                # dz/dc on the full value, for the distance estimate below.
                der = 2.0 * z * der + 1.0
                z = z * z + c
                mag = z.real * z.real + z.imag * z.imag
                if mag > BAILOUT:
                    escaped = 1
                    break
                it += 1

            value = -1.0
            edge = 0.0
            if escaped == 1:
                nu = float(it) + 2.0 - log(0.5 * log(mag) / log(2.0)) / log(2.0)
                value = log(nu)
                # Distance to the set in pixels: |z| log|z| / |dz/dc|, over the pixel size.
                dmod = sqrt(der.real * der.real + der.imag * der.imag)
                dist = sqrt(mag) * log(mag) * 0.5 / max(dmod, 1e-300)
                d = dist / self.value_per_pixel
                edge = 1.0 / (1.0 + d * d * 4.0)
            self.field[i, j, 0] = value
            self.field[i, j, 1] = edge

    @kernel
    def paint(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            value = self.field[i, j, 0]
            r = 0.0
            g = 0.0
            b = 0.0
            if value >= 0.0:
                v = clamp((value - self.lo) / self.span, 0.0, 1.0)
                # Deep navy far from the set, through blue and gold, to white at the rim.
                dark = float3(0.004, 0.006, 0.02)
                blue = float3(0.10, 0.22, 0.75)
                gold = float3(1.0, 0.66, 0.24)
                white = float3(1.0, 0.92, 0.82)
                col = mix(dark, blue, smoothstep(0.0, 0.55, v))
                col = mix(col, gold, smoothstep(0.5, 0.88, v))
                col = mix(col, white, smoothstep(0.86, 1.0, v))
                col = col + float3(0.6, 0.65, 0.9) * (self.field[i, j, 1] * 0.5 * v)
                r = clamp(col.x, 0.0, 1.0)
                g = clamp(col.y, 0.0, 1.0)
                b = clamp(col.z, 0.0, 1.0)
            self.image[i, j, 0] = 255.0 * r
            self.image[i, j, 1] = 255.0 * g
            self.image[i, j, 2] = 255.0 * b

    def draw(self) -> None:
        """One frame: measure the whole frame, pick its palette range, then paint."""
        height, width = int(self.image.shape[0]), int(self.image.shape[1])
        if self.field.shape[:2] != (height, width):
            self.field = gnp.zeros((height, width, 2), dtype=np.float32)
        self.measure()
        self.field.to_host(wait=True)
        values = np.array(self.field[:, :, 0], copy=True)
        escaped = values[values >= 0.0]
        if escaped.size:
            lo, hi = np.percentile(escaped, [3.0, 99.7])
        else:
            lo, hi = 0.0, 1.0
        self.lo = float(lo)
        self.span = max(float(hi) - float(lo), 1e-3)
        self.paint()


if __name__ == "__main__":
    MandelbrotViewer().show()
