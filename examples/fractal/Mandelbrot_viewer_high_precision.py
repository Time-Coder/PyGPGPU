"""The Mandelbrot viewer with a reference orbit, on top of :class:`FractalViewer`.

Perturbation theory renders a pixel by its offset `dc = c - c_ref` from a reference point
instead of by `c` itself:

    z = Z + d        d' = 2*Z*d + d*d + dc

`dc` is the difference of two nearby points, so it keeps every digit of the zoom however small
it gets, while iterating `c` directly rounds it away once a pixel is finer than a double
(scale ~1e14). This is what lets the viewer keep resolving long past the point where
`Mandelbrot_viewer.py` turns into mush.

Two halves:

* the host picks a reference point and iterates *its* orbit in arbitrary precision once per
  frame (`update_reference`);
* the device colours every pixel from its float64 offset against that orbit
  (`color_of_pixel`). That is why this viewer overrides `color_of_pixel` rather than
  `color_of_complex`: at depth, one double cannot name the point a pixel shows, so there is no
  meaningful `c` to hand `color_of_complex`.

The camera, the window, panning and zooming are inherited; so is everything a subclass needs.
"""

from typing import Tuple

import numpy as np
import mpmath as mp

from pygf import color3f

from fractal_viewer import FractalViewer

# Digits kept for the view centre. A double render dies where one pixel (1/(200*scale)) falls
# under a double's 1e-16, i.e. around scale 1e14; the camera itself has to name a point finer
# than that, so it is kept in arbitrary precision. The reference orbit only needs as many digits
# as the current zoom needs (see reference_orbit).
CAMERA_DIGITS = 400

# Extra steps kept past `max_iter`. The budget grows with every wheel notch, and rebuilding the
# reference every notch would make its rounding (and so the colour of every pixel sitting on a
# band edge) jump while the view moves smoothly. With a margin the same orbit serves the next few
# decades of zooming.
REFERENCE_MARGIN = 512


class MandelbrotViewer(FractalViewer):

    def __init__(self, init_width: int = 800, init_height: int = 600, title: str = "fractal") -> None:
        # The camera is arbitrary precision, so the centre is kept as two mpf values rather
        # than as one python complex; `view_center` is rebuilt from them for the readout and
        # for anything that still expects a complex.
        self.center_re = mp.mpf(0)
        self.center_im = mp.mpf(0)
        # The reference point, its orbit, and where that orbit sits in pixels relative to the
        # view centre (update_reference fills all of it in).
        self.ref_re = mp.mpf(0)
        self.ref_im = mp.mpf(0)
        self.ref_z = np.zeros(1, dtype=np.complex128)
        self.ref_escaped_at = 0
        self.ref_dx = 0.0
        self.ref_dy = 0.0
        self.orbit_digits = 0
        super().__init__(init_width, init_height, title)

    # -- camera -----------------------------------------------------------

    def update_params(self) -> None:
        self.value_per_pixel = 1 / (self.scale * self.default_pixels_per_value)
        self.max_iter: int = int(100 * np.log10(1 + self.scale) + 500)
        # One digit per decade of zoom, plus guard digits: the deltas are float64, so digits
        # past that are what the *reference point* itself needs. It only ever grows, so zooming
        # back out does not change the rounding of a kept orbit.
        needed: int = max(20, int(np.log10(max(self.scale, 1.0))) + 20)
        self.orbit_digits = max(self.orbit_digits, needed)
        self.half_width = 0.5 * self.image.shape[1]
        self.half_height = 0.5 * self.image.shape[0]
        # A plain complex view centre, so the inherited status text and point_of_pixel agree
        # with the camera. It is only a double, so it is not what the reference is built from.
        self.view_center = complex(float(self.center_re), float(self.center_im))
        self.left_top_complex = complex(
            self.view_center.real - 0.5 * self.value_per_pixel * self.image.shape[1],
            self.view_center.imag + 0.5 * self.value_per_pixel * self.image.shape[0]
        )

    def status_text(self) -> str:
        with mp.workdps(20):
            return (f"center: {mp.nstr(self.center_re, 4)}+"
                    f"{mp.nstr(self.center_im, 4)}j, scale: {self.scale:.2g}")

    # -- the reference orbit ----------------------------------------------

    def reference_orbit(self, c_re: mp.mpf, c_im: mp.mpf) -> Tuple[np.ndarray, int]:
        """The orbit of one point, in arbitrary precision, as float64 for the device.

        Only the *reference* orbit has to be accurate; the deltas stay in float64, where their
        relative accuracy (~1e-16) is what the pixel grid needs. Returns the orbit and the step
        count at which it escaped (`max_iter + REFERENCE_MARGIN` when it never does).
        """
        orbit = np.empty(self.max_iter + REFERENCE_MARGIN + 1, dtype=np.complex128)
        escaped_at = self.max_iter + REFERENCE_MARGIN
        with mp.workdps(self.orbit_digits):
            c = mp.mpc(c_re, c_im)
            z = mp.mpc(0)
            for n in range(self.max_iter + REFERENCE_MARGIN + 1):
                orbit[n] = complex(z)
                z = z * z + c
                if abs(z) > 2:
                    escaped_at = n + 1
                    # Past its escape step the reference only feeds pixels it does not
                    # represent (the perturbation 'glitch'); freezing the last value keeps
                    # every later delta finite instead of inf/NaN.
                    orbit[n + 1:] = complex(z)
                    break
        return orbit, escaped_at

    def reference_offsets(self) -> None:
        """The reference point's position in pixels, relative to the view centre."""
        with mp.workdps(CAMERA_DIGITS):
            px = (self.ref_re - self.center_re) / mp.mpf(self.value_per_pixel)
            py = (self.ref_im - self.center_im) / mp.mpf(self.value_per_pixel)
        self.ref_dx, self.ref_dy = float(px), float(py)

    def update_reference(self) -> None:
        """Choose the reference point and rebuild its orbit, once per frame.

        This is the host-side half of `draw`, and it has to stay out of the kernel body: the
        values bound above a magic loop become kernel parameters, and the object scan walks
        every method the body calls, so an arbitrary-precision value reachable from the body is
        refused ("cannot become a structure member").

        The reference has to *outlive* the pixels it stands for: a pixel whose orbit outlasts
        the reference's is rendered from the wrong recurrence (the classic perturbation glitch).
        A reference that survives the budget is therefore kept -- across panning, and across
        zooming as long as the margin covers the grown budget -- because changing the reference
        changes how every pixel near a band edge rounds, which is visible as colours jumping
        while the view moves smoothly. When one has to be picked, the point already in use is
        tried first, then a 5x5 grid from the centre outwards, and the first survivor wins.
        """
        reach = 2.0 * max(self.half_width, self.half_height)
        in_view = max(abs(self.ref_dx), abs(self.ref_dy)) <= reach
        if (self.ref_escaped_at >= self.max_iter
                and len(self.ref_z) >= self.max_iter + 1 and in_view):
            self.reference_offsets()
            return

        points = []
        if self.ref_escaped_at:                       # there is a point in use already
            points.append((self.ref_re, self.ref_im))
        for k, l in [(0, 0)] + [(k, l) for k in (-1, 1) for l in (-1, 1)] \
                + [(k, 0) for k in (-1, 1)] + [(0, l) for l in (-1, 1)] \
                + [(k, l) for k in (-2, 2) for l in (-2, 2)] \
                + [(k, l) for k in (-2, 2) for l in (-1, 1)] \
                + [(k, l) for k in (-1, 1) for l in (-2, 2)]:
            px = 0.45 * k * self.half_width
            py = 0.45 * l * self.half_height
            with mp.workdps(CAMERA_DIGITS):
                points.append((self.center_re + mp.mpf(px) * mp.mpf(self.value_per_pixel),
                               self.center_im + mp.mpf(py) * mp.mpf(self.value_per_pixel)))

        best = None
        for c_re, c_im in points:
            orbit, escaped_at = self.reference_orbit(c_re, c_im)
            if best is None or escaped_at > best[1]:
                best = (orbit, escaped_at, c_re, c_im)
            if escaped_at >= self.max_iter:
                break

        # `points` always holds at least the 25-point grid above, so the loop has
        # set `best`; the assertion states that invariant for the reader as well.
        assert best is not None
        self.ref_z, self.ref_escaped_at, self.ref_re, self.ref_im = best
        self.reference_offsets()

    # -- colouring --------------------------------------------------------

    @staticmethod
    def escape_steps_delta(dc: complex, ref_z: np.ndarray, max_iter: int) -> float:
        """Smooth escape count of the point `dc` away from the reference orbit."""
        delta = 0j
        abs_zn = 0.0
        n: int = 0
        while abs_zn <= 2 and n < max_iter:
            zn = ref_z[n]
            delta = 2.0 * zn * delta + delta * delta + dc
            n += 1
            abs_zn = abs(ref_z[n] + delta)

        if n == max_iter or abs_zn <= 1.0:
            return float(n)

        return n + 1 - np.log2(np.log2(abs_zn))

    def color_of_pixel(self, i: int, j: int) -> color3f:
        """Colour pixel ``(i, j)`` from its offset against the reference orbit.

        The palette is the base class's; only the escape count is this viewer's own, and it
        takes the offset rather than the point itself (see the module docstring).
        """
        dc = complex(
            (j - self.half_width - self.ref_dx) * self.value_per_pixel,
            (self.half_height - i - self.ref_dy) * self.value_per_pixel
        )
        return self.jet_color(self.escape_steps_delta(dc, self.ref_z, self.max_iter) / self.max_iter)

    # -- redraw and camera moves ------------------------------------------

    def redraw(self) -> None:
        """Rebuild the reference for this camera, then render as usual."""
        self.update_reference()
        super().redraw()

    def pan_by_pixels(self, dx: float, dy: float) -> None:
        """Pan at the camera's precision.

        A pan is a step of a few pixels: at depth it has to land in the centre's low digits,
        which a double no longer has.
        """
        with mp.workdps(CAMERA_DIGITS):
            self.center_re -= mp.mpf(dx) * mp.mpf(self.value_per_pixel)
            self.center_im += mp.mpf(dy) * mp.mpf(self.value_per_pixel)
        self.update_params()

    def zoom_at_pixel(self, x: float, y: float, steps: float) -> None:
        """Zoom about the pointer, at the camera's precision."""
        with mp.workdps(CAMERA_DIGITS):
            offset_re = mp.mpf(x) - mp.mpf(0.5 * self.image.shape[1])
            offset_im = mp.mpf(0.5 * self.image.shape[0]) - mp.mpf(y)
            # The point under the cursor, then the centre that keeps it under the cursor.
            mouse_re = self.center_re + offset_re * mp.mpf(self.value_per_pixel)
            mouse_im = self.center_im + offset_im * mp.mpf(self.value_per_pixel)

            self.scale *= self.zoom_step ** steps
            self.update_params()
            self.center_re = mouse_re - offset_re * mp.mpf(self.value_per_pixel)
            self.center_im = mouse_im - offset_im * mp.mpf(self.value_per_pixel)
        self.update_params()


if __name__ == "__main__":
    MandelbrotViewer().show()
