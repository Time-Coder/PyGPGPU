from typing import Optional, Tuple

import numpy as np
import math
import mpmath as mp

from pygpgpu import kernel, config
from pygf import float2, color3f

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent, ResizeEvent
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from matplotlib.figure import Figure

config.print_info = False
config.type_map = {float: np.float64}

# Digits kept for the view centre. A double render dies where one pixel
# (1/(200*scale)) falls under a double's 1e-16, i.e. around scale 1e14; the camera
# itself has to name a point finer than that, so it is kept in arbitrary precision.
# The reference orbit only needs as many digits as the current zoom needs (see
# MandelbrotViewer.reference_orbit).
CAMERA_DIGITS = 400

# Extra steps kept past `max_iter`. The budget grows with every wheel notch, and
# rebuilding the reference every notch would make its rounding (and so the colour of
# every pixel sitting on a band edge) jump while the view moves smoothly. With a
# margin the same orbit serves the next few decades of zooming.
REFERENCE_MARGIN = 512

class MandelbrotViewer:

    def __init__(self, init_width:int=800, init_height:int=600):
        self.default_pixels_per_value = 200
        self.image = np.zeros((init_height, init_width, 3), dtype=np.uint8)
        self.center_re = mp.mpf(0)
        self.center_im = mp.mpf(0)
        self.scale: float = 1.0
        # The reference point, its orbit, and where that orbit sits in pixels relative
        # to the view centre (update_reference fills all of it in).
        self.ref_re = mp.mpf(0)
        self.ref_im = mp.mpf(0)
        self.ref_z = np.zeros(1, dtype=np.complex128)
        self.ref_escaped_at = 0
        self.ref_dx = 0.0
        self.ref_dy = 0.0
        self.orbit_digits = 0

        self.is_left_pressed: bool = False
        self.left_pressed_pos: float2 = float2()

        self.axes: Optional[Axes] = None
        self.figure: Optional[Figure] = None
        self.image_handle: Optional[AxesImage] = None
        self.text_handle = None

        self.update_params()

    def update_params(self):
        self.value_per_pixel = 1 / (self.scale * self.default_pixels_per_value)
        self.max_iter: int = int(100 * np.log10(1 + self.scale) + 500)
        # One digit per decade of zoom, plus guard digits: the deltas are float64, so
        # digits past that are what the *reference point* itself needs. It only ever
        # grows, so zooming back out does not change the rounding of a kept orbit.
        needed: int = max(20, int(np.log10(max(self.scale, 1.0))) + 20)
        self.orbit_digits: int = max(self.orbit_digits, needed)
        self.half_width = 0.5 * self.image.shape[1]
        self.half_height = 0.5 * self.image.shape[0]

    def center_text(self) -> str:
        with mp.workdps(20):
            return (f"center: {mp.nstr(self.center_re, 4)}+"
                    f"{mp.nstr(self.center_im, 4)}j, scale: {self.scale:.2g}")

    def reference_orbit(self, c_re: mp.mpf, c_im: mp.mpf) -> Tuple[np.ndarray, int]:
        """The orbit of one point, in arbitrary precision, as float64 for the device.

        Perturbation theory renders a pixel by its offset `dc = c - c_ref` from the
        reference point instead of by `c` itself:

            z = Z + d        d' = 2*Z*d + d*d + dc

        `dc` is the difference of two nearby points, so it keeps every digit of the
        zoom however small it gets, while iterating `c` directly rounds it away once a
        pixel is finer than a double (scale ~1e14). Only the *reference* orbit has to
        be accurate; the deltas stay in float64, where their relative accuracy
        (~1e-16) is what the pixel grid needs. Returns the orbit and the step count at
        which it escaped (`max_iter + REFERENCE_MARGIN` when it never does).
        """
        orbit = np.empty(self.max_iter + REFERENCE_MARGIN + 1, dtype=np.complex128)
        escaped_at = self.max_iter + REFERENCE_MARGIN
        with mp.workdps(self.orbit_digits):
            c = mp.mpc(c_re, c_im)
            z = mp.mpc(0)
            for n in range(self.max_iter + REFERENCE_MARGIN + 1):
                orbit[n] = complex(z)
                z = z*z + c
                if abs(z) > 2:
                    escaped_at = n + 1
                    # Past its escape step the reference only feeds pixels it does not
                    # represent (the perturbation 'glitch'); freezing the last value
                    # keeps every later delta finite instead of inf/NaN.
                    orbit[n + 1:] = complex(z)
                    break
        return orbit, escaped_at

    def reference_offsets(self) -> None:
        """The reference point's position in pixels, relative to the view centre."""
        with mp.workdps(CAMERA_DIGITS):
            px = (self.ref_re - self.center_re)/mp.mpf(self.value_per_pixel)
            py = (self.ref_im - self.center_im)/mp.mpf(self.value_per_pixel)
        self.ref_dx, self.ref_dy = float(px), float(py)

    def update_reference(self) -> None:
        """Choose the reference point and rebuild its orbit, once per frame.

        This is the host-side half of `draw`, and it has to stay out of the kernel
        body: the values bound above a magic loop become kernel parameters, and the
        object scan walks every method the body calls, so an arbitrary-precision
        value reachable from the body is refused ("cannot become a structure
        member"). The orbit itself is a plain complex128 member array, which the
        kernel reads like any other.

        The reference has to *outlive* the pixels it stands for: a pixel whose orbit
        outlasts the reference's is rendered from the wrong recurrence (the classic
        perturbation glitch). A reference that survives the budget is therefore kept --
        across panning, and across zooming as long as the margin covers the grown budget
        -- because changing the reference changes how every pixel near a band edge
        rounds, which is visible as colours jumping while the view moves smoothly. When
        one has to be picked, the point already in use is tried first, then a 5x5 grid
        from the centre outwards, and the first survivor wins.
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
            px = 0.45*k*self.half_width
            py = 0.45*l*self.half_height
            with mp.workdps(CAMERA_DIGITS):
                points.append((self.center_re + mp.mpf(px)*mp.mpf(self.value_per_pixel),
                               self.center_im + mp.mpf(py)*mp.mpf(self.value_per_pixel)))

        best = None
        for c_re, c_im in points:
            orbit, escaped_at = self.reference_orbit(c_re, c_im)
            if best is None or escaped_at > best[1]:
                best = (orbit, escaped_at, c_re, c_im)
            if escaped_at >= self.max_iter:
                break

        self.ref_z, self.ref_escaped_at, self.ref_re, self.ref_im = best
        self.reference_offsets()

    @staticmethod
    def escape_steps(dc: complex, ref_z: np.ndarray, max_iter: int) -> float:
        """Smooth escape count of the point `dc` away from the reference orbit."""
        delta = 0j
        abs_zn = 0.0
        n: int = 0
        while abs_zn <= 2 and n < max_iter:
            zn = ref_z[n]
            # `2*zn` rather than `2.0*zn`: scaling a complex value by a float currently
            # emits a helper the complex header does not define.
            delta = 2*zn*delta + delta*delta + dc
            n += 1
            abs_zn = abs(ref_z[n] + delta)

        if n == max_iter or abs_zn <= 1.0:
            return float(n)

        smooth = n + 1 - np.log2(np.log2(abs_zn))
        return smooth

    @staticmethod
    def smooth_jet_colormap(x: float)->color3f:
        return color3f(
            math.exp( - 60 * abs(x-3/4)**3.2 ),
            math.exp( - 60 * abs(x-2/4)**3.2 ),
            math.exp( - 60 * abs(x-1/4)**3.2 )
        )

    @kernel
    def draw(self):
        # `self.ref_z` is the reference orbit `update_reference()` rebuilt for this
        # frame, `ref_dx/ref_dy` where that reference sits in pixels; a pixel is its
        # offset from the reference (see escape_steps).
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            dc = complex(
                (j - self.half_width - self.ref_dx)*self.value_per_pixel,
                (self.half_height - i - self.ref_dy)*self.value_per_pixel
            )
            steps = self.escape_steps(dc, self.ref_z, self.max_iter)
            color = self.smooth_jet_colormap(steps/self.max_iter)
            self.image[i, j, 0] = 255 * color.r
            self.image[i, j, 1] = 255 * color.g
            self.image[i, j, 2] = 255 * color.b

    def show(self):
        self.update_reference()
        self.draw["cuda"]()
        plt.rcParams['toolbar'] = 'None'
        
        self.figure = plt.figure()
        self.figure.canvas.mpl_connect("resize_event", self.on_resized)
        self.figure.canvas.mpl_connect("button_press_event", self.on_pressed)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_moved)
        self.figure.canvas.mpl_connect("button_release_event", self.on_released)
        self.figure.canvas.mpl_connect("scroll_event", self.on_scrolled)

        self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.axes.set_aspect("auto", adjustable="box")
        self.axes.set_anchor("C")
        self.axes.set_autoscale_on(False)
        self.axes.axis('off')

        self.image_handle = self.axes.imshow(self.image)
        self.text_handle = self.axes.text(5, 15, self.center_text(), color='white')
        plt.show()

    def on_resized(self, event: ResizeEvent)->None:
        self.image = np.zeros((event.height, event.width, 3), dtype=np.uint8)
        self.update_params()

        self.update_reference()
        self.draw["cuda"]()
        self.image_handle.set_data(self.image)
        self.text_handle.set_text(self.center_text())
        self.image_handle.set_extent([0, event.width, event.height, 0])

        self.axes.set_xlim(0, event.width)
        self.axes.set_ylim(event.height, 0)
        self.figure.canvas.draw_idle()

    def redraw(self) -> None:
        self.update_reference()
        self.draw["cuda"]()
        self.image_handle.set_data(self.image)
        self.text_handle.set_text(self.center_text())
        self.figure.canvas.draw_idle()

    def on_pressed(self, event: MouseEvent) -> None:
        if event.button != MouseButton.LEFT or event.xdata is None or event.ydata is None:
            return
        self.is_left_pressed = True
        self.left_pressed_pos = float2(event.xdata, event.ydata)

    def on_moved(self, event: MouseEvent) -> None:
        if not self.is_left_pressed or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.left_pressed_pos.x
        dy = event.ydata - self.left_pressed_pos.y
        self.left_pressed_pos = float2(event.xdata, event.ydata)
        # A pan is a step of a few pixels: at depth it has to land in the centre's low
        # digits, which a double no longer has, so it is added at the camera's precision.
        with mp.workdps(CAMERA_DIGITS):
            self.center_re -= mp.mpf(dx) * mp.mpf(self.value_per_pixel)
            self.center_im += mp.mpf(dy) * mp.mpf(self.value_per_pixel)
        self.update_params()
        self.redraw()

    def on_released(self, event: MouseEvent) -> None:
        if event.button == MouseButton.LEFT:
            self.is_left_pressed = False

    def on_scrolled(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None or event.step == 0:
            return
        
        with mp.workdps(CAMERA_DIGITS):
            offset_re = mp.mpf(event.xdata) - mp.mpf(0.5*self.image.shape[1])
            offset_im = mp.mpf(0.5*self.image.shape[0]) - mp.mpf(event.ydata)
            # The point under the cursor, then the centre that keeps it under the cursor.
            mouse_re = self.center_re + offset_re*mp.mpf(self.value_per_pixel)
            mouse_im = self.center_im + offset_im*mp.mpf(self.value_per_pixel)

            self.scale *= 1.2 ** event.step
            self.update_params()
            self.center_re = mouse_re - offset_re*mp.mpf(self.value_per_pixel)
            self.center_im = mouse_im - offset_im*mp.mpf(self.value_per_pixel)
        self.update_params()
        self.redraw()


mandelbrot = MandelbrotViewer()
mandelbrot.show()