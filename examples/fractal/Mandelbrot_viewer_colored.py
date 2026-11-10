"""The Mandelbrot viewer with the two flat regions coloured.

`Mandelbrot_viewer.py` is deliberately minimal and pays for it: the escape
count is divided by `max_iter`, and at any ordinary zoom almost every pixel
escapes in a handful of steps, so the whole exterior lands on one dark tone,
while every pixel of the set lands on another one. Two large areas of the
picture are therefore flat colour -- the far field, where one or two steps is
all the escape there is, and the set itself, which has no escape count at all.

This file keeps that viewer's structure (one magic loop, one launch per redraw)
and only changes what a pixel's colour is made of:

* outside the set, the palette runs on a smooth escape value that repeats every
  `band` steps, so the bands keep their spacing however deep the zoom goes,
  and the phase also carries the logarithmic distance from the origin, which
  is what gives the far field its gradient -- out there the escape value is
  almost constant, and no function of it alone can be anything but one tone;
* inside the set there is no escape count, so the orbit itself does the
  shading: the mean orbit radius grows from the attracting centre out to the
  rim, and the angle of the last orbit point draws the fine filaments.
"""

from typing import Optional

import numpy as np
import math

from pygpgpu import kernel, config
from pygf import float2, color3f

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent, ResizeEvent
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from matplotlib.figure import Figure

config.print_info = False
config.type_map = {float: np.float64}

class MandelbrotViewer:

    def __init__(self, init_width:int=800, init_height:int=600):
        self.default_pixels_per_value = 200
        self.image = np.zeros((init_height, init_width, 3), dtype=np.uint8)
        self.view_center: complex = 0j
        self.scale: float = 1.0

        self.is_left_pressed: bool = False
        self.left_pressed_pos: float2 = float2()

        self.axes: Optional[Axes] = None
        self.figure: Optional[Figure] = None
        self.image_handle: Optional[AxesImage] = None
        self.text_handle = None

        self.update_params()

    def update_params(self):
        self.value_per_pixel = 1 / (self.scale * self.default_pixels_per_value)
        self.left_top_complex = complex(
            self.view_center.real - 0.5*self.value_per_pixel*self.image.shape[1],
            self.view_center.imag + 0.5*self.value_per_pixel*self.image.shape[0]
        )
        self.max_iter: int = int(100 * np.log10(1 + self.scale) + 500)

    @staticmethod
    def exterior_color(n: int, abs_zn: float, c: complex) -> color3f:
        # Smooth escape value: the step count with the last, partial step
        # resolved from how far past the escape radius the orbit went.
        t: float = float(n) + 1 - math.log2(math.log2(abs_zn))
        # The palette repeats every `band` steps, so the bands keep their
        # spacing at any zoom depth instead of collapsing into the end of a
        # once-per-render ramp. The logarithmic distance from the origin is
        # part of the phase for the far field, where `t` is nearly constant
        # and a colour read from it alone is one flat tone.
        band: float = 16.0
        x: float = t/band + 0.6*math.log2(abs(c))
        u: float = 6.2831853*(x - math.floor(x))
        # Faded by how far out the pixel escaped, so the neighbourhood of the
        # set carries the light and the picture has depth.
        fade: float = 0.5 + 0.5*(1 - math.exp(-t/6))
        return color3f(
            fade*(0.5 + 0.5*math.cos(u)),
            fade*(0.5 + 0.5*math.cos(u - 2.0944)),
            fade*(0.5 + 0.5*math.cos(u - 4.1888))
        )

    @staticmethod
    def interior_color(sum_r: float, zn: complex, max_iter: int) -> color3f:
        # Mean orbit radius: small at the attracting centre, larger at the
        # rim, so it is the height of the bulb rather than a uniform fill.
        mean_r: float = sum_r / float(max_iter)
        glow: float = min(1.0, 2.0*mean_r)
        # The last orbit point of a bounded orbit still circles the
        # attracting cycle, and its angle is what draws the filaments.
        ang: float = math.atan2(zn.imag, zn.real)
        fil: float = 0.5 + 0.5*math.sin(6.0*ang + 3.0*glow)
        # Dark violet, so the set reads as a silhouette against the warm
        # bands and its outline stays legible.
        return color3f(
            0.02 + 0.42*glow*fil,
            0.03 + 0.14*glow,
            0.10 + 0.55*glow
        )

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            c = complex(
                self.left_top_complex.real + j*self.value_per_pixel,
                self.left_top_complex.imag - i*self.value_per_pixel
            )
            zn = 0j
            abs_zn = abs(zn)
            n: int = 0
            sum_r: float = 0.0
            while abs_zn <= 2 and n < self.max_iter:
                zn = zn**2 + c
                abs_zn = abs(zn)
                n += 1
                sum_r += abs_zn
            color = color3f(0.0, 0.0, 0.0)
            if n < self.max_iter:
                color = self.exterior_color(n, abs_zn, c)
            else:
                color = self.interior_color(sum_r, zn, self.max_iter)
            self.image[i, j, 0] = 255 * color.r
            self.image[i, j, 1] = 255 * color.g
            self.image[i, j, 2] = 255 * color.b

    def show(self):
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
        self.text_handle = self.axes.text(5, 15, f"center: {self.view_center:.4f}, scale: {self.scale:.2g}", color='white')
        plt.show()

    def on_resized(self, event: ResizeEvent)->None:
        self.image = np.zeros((event.height, event.width, 3), dtype=np.uint8)
        self.update_params()

        self.draw["cuda"]()
        self.image_handle.set_data(self.image)
        self.text_handle.set_text(f"center: {self.view_center:.4f}, scale: {self.scale:.2g}")
        self.image_handle.set_extent([0, event.width, event.height, 0])

        self.axes.set_xlim(0, event.width)
        self.axes.set_ylim(event.height, 0)
        self.figure.canvas.draw_idle()

    def redraw(self) -> None:
        self.draw["cuda"]()
        self.image_handle.set_data(self.image)
        self.text_handle.set_text(f"center: {self.view_center:.4f}, scale: {self.scale:.2g}")
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
        self.view_center -= complex(dx, -dy) * self.value_per_pixel
        self.update_params()
        self.redraw()

    def on_released(self, event: MouseEvent) -> None:
        if event.button == MouseButton.LEFT:
            self.is_left_pressed = False

    def on_scrolled(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None or event.step == 0:
            return

        offset = complex(
            event.xdata - 0.5*self.image.shape[1],
            0.5*self.image.shape[0] - event.ydata
        )
        mouse_point = self.view_center + offset*self.value_per_pixel

        self.scale *= 1.2 ** event.step
        self.update_params()
        self.view_center = mouse_point - offset*self.value_per_pixel
        self.update_params()
        self.redraw()


if __name__ == "__main__":
    mandelbrot = MandelbrotViewer()
    mandelbrot.show()
