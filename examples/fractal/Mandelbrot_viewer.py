"""Mandelbrot viewer without perturbation theory, kept as the baseline.

Every pixel iterates its own `c = left_top + j*value_per_pixel` in float64, so the
picture stops resolving once one pixel is finer than a double's 1e-16, which at
200 pixels per unit is a scale of about 1e14. `Mandelbrot_viewer.py` is the same
viewer with a reference orbit and float64 deltas, which goes far deeper; this file
is what that one is compared against.
"""

from typing import Optional

import numpy as np
import math

from pygpgpu import kernel, set_print_info, set_type_map
from pygf import float2, color3f

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent, ResizeEvent
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from matplotlib.figure import Figure

set_print_info(False)
set_type_map(float=np.float64)

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
    def escape_steps(c: complex, max_iter: int) -> float:
        zn = 0j
        abs_zn = abs(zn)
        n: int = 0
        while abs_zn <= 2 and n < max_iter:
            zn = zn**2 + c
            abs_zn = abs(zn)
            n += 1

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
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            c = complex(
                self.left_top_complex.real + j*self.value_per_pixel,
                self.left_top_complex.imag - i*self.value_per_pixel
            )
            steps = self.escape_steps(c, self.max_iter)
            color = self.smooth_jet_colormap(steps/self.max_iter)
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


mandelbrot = MandelbrotViewer()
mandelbrot.show()
