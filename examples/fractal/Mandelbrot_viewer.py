"""Mandelbrot viewer without perturbation theory, kept as the baseline.

Every pixel iterates its own `c = left_top + j*value_per_pixel` in float64, so the
picture stops resolving once one pixel is finer than a double's 1e-16, which at
200 pixels per unit is a scale of about 1e14. `Mandelbrot_viewer_high_precision.py`
is the same viewer with a reference orbit and float64 deltas, which goes far
deeper; this file is what that one is compared against.

The coloring is a relief style instead of a flat palette: the first pass renders
the smooth escape count and an interior stripe field into buffers, and the second
pass lights them like surfaces -- the exterior becomes dark embossed terraces,
one smooth valley per escape step, and the interior becomes polished blue metal.
"""

from typing import Optional

import numpy as np
import math

from pygpgpu import kernel, normalize, set_print_info, set_type_map
from pygf import float2, float3

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
        # Light direction (x: left-to-right, y: top-to-bottom, z: toward the
        # viewer), pre-normalized. A vector member travels as one structure
        # member on both sides, so the kernel reads it per component.
        self.light_dir: float3 = normalize(float3(-0.35, -0.45, 0.82))

        self.update_params()

    def update_params(self):
        # The fields are rendered on an ss-times finer grid than the window
        # and the shaded result is box-downsampled into `image`: that is the
        # anti-aliasing, and it is what keeps the boundary speckle-free.
        self.ss: int = 2
        self.value_per_pixel = 1 / (self.scale * self.default_pixels_per_value)
        self.height: int = int(self.image.shape[0]) * self.ss
        self.width: int = int(self.image.shape[1]) * self.ss
        self.vpp_field: float = self.value_per_pixel / self.ss
        self.left_top_complex = complex(
            self.view_center.real - 0.5*self.vpp_field*self.width,
            self.view_center.imag + 0.5*self.vpp_field*self.height
        )
        self.max_iter: int = int(100 * np.log10(1 + self.scale) + 500)
        # Per-pixel fields filled by the first pass of `draw` and shaded by the
        # second: `field_s` already holds the *height* (exterior terraces,
        # interior domes) and `field_t` only marks the interior (1.0) against
        # the exterior (-1.0) so the shading picks the material.
        self.field_s = np.zeros((self.height, self.width), dtype=np.float64)
        self.field_t = np.zeros((self.height, self.width), dtype=np.float64)
        self.image_f = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    @staticmethod
    def valley_floor(s: float) -> float:
        # Depth of one escape-step terrace: a smooth valley bottoming at the
        # integer step count, so successive counts form matching terraces.
        k: float = math.floor(s + 0.5)
        return math.exp(-0.5 * abs((s - k) * 2.5)**2)

    @kernel
    def draw(self):
        # Pass 1: the per-pixel height field -- terraces outside, a dome inside.
        for i, j in np.ndindex(self.height, self.width):
            c = complex(
                self.left_top_complex.real + j*self.vpp_field,
                self.left_top_complex.imag - i*self.vpp_field
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
            if n < self.max_iter:
                # Exterior: the terrace height itself -- one smooth valley
                # per escape step, fading with the count into a calm plain.
                s: float = float(n) + 1 - np.log2(np.log2(abs_zn))
                self.field_s[i, j] = (0.30 + 0.70*self.valley_floor(s))*math.exp(-0.03*s)
                self.field_t[i, j] = -1.0
            else:
                # Interior: the dome -- mean orbit radius, a smooth
                # functional of a normal family, squared and scaled (0.01
                # deep inside to 0.5 at the rim, calibrated) and finally
                # raised to 1.5 so the attracting center sits in a flat
                # spoon instead of a pinched cone. `field_t` only marks
                # the interior; the dome carries all of the shading.
                mean_r: float = sum_r / float(self.max_iter)
                dome: float = min(1.0, 2.0*mean_r*mean_r)
                self.field_s[i, j] = dome*dome*math.sqrt(dome)
                self.field_t[i, j] = 1.0

        # Pass 2: shade the fields as a lit, embossed surface. Two magic loops in one
        # body are two passes: the kernel is launched once per loop, in order, and the
        # arrays between them stay on the device.
        for i, j in np.ndindex(self.height, self.width):
            t = self.field_t[i, j]
            i1 = i + 1 if i + 1 < self.height else i
            i0 = i - 1 if i > 0 else i
            j1 = j + 1 if j + 1 < self.width else j
            j0 = j - 1 if j > 0 else j
            # Central-difference normal of the height field, then the Blinn
            # terms in scalar components. The soft z flattens the slopes:
            # wide, calm highlights instead of hard chrome ripples. The
            # interior dome varies over hundreds of pixels, so its raw
            # per-pixel gradient is nearly flat -- it gets a boost to read
            # as an inflated cushion.
            dx = self.height_at(i, j1) - self.height_at(i, j0)
            dy = self.height_at(i1, j) - self.height_at(i0, j)
            boost: float = 1.0
            if t >= 0.0:
                boost = 22.0
            dx = boost*dx
            dy = boost*dy
            nz = 0.55
            inv_len = 1 / math.sqrt(dx*dx + dy*dy + nz*nz)
            nx: float = -2.0*dx*inv_len
            ny: float = 2.0*dy*inv_len
            shade = self.light_dir.x*nx + self.light_dir.y*ny + self.light_dir.z*nz*inv_len
            if shade < 0.0:
                shade = 0.0
            # Half vector: the view direction is the +z axis, so it is the
            # normalized (light + view) sum.
            hx: float = self.light_dir.x
            hy: float = self.light_dir.y
            hz: float = self.light_dir.z + 1.0
            h_inv = 1 / math.sqrt(hx*hx + hy*hy + hz*hz)
            h_dot = (nx*hx + ny*hy + nz*inv_len*hz) * h_inv
            if h_dot < 0.0:
                h_dot = 0.0
            # Exterior: dark charcoal terraces. The material constants are
            # declared before the branch and only assigned inside it: a name
            # first bound inside a block does not survive it.
            mr: float = 0.10
            mg: float = 0.11
            mb: float = 0.13
            spec: float = 3.0
            shininess: float = 32.0
            if t >= 0.0:
                # Interior: deep sapphire. The dome shape carries the light:
                # the rim turns away from the head light, so every bulb gets
                # a bright crest on its upper-left flank and falls into deep
                # blue around the far side, with one soft highlight.
                mr = 0.07
                mg = 0.30
                mb = 0.75
                spec = 3.5
                shininess = 16.0
            glow = spec * h_dot**shininess
            color_r = mr*(0.30 + 0.80*shade) + mr*glow
            color_g = mg*(0.30 + 0.80*shade) + mg*glow
            color_b = mb*(0.30 + 0.80*shade) + mb*glow
            self.image_f[i, j, 0] = 255 * min(1.0, color_r)
            self.image_f[i, j, 1] = 255 * min(1.0, color_g)
            self.image_f[i, j, 2] = 255 * min(1.0, color_b)

    def height_at(self, i: int, j: int) -> float:
        # Pass 1 already wrote the height: terrace relief outside, the dome
        # of each bulb inside.
        return self.field_s[i, j]

    def render(self) -> None:
        # One kernel, two passes: the body has two magic loops, so it is launched once
        # per loop and the fields stay on the device between them.
        self.draw["cuda"]()
        # Box-downsample the supersampled render into the window image.
        ss: int = self.ss
        h: int = self.image.shape[0]
        w: int = self.image.shape[1]
        grouped = self.image_f.reshape(h, ss, w, ss, 3).astype(np.float64)
        self.image[:, :, :] = grouped.mean(axis=(1, 3)).astype(np.uint8)

    def show(self):
        self.render()
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

        self.render()
        self.image_handle.set_data(self.image)
        self.text_handle.set_text(f"center: {self.view_center:.4f}, scale: {self.scale:.2g}")
        self.image_handle.set_extent([0, event.width, event.height, 0])

        self.axes.set_xlim(0, event.width)
        self.axes.set_ylim(event.height, 0)
        self.figure.canvas.draw_idle()

    def redraw(self) -> None:
        self.render()
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
