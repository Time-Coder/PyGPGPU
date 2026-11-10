"""A Mandelbrot-family viewer drawn with ``QPainter``, meant to be subclassed.

:class:`FractalViewer` owns the camera, the GPU launch, the window and the mouse. What a pixel
looks like is the subclass's business, and there are two places to say it:

* :meth:`FractalViewer.color_of_complex` -- the colour of a point of the plane, which is all a
  plain escape-time fractal needs;
* :meth:`FractalViewer.color_of_pixel` -- the colour of pixel ``(i, j)`` directly, for a viewer
  whose colour cannot be read off the point alone (the perturbation viewer, where one double is
  not enough to name the point a pixel shows).

Nothing here is abstract: a viewer implements whichever of the two fits, and the base class
raises ``NotImplementedError`` if a render reaches neither.

    class Mine(FractalViewer):
        def color_of_complex(self, c: complex) -> color3f:
            return color3f(1.0, 0.0, 0.0)

    Mine().show()

The camera is the one the matplotlib viewers used, so the numbers mean the same thing:

* ``view_center`` -- the complex number at the middle of the window;
* ``scale`` -- how far the view is zoomed in, and ``default_pixels_per_value`` turns it into
  ``value_per_pixel``;
* a pixel ``(i, j)`` shows ``left_top_complex + (j, -i) * value_per_pixel``.

The window is a plain ``QWidget`` that paints the frame as a ``QImage`` with ``QPainter``, the
same way ``digital_image_processing/image_viewer.py`` does. The image is regenerated at the
widget's own size, so the picture always fills the window exactly and there is nothing to pan
or scale at the widget level.

Mouse and wheel events reproduce the matplotlib viewers' behaviour:

* drag with the left button to pan -- the point under the pointer follows it;
* turn the wheel to zoom about the pointer;
* resizing the window renders a new frame at the new size.
"""

import sys
import weakref
from typing import List, Optional

import math

import numpy as np

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from pygpgpu import kernel, config
from pygf import color3f

config.print_info = False
config.type_map = {float: np.float64}


def _qimage_from_array(array: np.ndarray) -> QImage:
    """An ``(height, width, 3)`` uint8 array as a ``QImage`` that owns its pixels.

    Qt expects every scanline to start on a 32-bit boundary, so a row whose byte length is not
    a multiple of four is padded first -- an image 3, 5, ... pixels wide would otherwise read
    past the end of the buffer and show garbage.
    """
    height, width = int(array.shape[0]), int(array.shape[1])
    packed = np.ascontiguousarray(array)
    stride = (width * 3 + 3) & ~3
    if stride != width * 3:
        padded = np.zeros((height, stride), dtype=np.uint8)
        padded[:, :width * 3] = packed.reshape(height, width * 3)
        packed = padded
    # copy() detaches the pixels from the temporary buffer, so they outlive this call.
    return QImage(packed.tobytes(), width, height, stride, QImage.Format.Format_RGB888).copy()


class FractalViewer(QWidget):
    """Camera + GPU draw + window, with the colouring left to the subclass.

    Nothing is abstract: a subclass colours through :meth:`color_of_complex`, or overrides
    :meth:`color_of_pixel` when the colour cannot be read off the point alone.
    """

    # Pixels per unit at scale 1.0: 200 means the window spans 800/200 = 4 units at 800 px wide.
    default_pixels_per_value: int = 200
    # One wheel notch multiplies the scale by this, exactly as the matplotlib viewers did.
    zoom_step: float = 1.2

    def __new__(cls, *args: object, **kwargs: object) -> "FractalViewer":
        """Create the ``QApplication`` if the caller has none, before the widget is built.

        This has to happen here and not in ``__init__`` or ``show``: ``QWidget`` refuses to be
        constructed without an application ("Must construct a QApplication before a QWidget"),
        and ``MandelbrotViewer().show()`` builds the widget before ``show`` ever runs. An
        application the caller already made is left alone.
        """
        _ensure_application()
        return super().__new__(cls)

    def __init__(self, init_width: int = 800, init_height: int = 600, title: str = "fractal") -> None:
        super().__init__()
        self.image = np.zeros((init_height, init_width, 3), dtype=np.uint8)
        self.view_center: complex = 0j
        self.scale: float = 1.0
        self.is_left_pressed: bool = False
        self.left_pressed_pos = QPointF()
        self._title = title
        self._frame: Optional[QImage] = None
        # A resize arriving before the camera is set up (the resize below, on some platforms)
        # must not render: `update_params` has not run yet, so there is no `max_iter` to use.
        self._ready = False
        self.setWindowTitle(title)
        self.update_params()
        self._ready = True
        self.resize(init_width, init_height)

    # -- camera -----------------------------------------------------------

    def update_params(self) -> None:
        """Recompute what the camera implies: pixel size, top-left corner, iteration budget."""
        self.value_per_pixel = 1 / (self.scale * self.default_pixels_per_value)
        self.left_top_complex = complex(
            self.view_center.real - 0.5 * self.value_per_pixel * self.image.shape[1],
            self.view_center.imag + 0.5 * self.value_per_pixel * self.image.shape[0]
        )
        # A deeper view needs more iterations to resolve the rim, which is what the log buys.
        self.max_iter: int = int(100 * np.log10(1 + self.scale) + 500)

    def point_of_pixel(self, i: int, j: int) -> complex:
        """The complex number pixel ``(i, j)`` stands for: row 0 is the top of the picture.

        A viewer that works in other coordinates overrides this rather than the colouring: the
        perturbation viewer, for one, has every pixel stand for its offset from the reference
        point, which is the quantity its orbit is computed from.
        """
        return complex(
            self.left_top_complex.real + j * self.value_per_pixel,
            self.left_top_complex.imag - i * self.value_per_pixel
        )

    # -- colouring: what a subclass changes -------------------------------

    @staticmethod
    def jet_color(x: float) -> color3f:
        """The shared palette: a smooth white-on-blue jet over ``x`` in 0..1."""
        return color3f(
            math.exp(-60 * abs(x - 3 / 4) ** 3.2),
            math.exp(-60 * abs(x - 2 / 4) ** 3.2),
            math.exp(-60 * abs(x - 1 / 4) ** 3.2)
        )

    def escape_steps(self, c: complex) -> float:
        """How many steps the orbit of ``c`` survives, with a fractional last step.

        The default colouring below needs it, so a viewer that keeps that colouring provides it;
        the base cannot count the steps of a fractal it does not know.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement escape_steps, so the default colouring "
            f"has nothing to draw; implement escape_steps or override color_of_complex")

    def color_of_complex(self, c: complex) -> color3f:
        """The colour of the complex point ``c``: the shared palette over the escape count.

        A viewer with a palette of its own overrides this (and may then ignore
        :meth:`escape_steps` entirely); one that keeps this colouring only implements
        :meth:`escape_steps`.
        """
        return self.jet_color(self.escape_steps(c) / self.max_iter)

    def color_of_pixel(self, i: int, j: int) -> color3f:
        """The colour of pixel ``(i, j)``, which is ``color_of_complex`` of what it stands for.

        A viewer whose colour cannot be read off the point alone overrides this instead -- the
        perturbation viewer does, because at depth one double is not enough to name the point a
        pixel shows, so it colours from that pixel's offset against a reference orbit.
        """
        return self.color_of_complex(self.point_of_pixel(i, j))

    # -- the launch -------------------------------------------------------

    @kernel
    def draw(self):
        """One frame: every pixel of ``self.image`` gets the colour of its point ``c``."""
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            color = self.color_of_pixel(i, j)
            self.image[i, j, 0] = 255 * color.r
            self.image[i, j, 1] = 255 * color.g
            self.image[i, j, 2] = 255 * color.b

    # -- painting ---------------------------------------------------------

    def status_text(self) -> str:
        """The centre/scale readout, shown in the window title."""
        return f"center: {self.view_center:.4f}, scale: {self.scale:.2g}"

    def redraw(self) -> None:
        """Render a frame on the device and show it."""
        self.draw()
        self._frame = _qimage_from_array(self.image)
        self.setWindowTitle(f"{self._title}  {self.status_text()}")
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        # The frame is exactly the widget size, so it is blitted 1:1; the widget's own
        # background shows through anywhere the frame does not reach.
        painter = QPainter(self)
        if self._frame is not None:
            painter.drawImage(0, 0, self._frame)
        painter.end()

    # -- camera moves: overridden by a viewer that needs more than a double ----

    def pan_by_pixels(self, dx: float, dy: float) -> None:
        """Move the view by a pointer step of ``(dx, dy)`` pixels."""
        # The picture follows the pointer, so the plane moves the other way.
        self.view_center -= complex(dx, -dy) * self.value_per_pixel

    def zoom_at_pixel(self, x: float, y: float, steps: float) -> None:
        """Zoom by ``steps`` wheel notches, keeping the point under ``(x, y)`` in place."""
        # Remember the point under the pointer, zoom, then put the centre where that point
        # lands in the new frame.
        offset = complex(
            x - 0.5 * self.image.shape[1],
            0.5 * self.image.shape[0] - y
        )
        mouse_point = self.view_center + offset * self.value_per_pixel

        self.scale *= self.zoom_step ** steps
        self.update_params()
        self.view_center = mouse_point - offset * self.value_per_pixel

    # -- interaction ------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.is_left_pressed = True
        self.left_pressed_pos = event.position()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.is_left_pressed:
            super().mouseMoveEvent(event)
            return
        dx = event.position().x() - self.left_pressed_pos.x()
        dy = event.position().y() - self.left_pressed_pos.y()
        self.left_pressed_pos = event.position()
        self.pan_by_pixels(dx, dy)
        self.update_params()
        self.redraw()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_left_pressed = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            event.ignore()
            return
        self.zoom_at_pixel(event.position().x(), event.position().y(), steps)
        self.update_params()
        self.redraw()
        event.accept()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        size = event.size()
        if size.width() < 1 or size.height() < 1 or not self._ready:
            return
        # The frame is always the widget size, so a resize renders a new one.
        self.image = np.zeros((size.height(), size.width(), 3), dtype=np.uint8)
        self.update_params()
        self.redraw()

    # -- lifetimes --------------------------------------------------------

    def show(self) -> None:
        """Render, show the window, and drive it (the matplotlib ``plt.show()``).

        The application was created in :meth:`__new__`, so by now there is one. When this
        module created it, it has no loop of its own yet and this call runs it -- which is what
        keeps the window up until the user closes it. An application the caller supplied is left
        alone: its own event loop drives the window, and this returns at once.
        """
        _keep_alive(self)
        self.redraw()
        super().show()
        if _OWNS_APPLICATION and not _LOOP_RUNNING:
            application = QApplication.instance()
            if application is not None:
                _run_loop(application)


# The windows are kept alive here: Python would otherwise collect a window with no parent.
_WINDOWS: List[FractalViewer] = []
# Whether this module is the one that created the application.
_OWNS_APPLICATION = False
# Whether the loop below is running right now, so a window shown from inside it does not nest.
_LOOP_RUNNING = False


def _ensure_application() -> QApplication:
    """The application, created on first use; an existing one is returned untouched."""
    global _OWNS_APPLICATION
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)
        _OWNS_APPLICATION = True
    if not isinstance(application, QApplication):
        raise RuntimeError("a fractal viewer needs a QApplication, but this process already has "
                           f"a {type(application).__name__} and Qt allows only one application")
    return application


def _run_loop(application: QApplication) -> None:
    """Run the event loop until the user has closed the last window."""
    global _LOOP_RUNNING
    _LOOP_RUNNING = True
    try:
        application.exec()
    finally:
        _LOOP_RUNNING = False


def _keep_alive(viewer: FractalViewer) -> None:
    """Hold a reference to a shown window, and drop it once the window is destroyed."""
    _WINDOWS.append(viewer)
    # A weak reference: a signal holding the window itself would keep it alive for good.
    reference = weakref.ref(viewer)

    def forget(_: object = None) -> None:
        viewer_ref = reference()
        if viewer_ref is not None:
            while viewer_ref in _WINDOWS:
                _WINDOWS.remove(viewer_ref)

    viewer.destroyed.connect(forget)
