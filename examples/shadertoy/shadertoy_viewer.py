"""A Shadertoy-style viewer: subclass, override ``mainImage``, and go.

:class:`ShadertoyViewer` owns the window, the mouse and the per-frame launch; a subclass writes
one method -- :meth:`ShadertoyViewer.mainImage` -- which answers the colour of one pixel. The
method runs on the device: it is called once per pixel of the frame, the way the real Shadertoy
runs the GLSL of the same name.

The uniforms a Shadertoy shader reads are ``self`` attributes, so a body spells them the way it
would spell any attribute of its viewer. The vectors are pygf vectors, not loose floats:

* ``mainImage(self, frag_coord)`` -- the hook itself; ``frag_coord`` is the pixel's centre
  ``(x + 0.5, y + 0.5)`` in pixels, with the origin at the bottom-left and y up, exactly the
  GLSL ``fragCoord`` convention;
* ``self.i_resolution`` -- ``float2``: the frame size, ``(width, height)``;
* ``self.i_mouse`` -- ``float4``: xy is the pointer position in pixels, z the click position,
  w the button state (1.0 held, 0.0 up) -- Shadertoy's own packing;
* ``self.i_time`` -- seconds since the viewer was constructed, advancing frame by frame
  while the window is shown (the animation tick); a viewer that wants stills stops
  ``self._timer``;
* ``self.i_frame`` -- frames rendered so far.

    class MyShader(ShadertoyViewer):
        def mainImage(self, frag_coord: float2) -> color3f:
            uv = frag_coord / self.i_resolution
            return color3f(uv.x, uv.y, 0.5)

    MyShader().show()

The names follow Shadertoy's GLSL uniforms where one exists (``iResolution`` spelled with an
underscore, to read as one name here); ``frag_coord`` is the launch's own per-pixel input.
Everything is a single-precision float (GLSL's own width), an int or a pygf vector, so the
interpreter backend and every
device backend run the same body.

The uniforms are rebound by the host before each launch and read by the device during it. A
body must not write them: a uniform is not the shader's to change. Assigning a whole new
vector is the viewer's job -- and the one spelling device code cannot use anyway (a component
store like ``self.i_mouse.x = ...`` is refused; see
``tests/known_failures/vector_component_store.py``).
"""

import sys
import time
import weakref

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (QHideEvent, QImage, QMouseEvent, QPainter, QPaintEvent, QResizeEvent,
                           QShowEvent, QWheelEvent)
from PySide6.QtWidgets import QApplication, QWidget

from pygpgpu import config, kernel
from pygf import color3f, float2, float4

config.print_info = False
# GLSL is float32 and so is Shadertoy; the deep-zoom fractal viewers want float64, but a
# raymarcher under double precision costs a consumer GPU its fp32-to-fp64 rate (1:64 on the
# development machine's GeForce) and the animation crawls. The shaders here live in a small
# coordinate range, so single precision is both the faithful and the fast choice.
config.type_map = {float: np.float32}


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


# The windows are kept alive here: Python would otherwise collect a window with no parent.
_WINDOWS: list["ShadertoyViewer"] = []
# Whether this module created the application (then ``show`` runs the loop), and whether the
# loop is running right now (so a window shown from inside it does not nest).
_OWNS_APPLICATION: bool = False
_LOOP_RUNNING: bool = False


class ShadertoyViewer(QWidget):
    """The window and the per-frame launch; a subclass writes :meth:`mainImage`.

    The uniforms are written by the host before every launch and read by the device inside
    :meth:`mainImage`, so a shader body uses them as ordinary attributes of ``self``.
    """

    # Pixels per unit at scale 1.0 is the fractal viewer's idea; here the one number is
    # the animation cadence: ~60 fps, Shadertoy's own. A shader whose render costs more
    # than the interval simply runs slower -- the title's fps readout shows the honest
    # number, and a subclass that wants stills stops `self._timer`.
    frame_interval_ms: int = 16

    def __new__(cls, *args: object, **kwargs: object) -> "ShadertoyViewer":
        """Create the ``QApplication`` if the caller has none, before the widget is built.

        This has to happen here and not in ``show``: ``QWidget`` refuses to be constructed
        without an application, and ``MyShader().show()`` builds the widget first. An
        application the caller already made is left alone.
        """
        global _OWNS_APPLICATION
        if QApplication.instance() is None:
            QApplication(sys.argv)
            _OWNS_APPLICATION = True
        return super().__new__(cls)

    def __init__(self, width: int = 640, height: int = 360, title: str = "shadertoy") -> None:
        super().__init__()
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        # The uniforms, in Shadertoy's names and packings: i_mouse is the float4 Shadertoy
        # passes (xy the pointer, z the click position, w the button state), the rest are
        # scalars. They are plain attributes: a body reads them, the host rebinds them before
        # every launch, and nothing else manages them. The pixel position is not one of them --
        # it is `mainImage`'s own argument.
        self.i_resolution = float2(float(width), float(height))
        self.i_time = 0.0
        self.i_frame = 0
        self.i_mouse = float4(0.0, 0.0, 0.0, 0.0)
        self._started = time.perf_counter()
        self._frame: QImage | None = None
        # The seconds the last few redraws cost, oldest first; the readout takes the median.
        self._frame_times: list[float] = []
        # The animation: while the window is shown, a tick renders the next frame, which is
        # what makes i_time advance -- Shadertoy's shaders are animations by default, and a
        # viewer whose clock only moves when the pointer does would freeze them.
        self._timer = QTimer(self)
        self._timer.setInterval(self.frame_interval_ms)
        self._timer.timeout.connect(self.redraw)
        self._title = title
        self.setWindowTitle(title)

    # -- what a subclass writes -------------------------------------------

    def mainImage(self, frag_coord: float2) -> color3f:
        """The colour of the pixel whose centre is ``frag_coord``. **Override this.**

        The coordinates are pixels, not the unit square: ``uv = frag_coord /
        self.i_resolution`` is the shader's own first line. The default answers a grey
        gradient, so a viewer with no shader at all still shows something.
        """
        uv = frag_coord / self.i_resolution
        return color3f(uv.x, uv.y, 0.5 - uv.x * uv.y)

    # -- the launch -------------------------------------------------------

    @kernel
    def draw(self):
        """One frame: every pixel's centre goes to ``mainImage``, the colour to the buffer."""
        # frag_coord follows GLSL: origin at the bottom-left, y up. The buffer's row 0 is
        # the top, so the centre handed to the shader is measured from the bottom -- a
        # shader ported from Shadertoy needs no flip of its own. The y flip is computed
        # here rather than passed as a separate argument: a hook whose signature carries
        # an argument the base call does not pass loses the self-attribute scan (see
        # tests/known_failures/extra_arg_override.py), so mainImage keeps the GLSL
        # signature and receives the flipped coordinate inside its one float2.
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            frag_coord = float2(float(j) + 0.5,
                                float(self.image.shape[0]) - float(i) - 0.5)
            color = self.mainImage(frag_coord)
            self.image[i, j, 0] = 255 * color.r
            self.image[i, j, 1] = 255 * color.g
            self.image[i, j, 2] = 255 * color.b

    def redraw(self) -> None:
        """Advance the clock, launch one frame, and show it."""
        frame_start = time.perf_counter()
        self.i_time = frame_start - self._started
        self.i_frame += 1
        self.draw()
        self._frame = _qimage_from_array(self.image)
        # The readout is the render cost, not the event rate: redraws fire on show, resize
        # and pointer events, so an idle window renders nothing and an interval would read
        # as zero. What the number answers is how many frames a second this shader costs at
        # this size. The median over the last few frames, not a mean: the first launch of
        # a size compiles (about a second), and one outlier must not pin the readout for
        # the frames a running average needs to forget it.
        self._frame_times.append(time.perf_counter() - frame_start)
        if len(self._frame_times) > 5:
            del self._frame_times[0]
        middle = sorted(self._frame_times)[len(self._frame_times) // 2]
        fps = 1.0 / middle if middle > 0.0 else 0.0
        self.setWindowTitle(
            f"{self._title}  t={self.i_time:.1f}s  frame={self.i_frame}  {fps:.0f} fps")
        self.update()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        # The frame is exactly the widget size, so it is blitted 1:1.
        painter = QPainter(self)
        if self._frame is not None:
            painter.drawImage(0, 0, self._frame)
        painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        size = event.size()
        if size.width() < 1 or size.height() < 1:
            return
        self.image = np.zeros((size.height(), size.width(), 3), dtype=np.uint8)
        self.i_resolution = float2(float(size.width()), float(size.height()))
        self.redraw()

    def showEvent(self, event: QShowEvent) -> None:
        # The clock runs while the window does: starting on show (not in __init__) means an
        # invisible viewer renders nothing, and hideEvent is the matching stop.
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        # Closing hides the window first, so this one stop covers close, minimize and hide.
        self._timer.stop()
        super().hideEvent(event)

    # -- pointer and wheel ------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        position = event.position()
        # Shadertoy's packing: xy the pointer now, z where the click landed, w the button.
        self.i_mouse = float4(position.x(), position.y(), position.x(), 1.0)
        self.redraw()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not event.buttons() & Qt.MouseButton.LeftButton:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        # xy follows the drag; z stays where the click landed; w stays held.
        self.i_mouse = float4(position.x(), position.y(), self.i_mouse.z, 1.0)
        self.redraw()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Shadertoy freezes the whole uniform where the button came up.
            position = event.position()
            self.i_mouse = float4(position.x(), position.y(), position.x(), 0.0)
            self.redraw()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        # The unit square is the shader's whole world, so a wheel step has no camera meaning;
        # a shader that wants one reads the wheel through its own state. Accepted and ignored.
        event.accept()

    # -- lifetimes --------------------------------------------------------

    def show(self) -> None:
        """Render, show the window, and drive it (the matplotlib ``plt.show()``).

        The application was created in :meth:`__new__`, so by now there is one. When this
        module created it, this call runs its loop -- which keeps the window up until the user
        closes it. An application the caller supplied is left alone: its own loop drives the
        window, and this returns at once.
        """
        _keep_alive(self)
        self.redraw()
        super().show()
        application = QApplication.instance()
        global _LOOP_RUNNING
        if _OWNS_APPLICATION and not _LOOP_RUNNING and application is not None:
            _LOOP_RUNNING = True
            try:
                application.exec()
            finally:
                _LOOP_RUNNING = False


def _keep_alive(viewer: "ShadertoyViewer") -> None:
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
