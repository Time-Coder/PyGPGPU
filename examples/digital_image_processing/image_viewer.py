"""A minimal PySide6 image viewer drawn with ``QPainter``.

The viewer is a plain ``QWidget`` that paints one ``QImage``. There is no scroll area and no
scroll bar anywhere, so dragging never fights a scroll bar: the image itself is moved and
scaled inside the window.

    import numpy as np
    from image_viewer import imshow

    imshow(np.zeros((480, 640, 3), np.uint8), "black")

Interaction:

* drag with the left or the middle mouse button to move the image;
* turn the wheel to zoom about the mouse pointer, anywhere from MIN_SCALE to MAX_SCALE;
* double-click to fit the image to the window and centre it -- scaled up as well as down, so it
  fills the window on its limiting axis.

The area around the image is not painted, so the widget's own background shows through, and the
pointer keeps the default arrow: only the image moves, the window itself does not react.

Initial geometry: the window opens at the image size, in the middle of the screen. An image
larger than the screen gets a proportionally smaller window, just small enough for the window
frame (title bar included) to fit the screen's available area; the image then fills that window
exactly, so its aspect ratio is preserved. Pass ``window_size=(width, height)`` in pixels to ask
for another initial size instead, and the image is fitted into it.

Resizing the window: the image centre keeps its position relative to the window, because the
view stores that position as a fraction of the window size rather than in pixels -- a centred
image stays centred, and a panned image moves proportionally.

The Qt application: :func:`imshow` works whether or not a ``QApplication`` already exists, and
it never blocks. When it has to create the application, a script can call ``imshow`` as often as
it likes -- every call opens another window right away, all of them stay on screen, and the
event loop then runs at interpreter exit (through ``atexit``) until the user has closed the
last window. :func:`wait` is there for a script that wants to block at that point instead. If
the caller already owns an application, the window is shown and returned immediately and the
caller's own event loop drives it.

    imshow(first, "first")
    imshow(second, "second")   # both windows are up and waiting
"""

import atexit
import sys
import weakref
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (QCloseEvent, QGuiApplication, QImage, QMouseEvent, QPainter, QPaintEvent,
                           QResizeEvent, QShowEvent, QWheelEvent)
from PySide6.QtWidgets import QApplication, QWidget

# How far each further window is offset from the centred position, in pixels. Windows opened by
# successive imshow calls would otherwise sit exactly on top of each other, so only the last one
# would be reachable; this lays them out in a cascade instead.
CASCADE_STEP = 32

# Zoom limits, in widget pixels per image pixel, and the factor of one wheel notch. They bound
# the wheel only: unlike the fit view, the wheel may shrink the image well below its own size.
MIN_SCALE = 0.01
MAX_SCALE = 64.0
ZOOM_STEP = 1.25


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Map any numeric array onto the 0..255 range, without modifying the input."""
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.complexfloating):
        # A complex image has no direct 8-bit form; show the magnitude.
        return _to_uint8(np.abs(array))
    if array.dtype == np.uint16:
        # Keep the high byte, so the full 16-bit range maps onto 0..255.
        return (array >> 8).astype(np.uint8)
    values = array.astype(np.float64)
    if np.issubdtype(array.dtype, np.floating):
        finite = values[np.isfinite(values)]
        # 0..1 is the usual float image range; anything larger is read as 0..255.
        if finite.size == 0 or float(finite.max()) <= 1.0:
            values = values * 255.0
    # NaN and the infinities have no byte value, and casting them is undefined behaviour.
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _qimage_from_rows(packed: np.ndarray, width: int, height: int, channels: int) -> QImage:
    """Build a ``QImage`` from rows of 8-bit samples, padded to Qt's scanline alignment."""
    stride = (width * channels + 3) & ~3
    if stride != width * channels:
        padded = np.zeros((height, stride), dtype=np.uint8)
        padded[:, :width * channels] = packed.reshape(height, width * channels)
        packed = padded
    image_format = QImage.Format.Format_Grayscale8 if channels == 1 else (
        QImage.Format.Format_RGB888 if channels == 3 else QImage.Format.Format_RGBA8888)
    # copy() detaches the pixels from the temporary buffer, so they outlive this call.
    return QImage(packed.tobytes(), width, height, stride, image_format).copy()


def to_qimage(image: np.ndarray) -> QImage:
    """Convert a grayscale / RGB / RGBA numpy array into an 8-bit ``QImage``.

    The returned image owns its pixels (it is a copy), so the array -- and any temporary
    buffer made here -- may be freed as soon as this function returns.

    Supported layouts are ``(height, width)``, ``(height, width, 1)``, ``(height, width, 3)``
    and ``(height, width, 4)``. ``uint8`` is used as it is; ``bool`` and ``uint16`` are scaled
    onto 0..255; floating point arrays are read as 0..1 when their maximum is at most 1 and as
    0..255 otherwise; a complex array shows its magnitude; every other type is clipped to
    0..255. Rows may be of any length, because the buffer is padded to Qt's 32-bit alignment.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError("image must have shape (height, width), (height, width, 1), (height, width, 3) or "
                         f"(height, width, 4), got {array.shape}")
    channels = 1 if array.ndim == 2 else int(array.shape[2])
    packed = np.ascontiguousarray(_to_uint8(array))
    height, width = packed.shape[0], packed.shape[1]
    if height < 1 or width < 1:
        raise ValueError("cannot display an empty image")
    return _qimage_from_rows(packed, width, height, channels)


def _available_screen() -> QRect:
    """The primary screen's usable area, in Qt's logical pixels."""
    screen = QGuiApplication.primaryScreen()
    if screen is None:  # pragma: no cover - every real platform reports one
        return QRect(0, 0, 1280, 720)
    return screen.availableGeometry()


def _checked_window_size(window_size: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Validate ``imshow(..., window_size=...)``: a pair of positive pixel counts, or None."""
    if window_size is None:
        return None
    try:
        width, height = window_size
    except (TypeError, ValueError):
        raise ValueError(f"window_size must be a (width, height) pair of ints in pixels, got {window_size!r}")
    # bool is an int in python, but True/the pixel count 1 is never what a caller means here.
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"window_size {name} must be an int in pixels, got {value!r}")
        if value < 1:
            raise ValueError(f"window_size {name} must be at least 1 pixel, got {value}")
    return int(width), int(height)


class ImageViewer(QWidget):
    """One window showing one ``QImage``: drag to move, wheel to zoom, double-click to fit."""

    def __init__(self, image: QImage, title: str = "", parent: Optional[QWidget] = None,
                 window_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(parent)
        if image.isNull() or image.width() < 1 or image.height() < 1:
            raise ValueError("ImageViewer needs a non-empty image")
        self._image = image
        # The initial client size in pixels, or None for "the image's own size".
        self._requested_window_size = _checked_window_size(window_size)
        # Widget pixels per image pixel.
        self._scale = 1.0
        # The image centre as a fraction of the window size: (0.5, 0.5) is exactly the
        # middle. A resize cannot move a fraction, which is the "relative position
        # unchanged" rule, and it needs no resize handler of its own.
        self._center = QPointF(0.5, 0.5)
        self._drag_origin: Optional[QPointF] = None
        self._drag_center = QPointF(0.5, 0.5)
        # The client size asked for while opening, until the window manager has confirmed it.
        self._opening_size: Optional[QSize] = None
        # Which slot of the cascade this window takes: 0 is the centred first window.
        self._cascade_step = len(_live_viewers())
        self.setWindowTitle(title)
        self._open_at_image_size()

    # -- geometry ---------------------------------------------------------

    def _open_at_image_size(self) -> None:
        """Size, place and fit the window for the image the first time it is shown."""
        # Realize the native window first: only then does Qt know how much frame (title bar
        # and borders) surrounds the client area, and that frame has to fit the screen too.
        self.winId()
        self._apply_opening_geometry(_available_screen())
        self.fit_to_window()

    def _apply_opening_geometry(self, area: QRect) -> None:
        """Size the window and centre the whole window on ``area``.

        Without a requested window size the window takes the image's own size, scaled down if
        the screen cannot hold it, so the image fills the window exactly. With one, that size
        wins on both axes -- the image is simply fitted into it afterwards, and the size is
        capped to the room the screen has for a client area.
        """
        image_width = self._image.width()
        image_height = self._image.height()
        frame = self.frameGeometry()
        client = self.geometry()
        frame_width = max(0, frame.width() - client.width())
        frame_height = max(0, frame.height() - client.height())
        room_width = max(1, area.width() - frame_width)
        room_height = max(1, area.height() - frame_height)
        if self._requested_window_size is not None:
            client_width = min(room_width, self._requested_window_size[0])
            client_height = min(room_height, self._requested_window_size[1])
        else:
            # One factor for both axes, so the window keeps the image's aspect ratio, and never
            # above 1: an image smaller than the screen is shown at its own size.
            factor = min(1.0, room_width / image_width, room_height / image_height)
            client_width = max(1, min(room_width, int(round(image_width * factor))))
            client_height = max(1, min(room_height, int(round(image_height * factor))))
        self._opening_size = QSize(max(1, client_width), max(1, client_height))
        self.resize(self._opening_size)
        self._center_on_screen(area)

    def _center_on_screen(self, area: QRect) -> None:
        """Put the whole window -- frame included -- in the middle of ``area``.

        Further windows of the same session are shifted by one, two, ... cascade steps, so they
        are all visible and reachable instead of hiding behind the newest one. The shift counts
        the windows already open and wraps around the room the screen has, which keeps every
        window inside the available area.
        """
        frame = self.frameGeometry()
        # move() positions the window frame (title bar included), so this centres the whole
        # window, not just its client area.
        left = area.x() + (area.width() - frame.width()) // 2
        top = area.y() + (area.height() - frame.height()) // 2
        step = self._cascade_step
        # Keep the whole frame on screen: the offset never pushes it past a screen edge.
        max_x = max(0, (area.width() - frame.width()) // 2)
        max_y = max(0, (area.height() - frame.height()) // 2)
        left += min(step * CASCADE_STEP, max_x)
        top += min(step * CASCADE_STEP, max_y)
        self.move(left, top)

    def _image_rect(self) -> QRectF:
        """Where the image lies in widget coordinates."""
        width = self._image.width() * self._scale
        height = self._image.height() * self._scale
        center_x = self._center.x() * self.width()
        center_y = self._center.y() * self.height()
        return QRectF(center_x - width / 2.0, center_y - height / 2.0, width, height)

    def fit_to_window(self) -> None:
        """Fit the image to the window on both axes (aspect kept) and centre it.

        The image is scaled up as well as down, so it always covers the window on its limiting
        axis -- a small image is magnified until it fills the window.
        """
        if self.width() < 1 or self.height() < 1:
            return
        self._scale = self._fit_scale()
        self._center = QPointF(0.5, 0.5)
        self.update()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        # The first paint means the window manager has finished with the opening geometry, so
        # any later resize is the user's own and must leave the relative centre alone.
        self._opening_size = None
        # The area around the image is left unpainted, so the widget's own background shows
        # through -- whatever the platform style gives it.
        painter = QPainter(self)
        # Smooth while shrinking (the usual downscale blur), crisp while magnifying.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._scale < 1.0)
        painter.drawImage(self._image_rect(), self._image)
        painter.end()

    # -- interaction ------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_origin = event.position()
            self._drag_center = QPointF(self._center)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None:
            delta = event.position() - self._drag_origin
            # Dragging moves the image with the pointer: the centre follows the mouse.
            self._center = QPointF(self._drag_center.x() + delta.x() / max(1, self.width()),
                                   self._drag_center.y() + delta.y() / max(1, self.height()))
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None and event.button() in (Qt.MouseButton.LeftButton,
                                                                Qt.MouseButton.MiddleButton):
            self._drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            self.fit_to_window()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps == 0.0:
            event.ignore()
            return
        self._zoom_about(self._scale * ZOOM_STEP ** steps, event.position())
        event.accept()

    def _zoom_about(self, scale: float, anchor: QPointF) -> None:
        """Zoom to ``scale`` while keeping the image point under ``anchor`` in place."""
        # The wheel is bounded only by the two constants, so any image can be shrunk right down
        # to MIN_SCALE instead of stopping where the window happens to fit it.
        scale = min(MAX_SCALE, max(MIN_SCALE, scale))
        rect = self._image_rect()
        if scale == self._scale or rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        # The image fraction under the pointer, which must not move.
        u = (anchor.x() - rect.left()) / rect.width()
        v = (anchor.y() - rect.top()) / rect.height()
        self._scale = scale
        width = self._image.width() * scale
        height = self._image.height() * scale
        left = anchor.x() - u * width
        top = anchor.y() - v * height
        self._center = QPointF((left + width / 2.0) / max(1, self.width()),
                               (top + height / 2.0) / max(1, self.height()))
        self.update()

    def _fit_scale(self) -> float:
        """The scale at which the whole image fills the window on its limiting axis."""
        if self.width() < 1 or self.height() < 1:
            return MAX_SCALE
        return min(self.width() / self._image.width(), self.height() / self._image.height())

    # -- lifetimes --------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        # Some platforms only reveal the real frame once the window is mapped, so the opening
        # geometry is computed again here, where a frame is guaranteed to be known, as long as
        # the window is still being opened.
        if self._opening_size is not None:
            self._apply_opening_geometry(_available_screen())
            self.fit_to_window()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        # A window manager may refuse the client size asked for while mapping the window and
        # grant another one instead (Windows keeps a titled window about 120 px wide). The
        # granted size is then the truth: fit the image to it and centre the window again, so
        # the opening geometry still holds as closely as the window manager allows. Clearing
        # the request is what makes this a one-off: a later user resize only follows the
        # fractional centre, which is the "relative position unchanged" rule.
        if self._opening_size is not None and event.size() != self._opening_size:
            self._opening_size = None
            self._center_on_screen(_available_screen())
            self.fit_to_window()

    def closeEvent(self, event: QCloseEvent) -> None:
        _release(self)
        super().closeEvent(event)


# The windows are kept alive here: Python would otherwise collect a window with no parent.
_VIEWERS: List[ImageViewer] = []
# The application this module created itself, if any.
_OWNED_APPLICATION: Optional[QApplication] = None
# Whether the event loop of that application is running right now.
_LOOP_RUNNING = False


def _live_viewers() -> List[ImageViewer]:
    """The windows that still exist, dropping any whose C++ object is already gone."""
    live: List[ImageViewer] = []
    for viewer in list(_VIEWERS):
        try:
            viewer.isVisible()
        except RuntimeError:
            _release(viewer)
            continue
        live.append(viewer)
    return live


def _release(viewer: ImageViewer) -> None:
    """Forget a closed viewer, so it can be collected."""
    while any(viewer is kept for kept in _VIEWERS):
        _VIEWERS.remove(viewer)


def _forget(reference: "weakref.ReferenceType[ImageViewer]") -> None:
    """Drop the closed viewer a weak reference pointed at."""
    viewer = reference()
    if viewer is not None:
        _release(viewer)


def _ensure_application() -> QApplication:
    """Return the running application, creating one if the caller has none."""
    global _OWNED_APPLICATION
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)
        _OWNED_APPLICATION = application
        # The event loop of an application created here runs when the interpreter exits, so
        # that a script may call imshow any number of times without any of them blocking, and
        # the windows stay up until the user closes them.
        atexit.register(_wait_for_user)
    if not isinstance(application, QApplication):
        # A QCoreApplication is already the singleton and Qt allows only one, so no widget can
        # be shown in this process any more.
        raise RuntimeError("imshow needs a QApplication, but this process already has a "
                           f"{type(application).__name__} and Qt allows only one application object")
    return application


def _any_window_visible() -> bool:
    """Whether at least one window this module opened is still on screen."""
    return any(viewer.isVisible() for viewer in _live_viewers())


def _wait_for_user() -> None:
    """Run the event loop until the user has closed the last window.

    This is what ``atexit`` calls. Qt's default is for the application to quit once its last
    window closes, so the loop ends when the user closes the last window -- and a session that
    never opened a window ends at once, rather than waiting for nothing.
    """
    global _LOOP_RUNNING
    application = QApplication.instance()
    if application is None or application is not _OWNED_APPLICATION or _LOOP_RUNNING:
        return
    if not _any_window_visible():
        return
    _LOOP_RUNNING = True
    try:
        application.exec()
    finally:
        _LOOP_RUNNING = False


def wait() -> None:
    """Block until the user has closed every window this module opened.

    ``imshow`` never blocks, so a script that wants its windows to hold it does this explicitly
    -- it is exactly the wait that happens at interpreter exit. Calling it when no window is
    open, when the caller owns the event loop, or while it is already waiting returns at once.
    """
    _wait_for_user()


def imshow(image: np.ndarray, title: str = "", window_size: Optional[Tuple[int, int]] = None) -> ImageViewer:
    """Display ``image`` (a numpy array) in its own window and return the viewer.

    ``image`` may be grayscale ``(height, width)`` or ``(height, width, 1)``, RGB
    ``(height, width, 3)`` or RGBA ``(height, width, 4)``; see :func:`to_qimage` for how the
    dtypes are mapped onto 8 bits. ``title`` is the window title.

    ``window_size`` sets the initial window size as ``(width, height)`` in pixels, measured on
    the client area -- the part the image is drawn in -- and the image is then fitted into it.
    Without it the window opens at the image's own size (scaled down when the screen cannot hold
    it). Either way the size is capped to the room the screen has, and the window is centred.

    The call never blocks, so several images can be shown from one script: every call opens a
    window, they are all on screen at the same time, and they stay there until the user closes
    them. When this module has to create the ``QApplication``, the event loop runs at
    interpreter exit and keeps the process alive until the last window is closed; a script that
    wants to wait earlier calls :func:`wait`. If the caller already owns an application, its own
    event loop drives the windows.
    """
    _ensure_application()
    viewer = ImageViewer(to_qimage(image), title, window_size=window_size)
    _VIEWERS.append(viewer)
    # A weak reference: a signal holding the window itself would keep it alive for good.
    reference = weakref.ref(viewer)
    viewer.destroyed.connect(lambda _=None: _forget(reference))
    viewer.show()
    return viewer
