# The process-wide settings object, published as ``pygpgpu.config``.
#
# Every global knob is one attribute of one object. The state itself still lives where its
# subject lives -- the device selector list in ``kernel/decorator/devices.py``, the width
# mapping in ``kernel/decorator/type_map.py``, the logging default in
# ``utils/print_info.py`` -- and this module is the single validating front door to it.
# Reading a setting imports nothing: the modules are reached from inside each property, so
# ``import pygpgpu`` stays free of every backend and of the translator.
from __future__ import annotations
import threading
from collections.abc import Mapping as _Mapping
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)


# The python types ``type_map`` is keyed by, and the internal key each one names.
_TYPE_MAP_KEYS: Dict[type, str] = {int: "int", float: "float"}

# The backends a ``kernel_import`` build option may name, in the hook's own probe order.
_KERNEL_IMPORT_BACKENDS = ("cuda", "hip", "metal", "opencl")


def _require_bool(name: str, value: object) -> bool:
    """A setting that is on or off takes a bool, and says so when it does not get one."""
    if isinstance(value, bool):
        return value
    raise TypeError("%s takes True or False, got %s"
                    % (name, type(value).__name__))


class KernelImportConfig:
    """The handwritten-kernel import hook's settings, at ``config.kernel_import``.

    Three attributes, each read when a ``.cl``/``.cu``/``.hip``/``.metal`` module is
    imported, so a change applies to the next import rather than to a module that is
    already in ``sys.modules``:

    ``type_checked``
        Check a call's argument types against the generated stub (a wrong dtype then
        fails at the call, not on the device). Default ``False``.

    ``generate_pyi``
        Write the ``.pyi`` stub beside the source. Default ``True``.

    ``options``
        Per-backend build options, as ``{backend: options}``. A backend that is absent
        (or ``None``) keeps its own default.

    Assigning ``options`` replaces the mapping as a whole; a backend the new mapping does
    not name goes back to its default, so ``{"cuda": build_options}`` says "only cuda is
    configured" rather than adding a key to what was there before.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._type_checked: bool = False
        self._generate_pyi: bool = True
        self._options: Dict[str, object] = {
            backend: None for backend in _KERNEL_IMPORT_BACKENDS}

    @property
    def type_checked(self) -> bool:
        """Whether a call's argument types are checked against the generated stub."""
        with self._lock:
            return self._type_checked

    @type_checked.setter
    def type_checked(self, enabled: bool) -> None:
        enabled = _require_bool("config.kernel_import.type_checked", enabled)
        with self._lock:
            self._type_checked = enabled

    @property
    def generate_pyi(self) -> bool:
        """Whether importing a handwritten kernel writes its ``.pyi`` stub."""
        with self._lock:
            return self._generate_pyi

    @generate_pyi.setter
    def generate_pyi(self, enabled: bool) -> None:
        enabled = _require_bool("config.kernel_import.generate_pyi", enabled)
        with self._lock:
            self._generate_pyi = enabled

    @property
    def options(self) -> Dict[str, object]:
        """The per-backend build options, as a snapshot keyed by backend name."""
        with self._lock:
            return dict(self._options)

    @options.setter
    def options(self, mapping: Optional[Mapping[str, object]]) -> None:
        if mapping is None:
            normalized: Dict[str, object] = {
                backend: None for backend in _KERNEL_IMPORT_BACKENDS}
        else:
            if not isinstance(mapping, _Mapping):
                raise TypeError(
                    "config.kernel_import.options takes a mapping like "
                    "{\"cuda\": options} or None, got %s" % type(mapping).__name__)
            unknown = [key for key in mapping if key not in _KERNEL_IMPORT_BACKENDS]
            if unknown:
                raise ValueError(
                    "config.kernel_import.options is keyed by backend names (%s); got %s"
                    % (", ".join(_KERNEL_IMPORT_BACKENDS),
                       ", ".join(repr(key) for key in unknown)))
            normalized = {backend: mapping.get(backend)
                          for backend in _KERNEL_IMPORT_BACKENDS}
        with self._lock:
            self._options = normalized

    def __repr__(self) -> str:
        return ("KernelImportConfig(type_checked=%r, generate_pyi=%r, options=%r)"
                % (self.type_checked, self.generate_pyi, self.options))


class Config:
    """The process-wide settings, one attribute each.

    One instance is published as ``pygpgpu.config``, and it is the only way a setting is
    changed::

        import pygpgpu

        pygpgpu.config.print_info = False                  # quiet progress logging
        pygpgpu.config.type_map = {float: np.float64}      # a python float is a `double`
        pygpgpu.config.default_device = "cuda:0"           # what a plain k(...) runs on
        pygpgpu.config.default_device = ["cuda:0", "hip:0"]  # ... tried in this order
        pygpgpu.config.kernel_import.type_checked = True   # check imported kernels' calls

    ``print_info``
        Whether the backends log translation, compilation, copy and launch lines. The
        value is process-wide: assigning it flips the flag of every backend already
        loaded *and* records the default a backend loaded later starts from, so it does
        not matter whether the assignment or the backend import comes first. A single
        backend can still be silenced on its own with its own class attribute, e.g.
        ``pygpgpu.opencl.driver.CL.print_info = False``; that does not move this setting.

    ``type_map``
        The device width a python ``int`` / ``float`` travels as, keyed by the python
        type: ``{float: np.float64}``, ``{int: np.int64, float: np.float64}``. Values are
        the same type objects ``@kernel(int=..., float=...)`` takes -- a numpy dtype or
        scalar type, a ctypes scalar class, or a device type name. Reading it gives the
        resolved device type names, e.g. ``{float: "double"}``. Assigning replaces the
        mapping as a whole (``{}`` or ``None`` restores the 32-bit defaults), and a python
        ``complex`` follows the ``float`` entry. **It applies at decoration time:** a
        kernel already declared keeps the mapping it was declared with, and the mapping is
        part of the kernel's identity (cache key and C name), so set it before decorating.
        A decorator's own ``float=``/``int=`` wins over this.

    ``default_device``
        What a kernel call without a bracket runs on, as a selector string (``"cuda:0"``,
        ``"opencl:cpu"``, a bare device name) or a list of them tried in order, falling
        back to the next when one is unusable. Reading it gives the effective list -- the
        built-in probe order (``cuda -> hip -> metal -> opencl``) until one is set -- and
        assigning ``None`` restores that order. It is read at call time, so it applies to
        kernels that were already decorated; a kernel declared ``@kernel(backend=...)``
        keeps that backend and is unaffected.

    ``kernel_import``
        The import hook's settings (see :class:`KernelImportConfig`).

    The settings are process-wide, not per-thread and not per-kernel. Nothing here is
    cached on the Python side beyond the values above themselves: the device choice is
    resolved on each launch, the width mapping is snapshotted when a kernel is decorated,
    and the logging flag is read by the backend that owns it.
    """

    __slots__ = ("_kernel_import",)

    def __init__(self) -> None:
        self._kernel_import = KernelImportConfig()

    @property
    def kernel_import(self) -> KernelImportConfig:
        """The import hook's settings."""
        return self._kernel_import

    @property
    def print_info(self) -> bool:
        """Whether the backends log their progress."""
        from .utils.print_info import initial_print_info
        return initial_print_info()

    @print_info.setter
    def print_info(self, enabled: bool) -> None:
        from .utils.print_info import _set_print_info
        _set_print_info(_require_bool("config.print_info", enabled))

    @property
    def type_map(self) -> Dict[type, str]:
        """The python ``int``/``float`` width mapping, as device type names."""
        from .kernel.decorator.type_map import _type_map_default_snapshot
        snapshot: Dict[str, str] = _type_map_default_snapshot()
        return {python_type: snapshot[name]
                for name, python_type in (("int", int), ("float", float))
                if name in snapshot}

    @type_map.setter
    def type_map(self, mapping: Optional[Mapping[type, object]]) -> None:
        from .kernel.decorator.type_map import _install_type_map
        if mapping is None:
            _install_type_map(None)
            return
        if not isinstance(mapping, _Mapping):
            raise TypeError(
                "config.type_map takes a mapping like {float: np.float64} or None, "
                "got %s" % type(mapping).__name__)
        declared: Dict[str, object] = {}
        for python_type, value in mapping.items():
            name = _TYPE_MAP_KEYS.get(python_type) if isinstance(python_type, type) else None
            if name is None:
                raise ValueError(
                    "config.type_map is keyed by the python types int and float; "
                    "got %r (a numpy scalar already names its own width)" % (python_type,))
            declared[name] = value
        # The widths themselves are validated here: an entry the key cannot take raises
        # the same ValueError `@kernel(float=...)` does.
        _install_type_map(declared)

    @property
    def default_device(self) -> List[str]:
        """The selectors an unspecified kernel call walks, in order."""
        from .kernel.decorator.devices import _default_device_selectors
        return _default_device_selectors()

    @default_device.setter
    def default_device(self, value: Optional[Union[str, Sequence[str]]]) -> None:
        from .kernel.decorator.devices import _install_default_devices
        if value is None:
            _install_default_devices(None)
        elif isinstance(value, str):
            _install_default_devices([value])
        elif isinstance(value, (list, tuple)):
            _install_default_devices(list(value))
        else:
            raise TypeError(
                "config.default_device takes a selector string like 'cuda:0', a list "
                "of them, or None; got %s" % type(value).__name__)

    def __repr__(self) -> str:
        return ("Config(print_info=%r, type_map=%r, default_device=%r, kernel_import=%r)"
                % (self.print_info, self.type_map, self.default_device,
                   self._kernel_import))


# The one settings object. ``pygpgpu.config`` is this instance, and it is the only
# process-wide state a caller writes to.
config = Config()
