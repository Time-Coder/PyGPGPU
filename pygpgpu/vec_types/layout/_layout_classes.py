# The host and device classes of the vector types: the registry the packing
# reads and the device-side class a vector is carried as.
from __future__ import annotations
from typing import Optional



_HOST_CLASSES: Optional[dict[str, type]] = None


def _host_classes()->dict[str, type]:
    """name -> the class the host uses for that vector type (`pygf`'s classes)."""
    global _HOST_CLASSES
    if _HOST_CLASSES is not None:
        return _HOST_CLASSES
    import pygf
    from ...kernel.translator import _vec_registry

    classes:dict[str, type] = {}
    for name in _vec_registry():
        # `getattr` with a runtime name returns `Any`; the value is tested at once.
        host:object = getattr(pygf, name, None)
        classes[name] = host if isinstance(host, type) else _vec_registry()[name]
    _HOST_CLASSES = classes
    return classes


def _device_class(name:str, backend:str)->type:
    """The class whose memory layout the device uses for this vector type."""
    if backend != "opencl":
        return _host_classes()[name]
    from ...kernel.translator import _vec_registry
    return _vec_registry()[name]
