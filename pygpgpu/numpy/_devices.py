"""Backend plumbing for the gnp array: selectors, devices, and command queues.

A gnp array keeps one **binding** per device it has been copied to. Identifying that
device is what this module does, along with opening the execution channel a copy runs
on. Nothing here knows about the array's state; it only answers "which device is this,
and what do I queue work on".

The backend packages are imported lazily -- ``import pygpgpu.numpy`` must stay free of
GPU runtimes, and ``pygpgpu`` keeps its backends behind the same rule.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

# Backend names, in the same spelling the kernel brackets use.
BACKENDS: Tuple[str, ...] = ("cuda", "hip", "opencl", "metal")


class Target(NamedTuple):
    """One device a copy can target."""

    backend: str
    device: Any
    name: str

    @property
    def key(self) -> Tuple[str, Any]:
        """The dict key a binding is stored under.

        The device object itself is the key, because every backend's Device compares --
        and hashes -- by the handle it wraps, so two resolutions of "cuda:0" are one key.
        Metal's Device has no equality of its own, which is why its device list is cached
        by the backend: the same selector answers the same object.
        """
        return (self.backend, self.device)

    def __str__(self) -> str:
        return f"{self.backend}:{self.name}"


def parse_selector(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a selector into ``(backend, device)``; unknown parts come back None.

    The spelling is the kernel bracket's own (``"cuda"``, ``"opencl:1"``, ``"hip:gfx1036"``),
    which is what makes ``to_device("cuda:0")`` read like ``kernel["cuda:0"]``.
    """
    from ..kernel.decorator import _parse_selector

    return _parse_selector(text)


def backend_devices(backend: str) -> list:
    """The backend's device list, in the numbering its selectors use."""
    from ..kernel.decorator import _backend_devices

    return list(_backend_devices(backend))


def probe(backend: str) -> bool:
    """Whether the backend is usable on this machine (it enumerates a device)."""
    from ..kernel.decorator import _probe_backend

    return _probe_backend(backend) is not None


def default_target() -> Target:
    """The device a copy targets when the caller names none.

    It is the kernel call's own default -- ``set_default_device()`` when configured,
    otherwise the ``cuda -> hip -> metal -> opencl`` probe -- so an array's plain
    ``to_device()`` lands where a plain ``kernel(...)`` call would run.
    """
    from ..kernel.decorator import _default_target

    backend, device_selector = _default_target()
    return resolve(backend, device_selector)


def resolve(backend: Optional[str], device_selector: Optional[Any] = None) -> Target:
    """Resolve a (backend, device) pair to a concrete device object.

    ``backend=None`` means "find the device on any backend, in the default order", which
    is how a selector that names only a device (``"gfx1036"``) is understood. A device
    that is already a Device object (the launch path passes one) is taken as-is: it is
    the answer, not a selector to look up.
    """
    from ..kernel.decorator import DEFAULT_BACKEND_ORDER, _default_backend, _resolve_device

    if backend is not None and device_selector is not None and \
            not isinstance(device_selector, str):
        return Target(backend, device_selector,
                      getattr(device_selector, "name", str(device_selector)))

    if backend is None:
        candidates = list(DEFAULT_BACKEND_ORDER) if device_selector else [_default_backend()]
        last_error: Optional[BaseException] = None
        for candidate in candidates:
            if not probe(candidate):
                continue
            try:
                device, name = _resolve_device(candidate, device_selector)
            except BaseException as error:
                last_error = error
                continue
            return Target(candidate, device, name)
        if last_error is not None:
            raise RuntimeError(
                f"no device matches {device_selector!r} on any available backend; "
                f"try list_devices()")
        raise RuntimeError(
            f"no device matches {device_selector!r}: no backend enumerates a device")

    device, name = _resolve_device(backend, device_selector)
    return Target(backend, device, name)


def target_of(selector: Any) -> Target:
    """Resolve anything the copy API accepts into a Target.

    Accepted: a selector string (``"cuda"``, ``"opencl:1"``, ``"gfx1036"``), ``None``
    (the default device), a ``(backend, device)`` pair, or a Device object from any of
    the backends.
    """
    if selector is None:
        return default_target()
    if isinstance(selector, Target):
        return selector
    if isinstance(selector, str):
        backend, device_selector = parse_selector(selector)
        return resolve(backend, device_selector)
    if isinstance(selector, tuple) and len(selector) == 2:
        backend, device_selector = selector
        return resolve(backend, device_selector)
    # A Device object: find which backend owns it, by identity against the enumerated list.
    for backend in BACKENDS:
        try:
            devices = backend_devices(backend)
        except BaseException:
            continue
        for device in devices:
            if device is selector or (hasattr(device, "__eq__") and device == selector):
                return Target(backend, device, getattr(device, "name", str(selector)))
    raise TypeError(
        f"cannot use {type(selector).__name__} as a device: pass a selector string like "
        f"'cuda:0', a (backend, device) pair, or a Device object")


def context_for(target: Target) -> Any:
    """The context a copy to ``target`` runs in."""
    if target.backend == "opencl":
        context = target.device.default_context
        if target.device not in context.devices:
            context = target.device.create_context()
        return context
    return target.device.default_context


def copy_queue(target: Target) -> Any:
    """Take an execution channel for one copy, from the asynchronous pool.

    Copies are asynchronous like an asynchronous launch, so they draw from the same
    non-blocking pool the ``submit``/``async_call`` launches use. The caller must return
    it with :func:`release_copy_queue` once the copy's event completes.
    """
    if target.backend == "metal":
        # Metal's launch path waits on its own command buffer and its buffers are shared
        # memory, so there is no pool to draw from and nothing to hand back.
        return None
    context = context_for(target)
    if target.backend == "cuda":
        context.make_current()
        return context.get_async_stream(blocking=True)
    if target.backend == "hip":
        context.make_current()
        return context.get_async_stream(blocking=True)
    if target.backend == "opencl":
        return context.get_async_queue(target.device, blocking=True)
    raise NotImplementedError(f"the {target.backend} backend has no copy channel")


def release_copy_queue(target: Target, queue: Any) -> None:
    """Return a channel taken by :func:`copy_queue` to its pool."""
    if queue is None:
        return
    if target.backend == "cuda":
        context_for(target).release_async_stream(queue)
    elif target.backend == "hip":
        context_for(target).release_async_stream(queue)
    elif target.backend == "opencl":
        context_for(target).release_async_queue(queue)


def describe(target: Target) -> str:
    """A copy's description, used in Transfer reprs and error messages."""
    return str(target)
