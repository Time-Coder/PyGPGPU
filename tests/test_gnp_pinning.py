"""The zero-copy pin table shares, merges and re-points registrations.

The driver refuses a host registration that overlaps one it holds
(`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`), and `cupMemHostUnregister` only drops a range
whose start is the registered one -- so a second array whose registered range overlaps an
existing one cannot be resolved at its own start. Arrays of one size allocated back to back
produce exactly that overlap (one straddles a page), which is how the launch used to fail.

Three rules follow, and each has a test here:

* a request a registration **covers** shares it, without asking the driver anything;
* a request that only **partly overlaps** becomes one registration over the union, and every
  binding already using it is re-pointed -- a re-registration can move the device address
  (measured on HIP), so a stale handle would read the wrong bytes;
* the registration is dropped with its **last** holder.

Registering page by page is deliberately not the answer: two separate registrations do not map
contiguously, so an array straddling the boundary would read through an unmapped device address.
"""
from __future__ import annotations

import pytest


def _fake_driver(monkeypatch) -> dict:
    """A driver that enforces the registration rules the real one does."""
    from pygpgpu.numpy._backends import _backends_pinning as pinning

    registrations: dict = {}
    refreshed: list = []

    def register(backend: str, start: int, size: int) -> None:
        for other_start, other_size in registrations.items():
            if start < other_start + other_size and other_start < start + size:
                raise RuntimeError("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED")
        registrations[start] = size

    def unregister(backend: str, start: int) -> None:
        # `...HostUnregister` refuses a range it did not register at that exact start, and
        # its caller swallows the refusal.
        registrations.pop(start, None)

    def device_pointer(backend: str, base: int) -> int:
        # A mapping that depends on where the registration begins, exactly the case a merge
        # changes -- and what a stale handle would then get wrong.
        return base * 2 + 0x1000

    def set_address(backend: str, buffer: object, address: int) -> None:
        buffer._id = address
        refreshed.append((buffer, address))

    monkeypatch.setattr(pinning, "_driver_register", register)
    monkeypatch.setattr(pinning, "_driver_unregister", unregister)
    monkeypatch.setattr(pinning, "_device_pointer", device_pointer)
    monkeypatch.setattr(pinning, "_set_buffer_address", set_address)
    monkeypatch.setattr(pinning, "_registered_ranges", {})
    monkeypatch.setattr(pinning, "_attachments", {})
    return {"registrations": registrations, "refreshed": refreshed}


class Holder:
    """A stand-in for a zero-copy buffer: weakref-able, with an address."""

    def __init__(self) -> None:
        self._id = 0


def test_a_range_a_registration_covers_shares_it(monkeypatch):
    """Two arrays on one page: one registration, two holders, no second driver call."""
    from pygpgpu.numpy._backends import _backends_pinning as pinning

    driver = _fake_driver(monkeypatch)
    page: int = pinning._PAGE_SIZE * 1000

    first = pinning._register_range("cuda", page, 2 * pinning._PAGE_SIZE)
    second = pinning._register_range("cuda", page + pinning._PAGE_SIZE, pinning._PAGE_SIZE)

    assert first == page and second == page
    assert driver["registrations"] == {page: 2 * pinning._PAGE_SIZE}
    assert driver["refreshed"] == []


def test_a_partial_overlap_merges_and_re_points_every_holder(monkeypatch):
    """A range that starts inside another one extends the registration to the union."""
    from pygpgpu.numpy._backends import _backends_pinning as pinning

    driver = _fake_driver(monkeypatch)
    page: int = pinning._PAGE_SIZE * 2000
    buffer_a = Holder()
    buffer_b = Holder()

    pinning._register_range("cuda", page, 2 * pinning._PAGE_SIZE)
    pinning._attach_holder("cuda", page, 2 * pinning._PAGE_SIZE, page, buffer_a)
    buffer_a._id = page * 2 + 0x1000

    # Starts on the second page of A's registration and runs past it: neither covers the
    # other, so the driver would refuse a second registration outright.
    base: int = pinning._register_range("cuda", page + pinning._PAGE_SIZE,
                                        2 * pinning._PAGE_SIZE)

    assert base == page
    assert driver["registrations"] == {page: 3 * pinning._PAGE_SIZE}
    # The union begins where A did, but its device mapping is a new one: A's handle moved.
    assert buffer_a._id == page * 2 + 0x1000
    assert (buffer_a, buffer_a._id) in driver["refreshed"]

    pinning._attach_holder("cuda", page + pinning._PAGE_SIZE, 2 * pinning._PAGE_SIZE,
                           page + pinning._PAGE_SIZE, buffer_b)
    assert buffer_b._id == 0     # the allocator computes the newcomer's own address


def test_the_registration_is_dropped_with_its_last_holder(monkeypatch):
    from pygpgpu.numpy._backends import _backends_pinning as pinning

    driver = _fake_driver(monkeypatch)
    page: int = pinning._PAGE_SIZE * 3000

    pinning._register_range("cuda", page, pinning._PAGE_SIZE)
    pinning._register_range("cuda", page, pinning._PAGE_SIZE)
    assert driver["registrations"] == {page: pinning._PAGE_SIZE}

    pinning._unregister_range("cuda", page, pinning._PAGE_SIZE)
    assert driver["registrations"], "the registration released its last holder too early"

    pinning._unregister_range("cuda", page, pinning._PAGE_SIZE)
    assert driver["registrations"] == {}
    assert pinning._registered_ranges == {}
    assert pinning._attachments == {}


def test_the_release_callable_does_not_hold_its_buffer(monkeypatch):
    """A binding must stay collectable: its finalizer may not close over the buffer.

    `weakref.finalize(buffer, ...)` keeps the callable it was given alive until it fires, so
    a release closure that captured the buffer made the buffer reachable from its own
    finalizer -- no zero-copy binding was ever collected, and every pinned page stayed
    registered for the life of the process.
    """
    import gc
    import weakref

    from pygpgpu.numpy._backends import _backends_alloc as alloc

    released: list = []
    buffer = Holder()
    alloc._replace_free_with_unregister(buffer, "cuda", 4096 * 5000, 4096,
                                        lambda *arguments: released.append(arguments))
    reference = weakref.ref(buffer)
    del buffer
    gc.collect()
    assert reference() is None, "the binding was kept alive by its own finalizer"
    assert released, "the release callable never ran"


def test_a_refused_merge_puts_the_old_registrations_back(monkeypatch):
    """The union could not be registered: the holders keep the ranges they had."""
    from pygpgpu.numpy._backends import _backends_pinning as pinning

    driver = _fake_driver(monkeypatch)
    page: int = pinning._PAGE_SIZE * 4000
    pinning._register_range("cuda", page, 2 * pinning._PAGE_SIZE)

    def refuse_the_union(backend: str, start: int, size: int) -> None:
        if start == page and size > 2 * pinning._PAGE_SIZE:
            raise RuntimeError("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED")
        driver["registrations"][start] = size

    monkeypatch.setattr(pinning, "_driver_register", refuse_the_union)

    with pytest.raises(RuntimeError):
        pinning._register_range("cuda", page + pinning._PAGE_SIZE, 2 * pinning._PAGE_SIZE)

    # The range that was there is registered again, so its holder is still valid.
    assert driver["registrations"] == {page: 2 * pinning._PAGE_SIZE}
    assert ("cuda", page) in pinning._registered_ranges
