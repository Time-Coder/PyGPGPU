"""`@kernel(default_device=...)`: the kernel's own list for its bracket-less calls.

The decorator takes the same shapes and the same selector strings
``config.default_device`` takes. Three behaviours are pinned here. (1) A kernel that named
entries walks its own list and never consults the process-wide setting; one that did not
follows it. (2) The entries are resolved at call time, so a list this machine cannot
satisfy right now still decorates, and the launch -- not the decoration -- is what
refuses it. (3) A bracket on the call overrides both lists, so a "default" never traps a
kernel onto one device.

The selector *strings* themselves ("backend", "backend:device", a bare device name) are
pinned by ``test_config.py`` (the setter shares the parser) and by the bracket tests;
this file pins how the two configurations combine.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import config, kernel
from pygpgpu.kernel.decorator import devices


def _device_backends() -> "list[str]":
    """Device backends this machine has, in the documented probe order."""
    return [name for name in devices.DEFAULT_BACKEND_ORDER
            if devices._probe_backend(name) is not None]


DEVICE_BACKENDS = _device_backends()


@pytest.fixture(autouse=True)
def _restore_default_device():
    """Every test ends with the process-wide setting as it found it."""
    saved = config.default_device
    yield
    config.default_device = saved


def _add_kernel():
    """The plainest kernel: one magic loop over three float32 arrays."""

    @kernel
    def add(a, b, out):
        for i in range(out.shape[0]):
            out[i] = a[i] + b[i]

    return add


def _add_on(default_device):
    """The same kernel, decorated with its own default-device list."""

    @kernel(default_device=default_device)
    def add(a, b, out):
        for i in range(out.shape[0]):
            out[i] = a[i] + b[i]

    return add


def _run_add(k, n=8):
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 2
    out = np.zeros(n, dtype=np.float32)
    k(a, b, out)
    np.testing.assert_allclose(out, a + b)
    return k


# ---------------------------------------------------------------------------
# Parsing: the same shapes config.default_device takes
# ---------------------------------------------------------------------------


def test_a_single_selector_is_a_one_entry_list():
    @kernel(default_device="cuda:0")
    def add(a, b, out): ...

    assert add._default_devices == [("cuda", "0")]


def test_a_bare_backend_and_a_bare_device_name_parse():
    @kernel(default_device="cuda")
    def add(a, b, out): ...

    @kernel(default_device="gfx1036")
    def scale(a, out, k): ...

    assert add._default_devices == [("cuda", None)]
    assert scale._default_devices == [(None, "gfx1036")]


def test_a_list_keeps_its_order():
    @kernel(default_device=["opencl:cpu", "cuda"])
    def add(a, b, out): ...

    assert add._default_devices == [("opencl", "cpu"), ("cuda", None)]


def test_a_tuple_is_accepted_like_a_list():
    @kernel(default_device=("cuda", "hip:0"))
    def add(a, b, out): ...

    assert add._default_devices == [("cuda", None), ("hip", "0")]


def test_none_and_empty_leave_the_choice_to_the_global_setting():
    @kernel
    def add(a, b, out): ...

    @kernel(default_device=None)
    def scale(a, out, k): ...

    @kernel(default_device=[])
    def twice(a, out): ...

    assert add._default_devices is None
    assert scale._default_devices is None
    assert twice._default_devices is None


def test_a_non_selector_is_refused_at_decoration():
    with pytest.raises(TypeError) as caught:
        @kernel(default_device=3)
        def add(a, b, out): ...

    assert "@kernel(default_device=...)" in str(caught.value)


def test_a_non_string_entry_is_refused_at_decoration():
    with pytest.raises(TypeError) as caught:
        @kernel(default_device=["cuda:0", 3])
        def add(a, b, out): ...

    assert "@kernel(default_device=...)" in str(caught.value)


def test_an_empty_selector_is_refused_at_decoration():
    with pytest.raises(ValueError) as caught:
        @kernel(default_device="")
        def add(a, b, out): ...

    assert "empty" in str(caught.value)


# ---------------------------------------------------------------------------
# Which list a bracket-less call walks
# ---------------------------------------------------------------------------


def test_the_kernel_entries_beat_the_global_config(monkeypatch):
    """A kernel that named entries walks them; config.default_device is not consulted."""
    config.default_device = "cuda:0"
    add = _add_on(["cuda:not-a-real-device", "opencl:mock"])

    walked: "list[object]" = []

    def fake_entry_usable(backend, device):
        walked.append((backend, device))
        if backend == "opencl":
            return backend, device
        return None

    monkeypatch.setattr(devices, "_entry_usable", fake_entry_usable)
    assert devices._default_target(add._default_devices) == ("opencl", "mock")
    assert walked == [("cuda", "not-a-real-device"), ("opencl", "mock")]


def test_without_own_entries_the_global_config_is_walked(monkeypatch):
    monkeypatch.setattr(devices, "_entry_usable",
                        lambda backend, device: (backend, device))

    @kernel
    def add(a, b, out): ...

    config.default_device = ["cuda:0"]
    assert devices._default_target(add._default_devices) == ("cuda", "0")


def test_the_bare_backend_entry_keeps_the_automatic_device():
    """`@kernel(default_device="cuda")` names the backend only: the device stays open."""
    if not DEVICE_BACKENDS:
        pytest.skip("needs a device backend")

    @kernel(default_device=DEVICE_BACKENDS[0])
    def add(a, b, out): ...

    assert devices._default_target(add._default_devices) == (DEVICE_BACKENDS[0], None)


def test_a_kernel_without_a_list_is_unchanged(monkeypatch):
    """Bare `@kernel` still walks config.default_device, the way it always did."""
    monkeypatch.setattr(devices, "_entry_usable",
                        lambda backend, device: (backend, device))
    config.default_device = ["hip:0"]
    assert devices._default_target(None) == ("hip", "0")


def test_the_python_backend_is_a_usable_default_too():
    """`default_device="python"` names the interpreter backend, which needs no device."""
    config.default_device = "python"
    assert devices._default_target(None) == ("python", None)


def test_a_device_part_on_the_python_backend_is_skipped():
    """`"python:0"` names a device the interpreter backend does not have: skip the entry."""
    config.default_device = ["python:0", "python"]
    assert devices._default_target(None) == ("python", None)


# ---------------------------------------------------------------------------
# What a call reports and what an unusable list does
# ---------------------------------------------------------------------------


def test_the_list_is_parsed_at_decoration_not_at_call():
    """The entries are snapshots: a later global assignment cannot rewrite them."""
    config.default_device = "cuda:0"

    @kernel(default_device="hip:0")
    def add(a, b, out): ...

    config.default_device = "opencl:0"
    assert add._default_devices == [("hip", "0")]
    config.default_device = None


def test_an_unusable_global_list_names_the_setting():
    config.default_device = "definitely-not-a-device"

    add = _add_kernel()
    a = np.zeros(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(RuntimeError) as caught:
        add(a, b, out)
    assert "config.default_device" in str(caught.value)
    assert "@kernel(default_device=...)" not in str(caught.value)


def test_an_unusable_kernel_list_names_the_decorator():
    @kernel(default_device=["definitely-not-a-device"])
    def add(a, b, out): ...

    a = np.zeros(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(RuntimeError) as caught:
        add(a, b, out)
    assert "@kernel(default_device=...)" in str(caught.value)
    # The global list is not the one that refused, so the message must not name it as the
    # one to fix.
    assert "assign None to config.default_device" not in str(caught.value)


# ---------------------------------------------------------------------------
# End to end, on whatever devices this machine actually has
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DEVICE_BACKENDS, reason="needs a device backend")
def test_a_bracketless_call_lands_on_the_named_device():
    add = _add_on(DEVICE_BACKENDS[0])

    _run_add(add)
    assert add.backend == DEVICE_BACKENDS[0]


@pytest.mark.skipif(not DEVICE_BACKENDS, reason="needs a device backend")
def test_a_bare_device_name_selects_the_backend_that_owns_it():
    selector = devices.list_devices()[0]
    add = _add_on(selector)

    _run_add(add)
    assert add.backend == selector.split(":", 1)[0]


@pytest.mark.skipif(len(DEVICE_BACKENDS) < 2, reason="needs two device backends")
def test_the_list_falls_back_to_the_next_entry():
    add = _add_on(["definitely-not-a-device", DEVICE_BACKENDS[0]])

    _run_add(add)
    assert add.backend == DEVICE_BACKENDS[0]


@pytest.mark.skipif(not DEVICE_BACKENDS, reason="needs a device backend")
def test_the_kernel_list_overrides_an_unusable_global():
    add = _add_on(DEVICE_BACKENDS[0])

    config.default_device = "definitely-not-a-device"
    try:
        _run_add(add)
        assert add.backend == DEVICE_BACKENDS[0]
    finally:
        config.default_device = None


@pytest.mark.skipif(len(DEVICE_BACKENDS) < 2, reason="needs two device backends")
def test_a_bracket_still_overrides_the_decorator_default():
    bracket = DEVICE_BACKENDS[1]
    add = _add_on(DEVICE_BACKENDS[0])

    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float32) * 2
    out = np.zeros(8, dtype=np.float32)
    add[bracket](a, b, out)
    np.testing.assert_allclose(out, a + b)
    assert add.backend == bracket


def test_the_python_backend_is_a_normal_entry():
    """`default_device="python"` names the interpreter backend like any other."""
    add = _add_on("python")

    _run_add(add)
    assert add.backend == "python"


def test_a_kernel_with_a_default_still_accepts_a_python_bracket():
    add = _add_on("python")

    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float32) * 2
    out = np.zeros(8, dtype=np.float32)
    add["python"](a, b, out)
    np.testing.assert_allclose(out, a + b)
