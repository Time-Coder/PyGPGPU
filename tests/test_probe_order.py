"""The probe order is platform-dependent: an Apple machine tries Metal first.

The order decides which backend a bracket-less call lands on, and the platform-blind
sequence answered that question the same way everywhere: `cuda -> hip -> metal ->
opencl`. On macOS that answer was wrong by construction -- no CUDA or HIP runtime
exists there, and the OpenCL runtime is Apple's deprecated one, so a Mac walked three
absent runtimes and landed on deprecated OpenCL while its native backend was
available. Metal now leads on `darwin`; everywhere else the sequence is unchanged.

The order is one constant (`DEFAULT_BACKEND_ORDER`), and everything that reads it
reads the same constant: the decorator's bracket-less walk, `config.default_device`'s
reported default, `list_devices()`, and `kernel_import`'s bare-call resolution.
"""
from __future__ import annotations

import sys

from pygpgpu.kernel.decorator import devices
from pygpgpu.kernel_import.tables import _BACKEND_EXTENSIONS


def _expected_order() -> "tuple[str, ...]":
    if sys.platform == "darwin":
        return ("metal", "cuda", "hip", "opencl")
    return ("cuda", "hip", "metal", "opencl")


def test_the_probe_order_leads_with_metal_on_macos() -> None:
    """On `darwin` Metal is tried first; elsewhere the sequence is unchanged."""
    assert devices.DEFAULT_BACKEND_ORDER == _expected_order()


def test_both_orders_cover_the_same_four_backends() -> None:
    """The platform only reorders; it never drops or adds a backend.

    A machine that moves between platforms (a packaged app, a shared `config` file)
    must see the same selector vocabulary everywhere, or a selector spelled on one
    platform would be refused on another.
    """
    assert set(devices._DEFAULT_DEVICE_ORDER) == set(devices._MACOS_DEVICE_ORDER)
    assert len(devices._DEFAULT_DEVICE_ORDER) == len(devices._MACOS_DEVICE_ORDER) == 4


def test_metal_precedes_cuda_and_hip_only_on_macos() -> None:
    """The reorder is exactly one fact: Metal before CUDA and HIP on `darwin`.

    Pinned through the public constant, because that is what every consumer reads --
    a test against the private tables could pass while the public constant stayed
    platform-blind.
    """
    order = devices.DEFAULT_BACKEND_ORDER
    if sys.platform == "darwin":
        assert order.index("metal") < order.index("cuda")
        assert order.index("metal") < order.index("hip")
    else:
        assert order.index("cuda") < order.index("metal")
        assert order.index("hip") < order.index("metal")


def test_kernel_import_walks_the_same_order() -> None:
    """A handwritten module's bare call resolves through the same platform-dependent order.

    `kernel_import` used to spell its own backend table; a Mac would then have resolved
    a bare call through the platform-blind sequence even after the decorator's order
    learned about Metal. Deriving the table from the one constant is what keeps the two
    surfaces in step -- this test is what makes the derivation load-bearing.
    """
    assert [backend for backend, _extension in _BACKEND_EXTENSIONS] == \
        list(devices.DEFAULT_BACKEND_ORDER)
    assert dict(_BACKEND_EXTENSIONS) == {
        "cuda": ".cu", "hip": ".hip", "metal": ".metal", "opencl": ".cl"}
