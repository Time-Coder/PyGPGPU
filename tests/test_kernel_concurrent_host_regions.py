"""One launch's host-region values are its own, even when the kernel object is shared.

`_phase1` runs the body's host regions and records the values it bound; `_phase2` reads them
back to build the launch arguments. The kernel object is shared by every call, so a launch
that records its values *on the kernel* hands the next launch whatever the last one wrote --
and a host-region value may depend on the call's own arguments (`base = len(label) * scale`),
which makes the two calls disagree.

The test fails deterministically rather than by luck: it widens the window between the
record and the read, so the two launches interleave every time. Without the fix (
`plan.host_region_snapshot` instead of `self.host_region_snapshot`) it fails; with it, it
passes. It is the concurrency contract for host regions, and it is the reason the value
travels on the plan.

The kernel is launched through a bracket rather than a `default_device=`, so it runs on
whatever backend the fixture picked and the comparison is about the *value*, not the device.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pygpgpu import kernel

ROUNDS = 12
THREADS = 4


@kernel
def host_region_kernel(out: np.ndarray, repeats: float):
    """A host-region value that depends on this call's argument."""
    label = "abcd"
    base = len(label) * 1.5 + repeats
    for i in range(out.shape[0]):
        out[i] = base


@pytest.fixture
def widened_phase_boundary(monkeypatch):
    """Sleep between phase 1 (records the values) and phase 2 (reads them back)."""
    from pygpgpu.kernel.decorator import _phase2

    original = _phase2._BuildLaunchMixin._build_launch

    def widened(self, plan, instance, args, backend_sel, device_sel, global_size,
                local_size, launch):
        time.sleep(0.001)
        return original(self, plan, instance, args, backend_sel, device_sel, global_size,
                        local_size, launch)

    monkeypatch.setattr(_phase2._BuildLaunchMixin, "_build_launch", widened)
    return widened


def test_a_host_region_value_belongs_to_its_own_launch(gpu_backend, widened_phase_boundary) -> None:
    failures: list[str] = []
    lock = threading.Lock()

    def worker(tag: int) -> None:
        for round_index in range(ROUNDS):
            repeats = float(tag * 100 + round_index)
            out = np.zeros(8, dtype=np.float32)
            try:
                host_region_kernel[gpu_backend](out, repeats)
            except BaseException as error:      # noqa: BLE001 - reported, not raised
                with lock:
                    failures.append(f"thread {tag} raised {type(error).__name__}: {error}")
                return
            expected = 4 * 1.5 + repeats
            if not np.allclose(out, np.full(8, expected, dtype=np.float32)):
                with lock:
                    failures.append(
                        f"thread {tag} round {round_index}: wanted {expected}, got {out[0]}")
                return

    threads = [threading.Thread(target=worker, args=(tag,))
               for tag in range(1, THREADS + 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures[:3]
