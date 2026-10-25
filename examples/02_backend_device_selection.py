"""02_backend_device_selection.py -- listing and selecting devices/backends.

The `@kernel` brackets and these two helpers all accept the same
"backend:device" syntax:
- `available_devices()` lists every usable "backend:device" string (in default
  backend order, backends missing at runtime are skipped); the numbering
  matches the order pure-numeric selectors see;
- `get_device(sel)` returns the matching Device object (each backend's existing
  Device class; no new class is introduced);
- the device part inside brackets accepts three forms: a numeric index / a
  name|vendor|architecture substring / "name:ordinal" for disambiguation.

Run:
    python examples/02_backend_device_selection.py
"""
import numpy as np

from pygpgpu import kernel, available_devices, get_device, DEFAULT_BACKEND_ORDER

# This @kernel specifies neither backend nor device; it only demonstrates
# device resolution.
@kernel
def one_fill(out):
    for i in range(out.shape[0]):
        out[i] = 1


def main():
    print("Default backend probe order:", DEFAULT_BACKEND_ORDER)

    sel = available_devices()
    print("Available devices:", sel)

    if not sel:
        print("This machine has no GPU backend at all.")
        return

    # 1) get_device fetches each real Device object
    for s in sel:
        dev = get_device(s)
        print(f"  {s:10s} -> {dev.name}")

    # 2) The brackets recognize the same strings; run the kernel once on the
    #    first entry
    pick = sel[0]
    out = np.zeros(8, dtype=np.int32)
    one_fill[pick](out)
    print(f"Running kernel with bare '{pick}' OK, sum = {int(out.sum())}")

    # 3) Name-substring matching (case-insensitive) is often more readable
    #    than an index. Take a name from the available devices, then run the
    #    first device matching one characteristic word as a demo.
    prio = ("nvidia", "radeon", "amd", "intel", "gfx")
    for key in sel:
        dev = get_device(key)
        if any(token in dev.name.lower() for token in prio):
            out = np.zeros(8, dtype=np.int32)
            one_fill[key](out)
            print(f"Name-substring match picked '{key}' ({dev.name}), kernel sum = {int(out.sum())}")
            break


if __name__ == "__main__":
    main()