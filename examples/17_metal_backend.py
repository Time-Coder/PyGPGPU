"""Run the same kernel on Metal, and see what makes that backend different.

Metal is the fifth backend and the only one that needs macOS plus a real Apple GPU:
`MTLCreateSystemDefaultDevice` answers nil on every other platform *and* on a Mac
virtual machine, so this script explains how to select the backend and what to expect
rather than failing when Metal is absent.

Three things this script shows:

- how a kernel picks Metal (the `"metal"`/`"mtl"` bracket, or the probe order);
- that the *python source* is the same one the other backends run -- Metal reuses the
  OpenCL-shaped code generation, and only the address spaces and the two work-item
  facts are Metal's own;
- where the generated MSL lands (`__metalcache__/pykernels`), which is what you open
  when a launch needs debugging on a Mac.

Run:
    python examples/17_metal_backend.py
"""
import glob
import os

import numpy as np

from pygpgpu import kernel, list_devices


@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    """One work item per element -- the same body every backend runs."""
    for i in range(x.shape[0]):
        out[i] = alpha * x[i] + y[i]


def metal_sources() -> list:
    """The generated MSL files this run has written."""
    cache = saxpy._cache_dir("metal")
    return sorted(glob.glob(os.path.join(cache, "*.metal")))


def main() -> None:
    devices = list_devices()
    metal_devices = [name for name in devices if name.startswith("metal:")]

    print("Default probe order:", " -> ".join(("cuda", "hip", "metal", "opencl")))
    print("Devices on this machine:", devices or "(none)")
    print()

    if not metal_devices:
        # Off macOS the backend is unavailable, and that is the only thing this script
        # has to say. It is *not* an error: the package imports fine everywhere, the
        # driver records the failure in `MetaMetal.last_init_error` instead of raising,
        # and every other backend keeps working.
        print("This machine has no Metal device, so the Metal backend is unavailable.")
        print("That is expected off macOS: MTLCreateSystemDefaultDevice returns nil")
        print("there (and on a Mac virtual machine too). The package still imports and")
        print("parses .metal sources; only a launch needs the real device.")
        print()
        print("On a Mac this script would:")
        print("  1. run the kernel through the 'metal' bracket and check the result;")
        print("  2. print the generated MSL that the launch compiled;")
        print("  3. show that the body is the same one OpenCL and CUDA run.")
        return

    x = np.arange(16, dtype=np.float32)
    y = np.ones_like(x)
    out = np.zeros_like(x)

    # 1) Explicitly on Metal. The bracket also accepts "mtl", the shorter alias.
    saxpy["metal"](x, y, out, 2.0)
    expected = 2.0 * x + y
    assert np.allclose(out, expected), (out, expected)
    print(f"metal:{saxpy.backend} -> OK  (out.sum()={float(out.sum())})")

    if saxpy.backend != "metal":
        print("note: the kernel kept the backend it was specialized for")
        return

    # 2) The MSL this launch compiled. Metal has no `get_global_id()`: the generated
    #    prelude spells those six names as macros over attribute-decorated parameters,
    #    which is why the body above needed no change.
    sources = metal_sources()
    if sources:
        path = sources[-1]
        with open(path, encoding="utf-8") as source_file:
            text = source_file.read()
        print()
        print(f"generated MSL: {path} ({len(text)} chars)")
        print("the body the backend emitted:")
        in_body = False
        for line in text.splitlines():
            if line.startswith("kernel void"):
                in_body = True
            if in_body:
                print("   ", line)


if __name__ == "__main__":
    main()
