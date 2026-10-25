"""07_low_level_oop.py -- the low-level OOP API (no @kernel magic loops).

Using the object-oriented OpenCL wrapper directly:
    Platforms -> Platform -> Device / default_context -> Context.compile -> Program.
    Program.kernel("name") / program.name -> Kernel; Kernel[(global, local)] sets the work size.
    Pass numpy arrays (or Buffer / structs) as arguments; the kernel handles the
    upload/download copies automatically.

There is no @kernel here, so the kernel source comes from a .cl file (reusing
vecadd.cl from this same directory). You could hand a source string to
ProgramParser to skip the file, but compiling .cl is the usual OOP case.

Run:
    python examples/07_low_level_oop.py
"""
import os
import numpy as np

from pygpgpu.opencl.oop.platforms import Platforms

_ex_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1) Enumerate platforms and devices
    print(f"{Platforms.n_platforms} OpenCL platform(s)")
    for p in Platforms:
        print(f"  platform: {p.name} (vendor={p.vendor})")
        for d in p.devices:
            print(f"    device: {d.name}")

    platform = Platforms[0]
    context = platform.default_context  # a context covering all devices of this platform

    # 2) Compile vecadd.cl (the first run creates the __clcache__ cache)
    program = context.compile(os.path.join(_ex_dir, "vecadd.cl"))
    kernel = program.add_offset       # program["add_offset"] also works
    print("kernel arguments:", kernel.args.keys())

    # 3) Allocate and run (work size automatic; kernel[(N, 256)] also works).
    # `n` is the real element count so the kernel can clamp: the launch size may be
    # rounded up to a multiple of the local size.
    n = 1_000_000
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.empty_like(a)
    kernel(a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    print("add_offset auto work size OK")

    out.fill(0.0)
    kernel[(1_000_192, 256)](a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    print("add_offset manual (global, local) OK")

    # 4) Multi-platform walk demo: you can also pick a device explicitly
    #    (it must belong to the context)
    if platform.n_devices > 1:
        other = platform.devices[1]
        out2 = np.empty_like(a)
        kernel[other.name](a, b, out2, np.float32(1.0), np.int32(n))
        assert np.allclose(out2, a + b + 1.0)
        print(f"Run on {other.name} OK")


if __name__ == "__main__":
    main()