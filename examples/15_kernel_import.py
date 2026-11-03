"""15_kernel_import.py -- one module name, three handwritten backends.

`kernel_import` merges same-named `.cl` / `.cu` / `.hip` files into a single importable
module, and publishes every kernel in it as an object used exactly like a `@kernel`
function: no bracket runs on the default device, `["cuda"]` / `["opencl"]` / `["hip"]` picks
a backend, and `["opencl:1"]` or `[(blocks, threads)]` picks a device or a work size the
same way.

`examples/kernel_import/` is an ordinary python package that holds the three sources:
`vecops.cl`, `vecops.cu` and `vecops.hip`. Importing `kernel_import.vecops` therefore imports
the merged module -- the same name, once, whatever backends the files cover.

What makes one module out of three files:

* a kernel declared in two or more files with the SAME parameter list is ONE function with
  several backend implementations -- the merged module declares one `__call__` for it and
  each bracket runs that backend's implementation;
* a kernel declared with DIFFERENT parameter lists is an overload -- the merged module
  declares one `__call__` per parameter list, each marked `@overload`;
* a kernel declared in only one of the files still exists: a bracket naming a backend with
  no implementation raises KeyError there, which is what the bracket is for.

Run:
    python examples/15_kernel_import.py
"""
import os
import sys

import numpy as np

from pygpgpu import list_devices
from pygpgpu import kernel_import as kimport    # the hook, aliased: `kernel_import` also names
                                                # the example package below, which is the point

# `examples/` holds the `kernel_import` package (`__init__.py`, `vecops.cl`, `vecops.cu`,
# `vecops.hip`); running this script directly already puts it on sys.path, and this makes the
# import work under any other runner too.
_examples_dir = os.path.dirname(os.path.abspath(__file__))
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

kimport.install()
from kernel_import import vecops      # the merged module: one name for three sources


# Which backends this machine can actually run, so the example adapts instead of failing
# on a missing runtime (importing a backend package initializes its driver).
present = {selector.split(":", 1)[0] for selector in list_devices()}


def run_on(selector, kernel_object, *args):
    if selector.split(":", 1)[0] not in present:
        print(f"{selector}: not available on this machine, skipped")
        return False
    kernel_object[selector](*args)
    return True


def main():
    print("Available devices:", list_devices())

    n = 1_000_000
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.empty_like(a)

    # 1) One parameter list, three implementations. No bracket = default device, work size
    #    derived from the argument shapes exactly as `@kernel` does it.
    vecops.add_offset(a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    print("add_offset on the default device OK")

    # 2) The same kernel with one backend named. The call has the kernel's own signature,
    #    so `add_offset["cuda"](a, b, out, offset, n)` completes and type-checks exactly
    #    like a decorated function.
    for selector in ("opencl", "cuda", "hip"):
        out.fill(0.0)
        if run_on(selector, vecops.add_offset, a, b, out, np.float32(1.0), np.int32(n)):
            assert np.allclose(out, a + b + 1.0)
            print(f"add_offset on {selector} OK")

    # 3) A bracket that names a backend without an implementation raises KeyError: `negate`
    #    is declared only by vecops.cl.
    for selector in ("cuda", "hip"):
        if selector not in present:
            continue
        try:
            vecops.negate[selector](out, np.int32(n))
        except KeyError as error:
            print(f"negate on {selector} raised KeyError as expected: {error}")

    # 4) Device and work size, in either order, exactly the decorator's bracket language.
    out.fill(0.0)
    vecops.add_offset[(3907, 256)](a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    for selector in list_devices():
        out.fill(0.0)
        vecops.add_offset[selector][(3907, 256)](a, b, out, np.float32(1.0), np.int32(n))
        assert np.allclose(out, a + b + 1.0)
        print(f"add_offset on {selector} with an explicit work size OK")

    # 5) Different parameter lists = overloads. `shift` is float in .cl/.hip and double in
    #    .cu, so the stub declares two __call__ overloads; which one a bracket runs is
    #    decided by the backend it resolves to, and the argument dtype has to agree.
    #    `shift` adds its factor to each element, so the buffers start at zero.
    out.fill(0.0)
    if run_on("opencl", vecops.shift, out, np.float32(2.0), np.int32(n)):
        assert np.allclose(out, 2.0)
    doubles = np.zeros(n, dtype=np.float64)
    if run_on("cuda", vecops.shift, doubles, 2.0, np.int32(n)):
        assert np.allclose(doubles, 2.0)
    # The device and the work size stack in either order, and the backend part decides which
    # of the two overloads runs.
    out.fill(0.0)
    if "hip" in present:
        vecops.shift["hip"][(3907, 256)](out, np.float32(2.0), np.int32(n))
        assert np.allclose(out, 2.0)

    # 6) Struct arguments and struct arrays cross the same merged type, whichever backend runs.
    Pt = vecops.Pt                     # a ctypes Structure built from the parsed declarations
    points = np.array([Pt(x=3.0, y=1.0), Pt(x=0.5, y=0.5)], dtype=Pt)
    results = np.zeros(2, dtype=np.int32)
    vecops.in_circle(points, results, Pt(x=0.5, y=0.5), np.float32(1.0), np.int32(2))
    assert results.tolist() == [0, 1], results
    print("in_circle (struct argument) OK:", results.tolist())


if __name__ == "__main__":
    main()
