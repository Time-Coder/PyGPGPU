"""The platform tests, split out of `test_metal_launch_path.py`.

The subject is platform.
"""
from __future__ import annotations

import pytest


def test_selecting_metal_off_macos_explains_the_requirement() -> None:
    """`k["metal"]` on a machine without Metal names the requirement, not a path.

    The driver records its load failure in `MetaMetal.last_init_error` instead of
    raising, so the first real use is what surfaces it -- and for a device selector
    that is `Devices`, whose raw failure is a `FileNotFoundError` about a file under
    `/System/Library`. That text says nothing about the backend being unavailable or
    about macOS being what it needs, so the enumeration re-raises with both.
    """
    from pygpgpu.metal.driver.objc import ObjC
    from pygpgpu.metal.oop.devices import Devices

    if ObjC.lib is not None:
        pytest.skip("this machine has the Metal framework")

    with pytest.raises(RuntimeError) as refused:
        len(Devices)
    message = str(refused.value)
    assert "Metal backend is unavailable" in message
    assert "macOS" in message
    assert "real Apple GPU" in message


def test_probing_metal_answers_unavailable_instead_of_raising() -> None:
    """The probe is the contract every backend shares: unavailable, not an error.

    A machine with no Metal must still be able to `import pygpgpu.metal`, parse
    `.metal` sources and probe the backend, so the probe swallows the failure and
    answers None. The readable error above is for a *selection*, not a probe.
    """
    from pygpgpu.kernel.decorator import _probe_backend

    assert _probe_backend("metal") is None


def test_nsinteger_arguments_are_pointer_sized() -> None:
    """`NSUInteger` is pointer-sized, so `c_size_t` is the type to name.

    `setBuffer:offset:atIndex:` and `setBytes:length:atIndex:` take `NSUInteger`
    arguments, and `MTLSize`'s three fields are `NSUInteger` too. `c_ulong` happens to
    be 8 bytes on macOS, so those bindings worked there **by accident of the platform**
    -- it is 4 bytes on Windows. `c_size_t` is pointer-sized everywhere, which is what
    the selector's own type means.
    """
    import ctypes

    import pygpgpu.metal.oop.kernel as metal_kernel

    # MTLSize is three NSUIntegers, so 24 bytes on a 64-bit host.
    assert ctypes.sizeof(metal_kernel.MTLSize) == 3 * ctypes.sizeof(ctypes.c_size_t)
    size = metal_kernel.MTLSize(8, 4, 1)
    assert (size.width, size.height, size.depth) == (8, 4, 1)

    source = open(metal_kernel.__file__, encoding="utf-8").read()
    # The selectors must not name `c_ulong`: it is the wrong width off macOS.
    assert "(ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong)" not in source
    assert "(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)" in source


def test_the_kernel_docstring_matches_the_scalar_binding() -> None:
    """A scalar is bound with `setBytes:`, and the module says so.

    The docstring used to claim "Metal has no by-value kernel argument: a scalar is
    bound as a one-element buffer", which is the behaviour `setBytes:length:atIndex:`
    replaced -- and a docstring that describes the old design sends a reader looking
    for the wrong thing. The binding description and the code have to agree.
    """
    import pygpgpu.metal.oop.kernel as metal_kernel

    doc = metal_kernel.__doc__ or ""
    assert "setBytes:length:atIndex:" in doc
    assert "Every argument travels through a buffer" not in doc


def test_no_metal_module_names_a_platform_dependent_ctypes_type() -> None:
    """Every `NSUInteger` in this backend is spelled `c_size_t`, nowhere `c_ulong`.

    Objective-C's `NSUInteger` is pointer-sized. `ctypes.c_ulong` is that size on
    macOS *by coincidence* -- it is 4 bytes on Windows -- so a `c_ulong` in a
    selector's argument tuple or a struct field binds correctly only on the platform
    this code happens to be developed on. `c_size_t` is pointer-sized everywhere.

    This walks every module in the backend rather than a list of known call sites,
    because the defect is invisible in review: the code works on macOS either way.
    """
    import glob
    import os
    import re

    import pygpgpu.metal

    package = os.path.dirname(pygpgpu.metal.__file__)
    offenders = []
    for path in glob.glob(os.path.join(package, "**", "*.py"), recursive=True):
        source = open(path, encoding="utf-8").read()
        for match in re.finditer(r"ctypes\.c_ulong\b", source):
            line = source[:match.start()].count("\n") + 1
            offenders.append(f"{os.path.basename(path)}:{line}")
    assert not offenders, (
        "NSUInteger must be c_size_t, not c_ulong (pointer-sized only on macOS): "
        + ", ".join(offenders))


def test_the_objc_bridge_declares_pointer_sized_integers() -> None:
    """`NSUInteger` results and struct fields are `c_size_t`, never `c_ulong`.

    `objc.py` is the layer below the one the earlier sweep covered, and it had the
    same defect: `send_int` declared its result `c_ulong` while documenting it as
    `NSUInteger`, and `struct_result` built `MTLSize`/`MTLOrigin` fields from it.
    `registryID` is a full 64-bit value, so on a host where `c_ulong` is 4 bytes the
    id would be truncated -- and the device cache key is built from it.
    """
    import ctypes
    import inspect

    import pygpgpu.metal.driver.objc as objc

    import ast

    # Read the module's real code, not its prose: the name legitimately appears in
    # the docstrings that explain why it is the wrong type.
    tree = ast.parse(inspect.getsource(objc))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "c_ulong":
            offenders.append(f"line {node.lineno}: .c_ulong")
        if isinstance(node, ast.Name) and node.id == "c_ulong":
            offenders.append(f"line {node.lineno}: c_ulong")
    assert not offenders, offenders

    assert "c_size_t" in inspect.getsource(objc.ObjC.send_int)
    # Struct fields are pointer-sized too.
    assert "c_size_t" in inspect.getsource(objc.ObjC.struct_result)
    assert ctypes.sizeof(ctypes.c_size_t) == ctypes.sizeof(ctypes.c_void_p)
