"""The oop tests, split out of `test_metal_launch_path.py`.

The subject is oop.
"""
from __future__ import annotations



def test_the_oop_layer_names_its_classes_like_the_other_backends() -> None:
    """`Kernel` is the cross-backend name for a launchable entry point.

    Each backend publishes `Buffer`/`Context`/`Event`/`Program` under the same names,
    and Metal's kernel class is `MetalKernel` because that is what its own machinery
    is called. A caller that walks the backends generically -- comparing them, or
    writing a backend-specific tool -- reaches `oop.kernel.Kernel`, so the alias
    exists to keep that name answering on Metal too. The class itself keeps its
    name; this is only the shared spelling.
    """
    import pygpgpu.metal.oop.kernel as metal_kernel
    import pygpgpu.metal.oop.program as metal_program

    assert metal_kernel.Kernel is metal_kernel.MetalKernel
    assert metal_program.Kernel is metal_kernel.MetalKernel
    # The shared classes really are shared names, not Metal-specific ones.
    for module, name in ((metal_kernel, "BoundKernel"),
                         (__import__("pygpgpu.metal.oop.buffer", fromlist=["Buffer"]),
                          "Buffer"),
                         (__import__("pygpgpu.metal.oop.context", fromlist=["Context"]),
                          "Context"),
                         (__import__("pygpgpu.metal.oop.event", fromlist=["Event"]),
                          "Event")):
        assert hasattr(module, name), name


def test_a_metal_program_publishes_the_shared_program_surface() -> None:
    """The attributes shared code reads from a `Program` exist on Metal's.

    `kernel_import` and the decorator reach `kernels`, `kernel_infos` and `structs`
    on every backend's `Program`; a missing one would surface as an `AttributeError`
    from inside the import hook rather than as a Metal-specific message.
    """
    import pygpgpu.metal.oop.program as metal_program

    for name in ("kernels", "kernel_infos", "structs", "file_name"):
        assert hasattr(metal_program.Program, name), name
