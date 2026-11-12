"""A helper module whose device code does vector arithmetic of its own.

A kernel in another module calls `shade` from here, and its own body does the same
kind of arithmetic: both modules' generated files used to define the *synthesized*
device helpers (`vec_sub_float3_float3`, `vec_floor_mod_float3_float3` and friends)
-- one copy each -- and the kernel's unit, which includes this module's file, read
the same definition twice. The definitions now live in one shared header per cache
(`pygpgpu.kernel.source_paths.append_helpers`), so the inclusion is safe.
"""
from pygf import float3


def shade(base: float3, tint: float3, k: float) -> float3:
    """Every operation here synthesizes a device helper of its own."""
    mixed: float3 = base + (tint - base) * k
    # The `%` needs a floor wrapper per component, the same one the kernel's own
    # `%` below spells -- which is exactly the definition that used to collide.
    folded: float3 = mixed % float3(8.0, 8.0, 8.0)
    return folded * 0.5
