"""A helper module whose body imports what it calls, like the ray example's `ray.py`.

`scale_by_its_own_global` reaches `pygf`'s `dot` through a name that only exists in *this*
module's globals. A kernel importing this module from elsewhere must still be able to have the
helper translated, so the helper's names have to resolve here first.
"""
from pygf import dot, float3

SCALE = 2.0


def scale_by_its_own_global(value: float) -> float:
    """`dot` comes from this module's import of pygf, not the caller's."""
    direction = float3(1.0, 0.0, 0.0)
    return dot(direction, direction) * value * SCALE
