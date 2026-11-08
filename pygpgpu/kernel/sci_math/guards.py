# The two guards the vocabulary shares: what a scalar-only builtin accepts, and how
# a complex value is recognised.
from __future__ import annotations



def _scalar_only(name:str, value:object)->object:
    """Refuse a vector argument: CUDA spells these for scalars only."""
    if hasattr(value, "_fields_") and hasattr(value, "__len__"):
        raise TypeError(
            f"{name}() takes a scalar, not a vector: CUDA has no vector overload for it, "
            f"so apply it per component")
    return value



def _is_complex(value:object)->bool:
    """Whether a value is a complex scalar -- python's, or numpy's.

    A complex value reaches a host body as either shape: a kernel body's `z[i]` off a
    complex array is a numpy scalar, while a python complex argument is python's. Both
    are two reals, which is what the rounding below is defined on.
    """
    return isinstance(value, complex) or type(value).__name__ in ("complex64",
                                                                  "complex128")
