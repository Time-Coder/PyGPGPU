# The per-backend scalar type tables, and the signature they bind against.
from __future__ import annotations
from typing import Optional, Union
from typing import Dict, List, Tuple



# The extension each backend's sources carry, in the probe order `@kernel` uses.
_BACKEND_EXTENSIONS:Tuple[Tuple[str, str], ...] = (
    ("cuda", ".cu"),
    ("hip", ".hip"),
    ("metal", ".metal"),
    ("opencl", ".cl"),
)

# Where each backend keeps the parser that reads its sources. Both are stated once, so the
# four backends differ only by name.
_PARSER_MODULES:Dict[str, str] = {
    "cuda": ".cuda.oop.program_parser",
    "hip": ".hip.oop.program_parser",
    "metal": ".metal.oop.program_parser",
    "opencl": ".opencl.oop.program_parser",
}

# The qualifier words a declaration may carry; dropped before two backends' parameter lists
# are compared, so OpenCL's `__global const float*` and CUDA's `const float*` agree.
_QUALIFIER_WORDS:Tuple[str, ...] = ("const", "volatile", "restrict", "__global",
                                    "__constant", "__local", "__private", "__restrict")


def normalized_signature(args:List[Tuple[str, str]])->Tuple[Tuple[str, str], ...]:
    """A backend-independent parameter list: names + normalized pointer/type shapes.

    Address-space qualifiers are dropped (they are OpenCL spellings of "a pointer") and so
    are `const`/`volatile` -- they describe what a kernel does with the argument, and every
    backend accepts the same host values either way.
    """
    normalized:List[Tuple[str, str]] = []
    for name, type_str in args:
        words:str = " ".join(type_str.split())
        for qualifier in _QUALIFIER_WORDS:
            words = words.replace(qualifier, " ")
        pointer:bool = words.strip().endswith("*")
        base:str = words.strip()[:-1].strip() if pointer else words.strip()
        normalized.append((name, base + ("*" if pointer else "")))
    return tuple(normalized)


def _scalar_tables(backend:str)->Dict[str, Dict[str, type]]:
    """The scalar/vector ctypes tables one backend parses and checks types against."""
    if backend == "opencl":
        from ..opencl.driver import CLInfo
        return {"basic": CLInfo.basic_types, "vec": CLInfo.vec_types}
    if backend == "cuda":
        from ..cuda.driver import CUInfo
        return {"basic": CUInfo.basic_types, "vec": CUInfo.vec_types}
    if backend == "hip":
        from ..hip.driver import HIPInfo
        return {"basic": HIPInfo.basic_types, "vec": HIPInfo.vec_types}
    if backend == "metal":
        return _metal_scalar_tables()
    return {}


# MSL's vector widths are 2, 3 and 4 -- there is no `float8`/`float16`, unlike
# OpenCL -- so `_metal_scalar_tables` keeps only those names. The width filter is
# what makes this safe to derive from OpenCL's table: the *types* are the very ones
# the other backends use (a vector is a layout of its components, and does not
# depend on which device reads it); only the names and the widths are MSL's.
_METAL_TABLES: Dict[str, Dict[str, type]] = {}

# The vector suffixes MSL declares. `2`/`3`/`4` only.
_METAL_VECTOR_WIDTHS: Tuple[str, ...] = ("2", "3", "4")

# The scalar spellings MSL declares, without the `f`-suffixed CUDA ones. `half` is
# left out on purpose: it is a storage type, and `kernel_import`'s type check binds
# a host array to a parameter by matching numpy dtypes, which a `half` parameter in
# a handwritten source cannot express.
_METAL_SCALAR_NAMES: Tuple[str, ...] = (
    "bool", "char", "uchar", "short", "ushort", "int", "uint",
    "long", "ulong", "float", "double",
)


def _metal_scalar_tables()->Dict[str, Dict[str, type]]:
    """Metal's scalar and vector tables, derived from the OpenCL ones.

    Built lazily on first use: the tables reach `pygpgpu/opencl/driver`, and
    `kernel_import` is imported on machines that only ever parse sources.

    The derived tables are what makes a `.metal` module type-checked like any other
    instead of silently skipping the check (`_declaration_accepts` answers `None`
    when a name is absent, which the caller reads as "cannot tell").
    """
    if _METAL_TABLES:
        return _METAL_TABLES

    from ..opencl.driver import CLInfo

    basic: Dict[str, type] = {}
    for name in _METAL_SCALAR_NAMES:
        scalar = CLInfo.basic_types.get(name)
        if scalar is not None:
            basic[name] = scalar

    vec: Dict[str, type] = {}
    for name, vector in CLInfo.vec_types.items():
        # Keep only what MSL declares: a vector of a known scalar, 2/3/4 wide.
        for suffix in _METAL_VECTOR_WIDTHS:
            if not name.endswith(suffix):
                continue
            base: str = name[: -len(suffix)]
            if base in _METAL_SCALAR_NAMES:
                vec[name] = vector
                basic[name] = vector
            break

    _METAL_TABLES.update({"basic": basic, "vec": vec})
    return _METAL_TABLES


def _signature_text(signature:Tuple[Tuple[str, str], ...])->str:
    return "(" + ", ".join(f"{name}: {type_str}" for name, type_str in signature) + ")"


def _work_size_item(bracket:Union[int, str, Tuple[int, ...]])->Tuple[Optional[int], Optional[int]]:
    """The bracket as the backend `Kernel.__getitem__` wants it: a ``(blocks, threads)`` pair.

    A bare count (`k[n]`, `k[(n,)]`) is a total with an automatic group size, which every
    backend spells ``(count, None)``. Passing a one-entry tuple through unchanged would be
    read as a *device* selector by OpenCL's `Kernel`, and rejected by CUDA's.
    """
    if isinstance(bracket, tuple):
        if len(bracket) == 1:
            return (bracket[0], None)
        return (bracket[0], bracket[1])
    return (bracket, None)
