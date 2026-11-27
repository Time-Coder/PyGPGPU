"""Random numbers on the device: the fixed header, the spellings it serves, and the seed.

A kernel body draws by writing the spelling it already knows -- ``random.random()``,
``random.uniform(0, 1)``, ``np.random.random()``, ``np.random.normal(0.0, 1.0)`` -- and
the translator turns that call into one of the functions ``include/random.h`` declares.
The body never names a work item or a seed: the launch supplies the seed
(:func:`launch_seed`), and the header derives one stream per work item from the seed and
the work item's own position. That is the contract ``taichi.random`` offers, and it is
what lets a kernel draw without the caller threading a ``tid`` and a ``seed`` through
every helper: ``random.random()`` in a model function means the same thing as in the
loop body.

The interpreter backend executes the body as it is written, so there a draw is python's
own ``random``/``numpy.random`` call. The distribution and the bounds are the same; the
sequence is python's rather than the device's. That follows from the interpreter having
no work items (a magic-loop body runs once, standing for the whole launch), not from a
switch this module could flip -- so a kernel whose exact numbers matter belongs on a
device backend.

Nothing here is imported by user code: the declarations live in the header, the
spellings are the standard library's own objects, and the translator is what connects
them (see ``translator/random_emit.py``).
"""
from __future__ import annotations

import os
import random as _stdlib_random
import threading
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


__all__ = [
    "RandomSpelling",
    "HEADER_NAME",
    "HEADER_SOURCE",
    "INCLUDE_FOLDER",
    "random_spelling",
    "is_random_module",
    "module_spelling",
    "supported_spellings",
    "header_source",
    "include_directory",
    "install_header",
    "include_line",
    "seed",
    "set_seed",
    "launch_seed",
    "phase_seed",
]

# The header is a real file beside this module, like the quaternion and complex ones: it
# can be read and edited as a source, and `install_header` copies it into the cache's
# include folder because a quoted `#include` resolves relative to the including file.
HEADER_NAME: str = "random.h"

HEADER_SOURCE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "include", HEADER_NAME)

# The folder a cache keeps its device headers in.
INCLUDE_FOLDER: str = "include"

# One draw: the operation the header implements, and the argument counts it accepts.
# `random.uniform` needs both bounds while `np.random.uniform()` may be called with none
# (numpy's signature defaults them to 0 and 1), which is why the counts are a tuple
# rather than a range.
RandomSpelling = Tuple[str, Tuple[int, ...]]

# The operations `include/random.h` implements. The translator maps them to the C
# function each one calls; the names are shared with the header's comments.
UNIFORM: str = "uniform"
RANGE: str = "range"
NORMAL: str = "normal"
INT: str = "int"
INT_INCLUSIVE: str = "int_inclusive"

# python's `random` module: a uniform draw, a ranged draw, two normal draw spellings, and
# the two integer draws. `randrange` is numpy's `randint` shape ([low, high)), while
# `randint` is python's own (both ends inclusive) -- the two spellings mean different
# things in the two libraries, and each keeps its own meaning here.
_STDLIB_SPELLINGS: Dict[str, RandomSpelling] = {
    "random": (UNIFORM, (0, )),
    "uniform": (RANGE, (2, )),
    "gauss": (NORMAL, (2, )),
    "normalvariate": (NORMAL, (2, )),
    "randint": (INT_INCLUSIVE, (2, )),
    "randrange": (INT, (1, 2)),
}

# numpy's legacy `np.random` namespace, same rule: `randint` there is [low, high), and
# `normal`/`uniform` may be called with no bounds at all.
_NUMPY_SPELLINGS: Dict[str, RandomSpelling] = {
    "random": (UNIFORM, (0, )),
    "random_sample": (UNIFORM, (0, )),
    "sample": (UNIFORM, (0, )),
    "rand": (UNIFORM, (0, )),
    "uniform": (RANGE, (0, 2)),
    "normal": (NORMAL, (0, 2)),
    "standard_normal": (NORMAL, (0, )),
    "randn": (NORMAL, (0, )),
    "randint": (INT, (1, 2)),
}

# The callable -> spelling table, built once on first use. It is keyed by the *object*
# `random.random` and `np.random.random` are, not by their spelling, so `import random
# as rnd`, `from random import random` and the package path all resolve to the same draw.
# A bound method hashes and compares by its function and instance, which makes the
# module's own attribute a stable key.
_TABLE: Optional[Dict[object, RandomSpelling]] = None
_TABLE_LOCK: threading.Lock = threading.Lock()


def _build_table() -> Dict[object, RandomSpelling]:
    table: Dict[object, RandomSpelling] = {}
    for name, spelling in _STDLIB_SPELLINGS.items():
        function: object = getattr(_stdlib_random, name, None)
        if function is not None:
            table[function] = spelling
    for name, spelling in _NUMPY_SPELLINGS.items():
        function = getattr(np.random, name, None)
        if function is not None:
            table[function] = spelling
    return table


def random_spelling(target: object) -> Optional[RandomSpelling]:
    """The draw `target` stands for, or None when it is not one of the random callables."""
    global _TABLE
    table: Optional[Dict[object, RandomSpelling]] = _TABLE
    if table is None:
        with _TABLE_LOCK:
            if _TABLE is None:
                _TABLE = _build_table()
            table = _TABLE
    try:
        return table.get(target)
    except TypeError:
        # An unhashable object cannot be a key: it is not one of the random callables.
        return None


def is_random_module(module: object) -> bool:
    """True for python's `random` module and numpy's `np.random` module."""
    return module is _stdlib_random or module is np.random


def module_spelling(module: object) -> str:
    """The spelling an error message names `module` by."""
    return "np.random" if module is np.random else "random"


def supported_spellings() -> Tuple[str, ...]:
    """The spellings a kernel body may draw with, for an error message."""
    names: list = [f"random.{name}" for name in sorted(_STDLIB_SPELLINGS)]
    names.extend(f"np.random.{name}" for name in sorted(_NUMPY_SPELLINGS))
    return tuple(names)


def header_source() -> str:
    """The header's text, read from the file beside this module.

    Read on demand rather than cached, for the reason the quaternion and complex headers
    are: the file is a normal source a user may edit while debugging, and a stale copy
    would make the cache silently disagree with it.
    """
    with open(HEADER_SOURCE, "r", encoding="utf-8") as file:
        return file.read()


def include_directory(source_directory: str) -> str:
    """The `include/` folder beside a generated source, where its headers live."""
    return os.path.join(source_directory, INCLUDE_FOLDER)


def _write_if_changed(path: str, text: str) -> None:
    existing: Optional[str] = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            existing = file.read()
    if existing != text:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)


def install_header(source_directory: str) -> str:
    """Put the header in the cache's `include/` folder and return its path."""
    target: str = include_directory(source_directory)
    os.makedirs(target, exist_ok=True)
    path: str = os.path.join(target, HEADER_NAME)
    _write_if_changed(path, header_source())
    return path


def include_line(widths: Iterable[str], backend: str, directory: str) -> str:
    """The lines a generated source needs to reach the header.

    The macros come first because the header reads them: the device qualifier a CUDA
    source has to carry on every function, the backend's 64-bit unsigned spelling
    (`ulong` in OpenCL C and MSL, `unsigned long long` in CUDA C, which has no `ulong`),
    and the double block only when a double draw was emitted (a device without fp64, and
    MSL, reject a `double` variable however unused).

    The caller passes the **cache directory**; this places the file, so do not
    pre-append `include/` at the call site or the folder nests twice.
    """
    install_header(directory)
    qualifier: str = "__device__ " if backend in ("cuda", "hip") else ""
    word: str = "ulong" if backend in ("opencl", "metal") else "unsigned long long"
    macros: str = "#define PYG_RNG_NEED_DOUBLE\n" if "double" in set(widths) else ""
    return (f"#define PYG_RNG_QUAL {qualifier}\n"
            f"#define PYG_RNG_WORD {word}\n"
            f"{macros}"
            f'#include "{INCLUDE_FOLDER}/{HEADER_NAME}"')


# The seed one launch uses. `config.random.seed` pins the base; the launch count then
# walks a golden-ratio sequence from it, so successive launches draw different numbers
# while the whole sequence stays reproducible from that one setting. With no base, the
# process draws its own seed once, which is the "unseeded generator" numpy's own
# `np.random.random()` behaves like.
_BASE_SEED: Optional[int] = None
_PROCESS_SEED: Optional[int] = None
_LAUNCHES: int = 0
_SEED_LOCK: threading.Lock = threading.Lock()

# The step successive launch seeds are spaced by: the golden ratio's 32-bit fraction,
# which spreads however many launches a process makes across the whole 32-bit space.
_SEED_STRIDE: int = 0x9E3779B9

_SEED_MAX: int = 0xFFFFFFFF


def seed() -> Optional[int]:
    """The configured base seed, or None when the process draws its own."""
    return _BASE_SEED


def set_seed(value: Optional[int]) -> None:
    """Pin the base seed, or None to draw a fresh one for this process.

    Assigning restarts the launch count, so a sequence of launches repeats exactly when
    the same seed is assigned again -- which is what makes a stochastic kernel testable.
    """
    global _BASE_SEED, _PROCESS_SEED, _LAUNCHES
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"a random seed is an integer, got {type(value).__name__}")
        if not 0 <= value <= _SEED_MAX:
            raise ValueError(
                f"a random seed is a 32-bit unsigned integer (0..{_SEED_MAX}), got {value}")
    with _SEED_LOCK:
        _BASE_SEED = value
        _PROCESS_SEED = None
        _LAUNCHES = 0


def launch_seed() -> int:
    """The seed one launch uses: the base seed walked by this process's launch count."""
    global _PROCESS_SEED, _LAUNCHES
    with _SEED_LOCK:
        base: int
        if _BASE_SEED is not None:
            base = _BASE_SEED
        else:
            if _PROCESS_SEED is None:
                _PROCESS_SEED = int.from_bytes(os.urandom(4), "little")
            base = _PROCESS_SEED
        value: int = (base + _LAUNCHES * _SEED_STRIDE) & _SEED_MAX
        _LAUNCHES += 1
    return value


def phase_seed(base: int, index: int) -> int:
    """The seed of the pass at `index` of a launch whose base seed is `base`.

    A body split into passes launches one kernel per pass, and each pass draws from its
    own stream: two passes sharing one stream would answer the same numbers, which is
    not what "draw twice" means. The pass index is mixed into the seed with the same
    mixer the header uses, so the streams are unrelated rather than merely adjacent --
    and the whole launch still replays from the one base seed.
    """
    mixed: int = ((base ^ ((index + 1) * 0x9E3779B9)) * 0x85EBCA6B) & 0xFFFFFFFF
    mixed ^= mixed >> 13
    return mixed & _SEED_MAX