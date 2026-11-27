"""Random numbers on the device: the fixed header, the spellings it serves, and the seed.

A kernel body draws by writing the spelling it already knows -- ``random.random()``,
``random.expovariate(1.0)``, ``np.random.gamma(2.0)``, ``np.random.normal(0.0, 1.0)`` --
and the translator turns that call into one of the functions ``include/random.h`` and
its two companion headers declare. The body never names a work item or a seed: the
launch supplies the seed, and the header derives one stream per work item from the
seed and the work item's own position. That is the contract ``taichi.random`` offers,
and it is what lets a kernel draw without the caller threading a ``tid`` and a
``seed`` through every helper.

**Where the seed comes from.** There is no setting to assign. On the first launch
whose kernel draws, this module digests the state of the generator the body drew
from: :func:`random.getstate` for ``random.*``, :func:`numpy.random.get_state` for
``np.random.*``. Python does not publish a generator's seed itself, but its state is
exactly what a seed determines, so ``random.seed(1234)`` before the first launch
replays the device sequence just as it replays the host one. The digest is taken
once per source per process, so later host draws -- or a later ``seed()`` call --
do not move a device stream that has already started. Successive launches walk a
golden-ratio stride from that one base, which is what makes two launches different
without giving up reproducibility from the host generator's state.

The interpreter backend executes the body as it is written, so there a draw is
python's own call: the same distribution, but the generator the body named and
python's sequence. That follows from the interpreter having no work items (a
magic-loop body runs once, standing for the whole launch), not from a switch this
module could flip.

Nothing here is imported by user code: the declarations live in the headers, the
spellings are the standard libraries' own objects, and the translator is what
connects them (see ``translator/random_emit.py``).
"""
from __future__ import annotations

import hashlib
import os
import random as _stdlib_random
import threading
from typing import Dict, Iterable, NamedTuple, Optional, Tuple

import numpy as np


__all__ = [
    "RandomSpelling",
    "SOURCE_RANDOM",
    "SOURCE_NUMPY",
    "HEADER_NAME",
    "HEADER_SOURCE",
    "DISTRIBUTION_HEADER_NAMES",
    "INCLUDE_FOLDER",
    "random_spelling",
    "is_random_module",
    "module_spelling",
    "supported_spellings",
    "header_source",
    "include_directory",
    "install_header",
    "include_line",
    "launch_seed",
    "phase_seed",
]

# The headers are real files beside this module, like the quaternion and complex ones:
# they can be read and edited as sources, and `install_header` copies them into the
# cache's include folder because a quoted `#include` resolves relative to the including
# file. The state and the elementary draws are in `random.h`; the probability
# distributions live in the two companion headers it includes, split by the kind of
# value they answer rather than by line count.
HEADER_NAME: str = "random.h"
DISTRIBUTION_HEADER_NAMES: Tuple[str, ...] = (
    "random_continuous.h",
    "random_discrete.h",
)

_HEADER_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "include")

HEADER_SOURCE: str = os.path.join(_HEADER_DIR, HEADER_NAME)

# The folder a cache keeps its device headers in.
INCLUDE_FOLDER: str = "include"

# The two host generators a kernel body can name. A launch whose draws come from both
# digests both states, so neither library's stream is thrown away.
SOURCE_RANDOM: str = "random"
SOURCE_NUMPY: str = "np.random"

# The elementary operations: a uniform value, a bounded uniform, a normal, and the two
# integer spellings (python's inclusive `randint`, numpy's exclusive `randint`).
UNIFORM: str = "uniform"
RANGE: str = "range"
NORMAL: str = "normal"
INT: str = "int"
INT_INCLUSIVE: str = "int_inclusive"

# A real-valued distribution's result follows the kernel's float width; a discrete one
# is an integer whatever width its arguments have.
REAL: str = "real"
INTEGER: str = "int"


class RandomSpelling(NamedTuple):
    """The draw one callable names.

    ``operation`` is the header function's stem; ``counts`` are the argument counts
    the host call accepts; ``source`` is the generator whose state seeds the launch;
    ``kind`` says whether the result is a real or an integer; and ``defaults`` are the
    C expressions a call may omit, indexed by argument position.
    """

    operation: str
    counts: Tuple[int, ...]
    source: str
    kind: str
    defaults: Tuple[str, ...] = ()


def _stdlib(entry: str, counts: Tuple[int, ...], kind: str = REAL,
            defaults: Tuple[str, ...] = ()) -> RandomSpelling:
    return RandomSpelling(entry, counts, SOURCE_RANDOM, kind, defaults)


def _numpy(entry: str, counts: Tuple[int, ...], kind: str = REAL,
           defaults: Tuple[str, ...] = ()) -> RandomSpelling:
    return RandomSpelling(entry, counts, SOURCE_NUMPY, kind, defaults)


# python's `random` module. `randint` includes its high end and `randrange` does not,
# the same distinction numpy's `randint` makes on the exclusive side. `triangular`
# takes (low, high, mode), where numpy takes (left, mode, right), so the two spellings
# get different stems and the emitter orders the arguments for each.
_STDLIB_SPELLINGS: Dict[str, RandomSpelling] = {
    "random": _stdlib(UNIFORM, (0, )),
    "uniform": _stdlib(RANGE, (2, )),
    "gauss": _stdlib(NORMAL, (2, )),
    "normalvariate": _stdlib(NORMAL, (2, )),
    "randint": _stdlib(INT_INCLUSIVE, (2, ), INTEGER),
    "randrange": _stdlib(INT, (1, 2), INTEGER),
    "expovariate": _stdlib("expovariate", (1, )),
    "betavariate": _stdlib("beta", (2, )),
    "gammavariate": _stdlib("gamma", (2, )),
    "lognormvariate": _stdlib("lognormal", (2, )),
    "triangular": _stdlib("triangular", (3, )),
    "vonmisesvariate": _stdlib("vonmisesvariate", (2, )),
    "paretovariate": _stdlib("paretovariate", (1, )),
    "weibullvariate": _stdlib("weibull", (2, )),
}

# numpy's legacy `np.random` namespace. A draw answers one number, so the array-valued
# members (`bytes`, `choice`, `permutation`, `shuffle`, `multinomial`, `dirichlet`,
# `multivariate_normal`) are deliberately absent: the emitter refuses them with the
# "one number per call" reason rather than inventing a device array for them.
_NUMPY_SPELLINGS: Dict[str, RandomSpelling] = {
    "random": _numpy(UNIFORM, (0, )),
    "random_sample": _numpy(UNIFORM, (0, )),
    "sample": _numpy(UNIFORM, (0, )),
    "rand": _numpy(UNIFORM, (0, )),
    "uniform": _numpy(RANGE, (0, 2), defaults=("0", "1")),
    "normal": _numpy(NORMAL, (0, 2), defaults=("0", "1")),
    "standard_normal": _numpy(NORMAL, (0, )),
    "randn": _numpy(NORMAL, (0, )),
    "randint": _numpy(INT, (1, 2), INTEGER),
    "exponential": _numpy("exponential", (0, 1), defaults=("1", )),
    "standard_exponential": _numpy("standard_exponential", (0, )),
    "beta": _numpy("beta", (2, )),
    "gamma": _numpy("gamma", (1, 2), defaults=("", "1")),
    "standard_gamma": _numpy("standard_gamma", (1, )),
    "lognormal": _numpy("lognormal", (2, )),
    "triangular": _numpy("triangular_np", (1, 2, 3), defaults=("0", "1")),
    "chisquare": _numpy("chisquare", (1, )),
    "noncentral_chisquare": _numpy("noncentral_chisquare", (2, )),
    "f": _numpy("f", (2, )),
    "noncentral_f": _numpy("noncentral_f", (3, )),
    "standard_cauchy": _numpy("cauchy", (0, )),
    "standard_t": _numpy("standard_t", (1, )),
    "gumbel": _numpy("gumbel", (0, 1, 2), defaults=("0", "1")),
    "laplace": _numpy("laplace", (0, 1, 2), defaults=("0", "1")),
    "logistic": _numpy("logistic", (0, 1, 2), defaults=("0", "1")),
    "pareto": _numpy("pareto", (1, )),
    "power": _numpy("power", (1, )),
    "rayleigh": _numpy("rayleigh", (0, 1), defaults=("1", )),
    "vonmises": _numpy("vonmises", (2, )),
    "wald": _numpy("wald", (2, )),
    "weibull": _numpy("weibull_np", (1, )),
    "geometric": _numpy("geometric", (1, ), INTEGER),
    "poisson": _numpy("poisson", (1, ), INTEGER),
    "binomial": _numpy("binomial", (2, ), INTEGER),
    "negative_binomial": _numpy("negative_binomial", (2, ), INTEGER),
    "hypergeometric": _numpy("hypergeometric", (3, ), INTEGER),
    "logseries": _numpy("logseries", (1, ), INTEGER),
    "zipf": _numpy("zipf", (1, ), INTEGER),
}

# The callable -> spelling table, built once on first use. It is keyed by the *object*
# `random.random` and `np.random.random` are, not by their spelling, so `import random
# as rnd`, `from random import random` and the package path all resolve to the same draw.
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
    return SOURCE_NUMPY if module is np.random else SOURCE_RANDOM


def supported_spellings() -> Tuple[str, ...]:
    """The spellings a kernel body may draw with, for an error message."""
    names: list = [f"random.{name}" for name in sorted(_STDLIB_SPELLINGS)]
    names.extend(f"np.random.{name}" for name in sorted(_NUMPY_SPELLINGS))
    return tuple(names)


def header_source() -> str:
    """`random.h`'s text, read from the file beside this module.

    Read on demand rather than cached, for the reason the quaternion and complex
    headers are: the file is a normal source a user may edit while debugging, and a
    stale copy would make the cache silently disagree with it.
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
    """Put the headers in the cache's `include/` folder and return `random.h`'s path."""
    target: str = include_directory(source_directory)
    os.makedirs(target, exist_ok=True)
    for name in (HEADER_NAME,) + DISTRIBUTION_HEADER_NAMES:
        path: str = os.path.join(target, name)
        with open(os.path.join(_HEADER_DIR, name), "r", encoding="utf-8") as file:
            _write_if_changed(path, file.read())
    return os.path.join(target, HEADER_NAME)


def include_line(widths: Iterable[str], backend: str, directory: str) -> str:
    """The lines a generated source needs to reach the header.

    The macros come first because the header reads them: the device qualifier a CUDA
    source has to carry on every function, the backend's 64-bit unsigned spelling
    (`ulong` in OpenCL C and MSL, `unsigned long long` in CUDA C, which has no `ulong`),
    and the double block only when a double draw was emitted (a device without fp64, and
    MSL, reject a `double` variable however unused).

    The caller passes the **cache directory**; this places the files there, so do not
    pre-append `include/` at the call site or the folder nests twice.
    """
    install_header(directory)
    qualifier: str = "__device__ " if backend in ("cuda", "hip") else ""
    word: str = "ulong" if backend in ("opencl", "metal") else "unsigned long long"
    macros: str = "#define RANDOM_NEED_DOUBLE\n" if "double" in set(widths) else ""
    return (f"#define RANDOM_QUAL {qualifier}\n"
            f"#define RANDOM_WORD {word}\n"
            f"{macros}"
            f'#include "{INCLUDE_FOLDER}/{HEADER_NAME}"')


# The seed one launch uses. There is no setting: the state of each host generator is
# digested the first time a kernel draws from it, and cached for the life of the
# process. The launch count then walks a golden-ratio sequence from that base, so
# successive launches draw different numbers while the whole sequence is reproducible
# from the host generator's own seed.
_BASE_SEEDS: Dict[str, int] = {}
_LAUNCHES: int = 0
_SEED_LOCK: threading.Lock = threading.Lock()

# The step successive launch seeds are spaced by: the golden ratio's 32-bit fraction,
# which spreads however many launches a process makes across the whole 32-bit space.
_SEED_STRIDE: int = 0x9E3779B9
_SEED_MAX: int = 0xFFFFFFFF


def _digest(parts: Iterable[bytes]) -> int:
    """A 32-bit digest of a generator's state, stable across processes and platforms."""
    hasher = hashlib.blake2b(digest_size=4)
    for part in parts:
        hasher.update(part)
    return int.from_bytes(hasher.digest(), "little")


def _capture_seed(source: str) -> int:
    """Digest the current state of the generator a kernel body named."""
    if source == SOURCE_NUMPY:
        name, keys, position, has_gauss, cached_gauss = np.random.get_state()
        return _digest((
            str(name).encode("utf-8"),
            np.ascontiguousarray(keys, dtype=np.uint32).tobytes(),
            int(position).to_bytes(8, "little"),
            b"\x01" if has_gauss else b"\x00",
            np.float64(cached_gauss).tobytes(),
        ))
    if source == SOURCE_RANDOM:
        _version, internal, gauss = _stdlib_random.getstate()
        return _digest((
            np.asarray(internal, dtype=np.uint64).tobytes(),
            b"\x01" if gauss is not None else b"\x00",
            b"" if gauss is None else np.float64(gauss).tobytes(),
        ))
    # An unknown source cannot be named by a body; this keeps the function total.
    return int.from_bytes(os.urandom(4), "little")


def _mix(value: int) -> int:
    """Splitmix32's finalizer, for combining two sources' seeds."""
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & _SEED_MAX
    value = ((value ^ (value >> 15)) * 0x846CA68B) & _SEED_MAX
    return (value ^ (value >> 16)) & _SEED_MAX


def launch_seed(sources: Iterable[str]) -> int:
    """The seed one launch uses, digested from the generators the body draws with.

    Each source's state is read on its **first** use in this process and kept, so a
    later `random.seed()` cannot move a stream that has already begun. A body that
    draws from both libraries combines both digests. The launch count advances only
    when a kernel actually draws, so kernels that never draw do not shift the stream.
    """
    global _LAUNCHES
    names: Tuple[str, ...] = tuple(sorted(set(sources)))
    if not names:
        names = (SOURCE_RANDOM,)
    with _SEED_LOCK:
        base: int = 0
        for name in names:
            value: Optional[int] = _BASE_SEEDS.get(name)
            if value is None:
                value = _capture_seed(name)
                _BASE_SEEDS[name] = value
            base = _mix(base ^ value)
        seed: int = (base + _LAUNCHES * _SEED_STRIDE) & _SEED_MAX
        _LAUNCHES += 1
    return seed


def _reset_generator_seeds() -> None:
    """Forget the captured seeds and the launch count (used by the tests)."""
    global _LAUNCHES
    with _SEED_LOCK:
        _BASE_SEEDS.clear()
        _LAUNCHES = 0


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
