from __future__ import annotations

import hashlib
import re
import os
import pickle
from functools import lru_cache
from collections.abc import Iterable
from typing import Union, Any, Tuple, Optional, List

import numpy as np


def product(values:Iterable[int])->int:
    """Product of an iterable (kept local so callers need no math import)."""
    total:int = 1
    for value in values:
        total *= value
    return total


def modify_time(file_path:str)->float:
    if not os.path.isfile(file_path):
        return 0
    
    return os.path.getmtime(file_path)

def md5sum(data_or_path:Union[str, np.ndarray, bytes, bytearray])->str:
    md5_hash = hashlib.md5()

    if isinstance(data_or_path, str):
        chunk_size:int = 4096
        
        with open(data_or_path, "rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                md5_hash.update(chunk)
    elif isinstance(data_or_path, (bytes, bytearray)):
        md5_hash.update(data_or_path)

    elif isinstance(data_or_path, np.ndarray):
        md5_hash.update(data_or_path.tobytes())
    
    return md5_hash.hexdigest()

def md5sums(content:str)->str:
    return md5sum(content.encode("utf-8"))

def cat(file_name: str) -> str:
    # charset_normalizer ships with imageio, but it was never a declared
    # dependency and importing it at module scope made this module -- and
    # therefore the whole package -- fail to import wherever it is absent.
    # It is only needed to guess the encoding of an unreadable file, so import
    # it on demand and fall back to UTF-8 when it is not installed.
    try:
        from charset_normalizer import from_path
    except ImportError:
        with open(file_name, "r", encoding="utf-8", errors="replace") as file:
            return file.read()
    return str(from_path(file_name).best())
    
def echo(content:str, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "w", encoding='utf-8') as file:
        file.write(content)

def save_var(var:Any, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "wb") as file:
        pickle.dump(var, file)

def load_var(file_name:str)->Any:
    with open(file_name, "rb") as file:
        return pickle.load(file)
    
def save_bin(bin:Union[bytes, bytearray], file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "wb") as file:
        file.write(bin)

def load_bin(file_name:str)->bytes:
    with open(file_name, "rb") as file:
        return file.read()

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    illegal_chars = r'[\\/:*?"<>|]'
    safe_name = re.sub(illegal_chars, '_', filename)
    
    safe_name = safe_name.strip().strip('.')
    
    if not safe_name:
        safe_name = "unnamed"
    
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    if safe_name.upper() in reserved_names:
        safe_name = f"_{safe_name}_"
    
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name

def complete_basename(file_name:str):
    return '.'.join(os.path.basename(file_name).split('.')[:-1])

# Warp size on every current NVIDIA/AMD GPU; a work group that is not a multiple of
# it leaves a partially idle warp.
_WARP_SIZE:int = 32

# Upper bound on search nodes while choosing a work group, so a highly composite extent
# with a large budget cannot blow up the search. The first candidates are the largest,
# which are also the ones that fit the budget, so the best group is found early.
_MAX_SEARCH_NODES:int = 1500


def _divisors(n: int, cap: int) -> List[int]:
    """Descending divisors of n that do not exceed cap."""
    n = int(n)
    if n <= 0:
        return [1]
    cap = max(1, int(cap))
    result:List[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i <= cap:
                result.append(i)
            other = n // i
            if other != i and other <= cap:
                result.append(other)
        i += 1

    return sorted(result, reverse=True) or [1]


def _pick_local_size(extents: Tuple[int, ...], budget: int,
                     max_item_sizes: Optional[Tuple[int, ...]] = None,
                     allow_padding: bool = False) -> Tuple[int, ...]:
    """Pick a per-dimension local size for the given extents.

    With `allow_padding=False` every chosen value must divide its extent, so the
    derived global work size equals the extent exactly and no work item ever runs
    past the array (used where the kernel has no bounds guard).

    With `allow_padding=True` the values only have to be <= the extent, which keeps
    work groups efficient for awkward sizes (e.g. a prime extent); the caller must
    then guarantee a bounds guard, because the global size is rounded up.

    Among the combinations whose product fits the budget, a warp-aligned group is
    preferred, then the largest product, then the most even split across dimensions.
    """
    ndim:int = len(extents)
    if ndim == 0:
        return ()
    budget = max(1, int(budget))

    per_dim:List[List[int]] = []
    for d, extent in enumerate(extents):
        extent = max(1, int(extent))
        cap = budget
        if max_item_sizes is not None and d < len(max_item_sizes):
            cap = min(cap, max(1, int(max_item_sizes[d])))
        cap = min(cap, extent)
        if allow_padding:
            # Every value up to the cap is admissible for this dimension.
            candidates = list(range(cap, 0, -1))
        else:
            # Only divisors keep global == extent.
            candidates = _divisors(extent, cap)
        per_dim.append(candidates or [1])

    best:Optional[Tuple[int, ...]] = None
    best_key:Optional[Tuple[int, float]] = None
    nodes:int = 0
    ratios:List[float] = [0.0] * ndim

    def walk(d: int, current: List[int], product: int) -> bool:
        """Depth-first over descending candidates; returns False once the node budget is spent."""
        nonlocal best, best_key, nodes
        nodes += 1
        if nodes > _MAX_SEARCH_NODES:
            return False
        if d == ndim:
            avg = sum(ratios) / ndim
            imbalance = sum((r - avg) ** 2 for r in ratios)
            # Prefer warp-aligned groups: a group that is not a multiple of the warp
            # size leaves a partially idle warp, which measurably beats a marginally
            # larger but misaligned group.
            aligned = 1 if product % _WARP_SIZE == 0 else 0
            key = (aligned, product, -imbalance)
            if best_key is None or key > best_key:
                best_key = key
                best = tuple(current)
            return True

        remaining = 1
        for d2 in range(d + 1, ndim):
            remaining *= per_dim[d2][0]
        for value in per_dim[d]:
            scaled = product * value
            if scaled > budget:
                continue
            # Once a warp-aligned group is known, a branch that cannot reach its product
            # is dead: alignment is the primary key, product the tie-break.
            if (best_key is not None and best_key[0] == 1
                    and scaled * remaining < best_key[1]):
                break
            current.append(value)
            ratios[d] = value / max(1, extents[d])
            alive = walk(d + 1, current, scaled)
            current.pop()
            if not alive:
                return False
        return True

    walk(0, [], 1)
    return best if best is not None else tuple(1 for _ in range(ndim))


@lru_cache(maxsize=256)
def _pick_local_size_cached(extents: Tuple[int, ...], budget: int,
                            max_item_sizes: Optional[Tuple[int, ...]],
                            allow_padding: bool) -> Tuple[int, ...]:
    return _pick_local_size(extents, budget, max_item_sizes, allow_padding)


def detect_work_size(total_size: int, shape: Tuple[int, ...],
                     max_work_group_size: Optional[int] = None,
                     max_work_item_sizes: Optional[Tuple[int, ...]] = None,
                     allow_padding: bool = False
                     ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Derive (global, local) work sizes for the given array shape.

    `total_size` is the preferred work-group size; it is additionally clamped to
    the device's real limits when they are known. With `allow_padding=False` the
    local size divides every extent, so global equals the shape exactly and no
    work item runs past the array (the caller then needs no bounds guard).
    """
    budget:int = int(total_size)
    if max_work_group_size is not None:
        budget = min(budget, int(max_work_group_size))
    # Called on every launch, so memoize: repeated calls with the same shape (the
    # common case for a hot kernel) must not re-run the search.
    local_size = _pick_local_size_cached(
        tuple(int(e) for e in shape), budget,
        tuple(max_work_item_sizes) if max_work_item_sizes is not None else None,
        allow_padding)
    global_size = detect_global_size(local_size, shape)
    return global_size, local_size

def detect_global_size(local_size:Tuple[int, ...], shape: Tuple[int, ...]):
    global_ws = []
    for s, l in zip(shape, local_size):
        if l == 0:
            raise ValueError("local size cannot be zero")
        g = ((int(s) + l - 1) // l) * l
        global_ws.append(g)

    return tuple(global_ws)

def detect_local_size(total_size: int, shape: Tuple[int, ...],
                     max_work_group_size: Optional[int] = None,
                     max_work_item_sizes: Optional[Tuple[int, ...]] = None,
                     allow_padding: bool = False
                     ) -> Tuple[int, ...]:
    """Local work size whose product fits `total_size` and divides every extent."""
    budget:int = int(total_size)
    if max_work_group_size is not None:
        budget = min(budget, int(max_work_group_size))
    if not shape:
        raise ValueError("shape cannot be empty")
    return _pick_local_size(tuple(shape), budget, max_work_item_sizes, allow_padding)

def join_with_and(items):
    if not items:
        return ''
    
    if len(items) == 1:
        return items[0]
    
    if len(items) == 2:
        return ' and '.join(items)
    
    return ', '.join(items[:-1]) + ', and ' + items[-1]

def device_limits(device: Any) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
    """Best-effort (max_work_group_size, max_work_item_sizes) for a backend device.

    OpenCL devices expose these names directly; CUDA/HIP expose the equivalent
    CUDA attributes (max_threads_per_block / max_block_dim_x..z). Returns
    (None, None) when nothing is available, so callers can fall back to a
    conservative default instead of failing.
    """
    def read(name: str) -> Any:
        try:
            value = getattr(device, name, None)
            return value() if callable(value) else value
        except Exception:
            return None

    max_group:Optional[int] = None
    max_items:Optional[Tuple[int, ...]] = None

    for name in ("max_work_group_size", "max_threads_per_block"):
        value = read(name)
        if isinstance(value, int) and value > 0:
            max_group = value
            break

    value = read("max_work_item_sizes")
    if value is not None:
        try:
            items = tuple(int(v) for v in value if int(v) > 0)
            if items:
                max_items = items
        except Exception:
            max_items = None

    if max_items is None:
        dims = [read("max_block_dim_x"), read("max_block_dim_y"), read("max_block_dim_z")]
        if all(isinstance(v, int) and v > 0 for v in dims):
            max_items = (dims[0], dims[1], dims[2])

    return max_group, max_items


def local_size_for_global(global_size: Tuple[int, ...], total_local_size: int = 256,
                          max_work_group_size: Optional[int] = None,
                          max_work_item_sizes: Optional[Tuple[int, ...]] = None
                          ) -> Tuple[int, ...]:
    """Derive a local_work_size that divides every dimension of `global_size`.

    Used when the caller supplies only a global size. Every value divides its
    dimension, so the resulting work groups tile the requested global size exactly
    and no work item runs past it; the caller must still clamp inside the kernel
    when the global exceeds the real array extent.
    """
    extents:Tuple[int, ...] = tuple(int(g) for g in global_size)
    if not extents:
        return ()

    budget:int = max(1, int(total_local_size))
    if max_work_group_size is not None:
        budget = min(budget, max(1, int(max_work_group_size)))

    return _pick_local_size(extents, budget, max_work_item_sizes)


# ---------- progress-logging switch ----------

# Modules that own a ``print_info`` flag. Kept here, below every backend, so the
# single source of truth is importable from any layer without pulling a GPU
# runtime in: the modules are looked up in sys.modules and skipped when absent.
_PRINT_INFO_MODULES:Tuple[str, ...] = (
    "pygpgpu.opencl.driver.cl",
    "pygpgpu.cuda.driver.cuda",
    "pygpgpu.cuda.driver.nvrtc",
    "pygpgpu.hip.driver.hip",
    "pygpgpu.hip.oop.hip_compiler",
)


def get_print_info() -> bool:
    """Whether progress logging is currently on for any already-imported backend.

    A backend that has not been imported yet keeps its default (verbose) flag, so
    this reports False only when every loaded backend has been silenced. That is
    what the copy logger wants: it should stay quiet when the backends the caller
    actually uses are quiet.
    """
    import sys

    for module_name in _PRINT_INFO_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and vars(value).get("print_info", False):
                return True
    return False


def set_print_info(enabled:bool) -> None:
    """Turn the progress logging of every already-imported backend on or off.

    Each backend keeps its own ``print_info`` flag (``CL``, ``CUDA``, ``NVRTC``,
    ``HIP`` and ``HipCompiler``), all defaulting to ``True``, so a single kernel
    call prints translation, compilation, copy and launch lines. This flips them
    all in one call and only touches modules already in ``sys.modules``, so it
    never pulls in a GPU runtime just to silence output::

        from pygpgpu import set_print_info
        set_print_info(False)      # quiet
        set_print_info(True)       # verbose again (the default)

    A backend imported *after* this call starts out verbose, because its flag is a
    class attribute that only exists once its module is loaded. To keep the whole
    process quiet, call this after the backends you use have been imported -- or
    silence one backend directly, e.g. ``pygpgpu.opencl.driver.CL.print_info = False``.
    """
    import sys

    for module_name in _PRINT_INFO_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and "print_info" in vars(value):
                setattr(value, "print_info", enabled)
