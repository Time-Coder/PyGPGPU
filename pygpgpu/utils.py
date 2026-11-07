from __future__ import annotations

import contextlib
import hashlib
import re
import os
import pickle
import threading
import time
import asyncio
from functools import lru_cache
from collections.abc import Iterable
from typing import List, Optional, Sequence, Tuple, Union, cast

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

def _atomic_write(file_name:str, data:bytes)->None:
    """Write `data` to `file_name` in a way no concurrent reader can observe half-done.

    The write goes to a temp file in the same directory and is moved into place with
    `os.replace`, which is atomic on every platform this targets, so a reader sees either
    the previous file or the finished new one. The temp name carries both the pid and the
    thread id: two writers in one process and two processes must never share a temp file, or
    the replace would install one writer's bytes over the other's unfinished ones.

    `os.replace` can still refuse on Windows with a sharing violation when another thread
    happens to hold the destination open for a read at that instant -- the name is in use,
    not the data. That is transient by construction, so it is retried briefly; a failure
    that outlives the retries falls back to a plain overwrite, which is the pre-existing
    behaviour rather than a new error.
    """
    temp_name:str = f"{file_name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temp_name, "wb") as file:
            file.write(data)
        for _attempt in range(8):
            try:
                os.replace(temp_name, file_name)
                return
            except PermissionError:
                # A reader has the destination open; let it finish and try again.
                time.sleep(0.05)
        with open(file_name, "wb") as file:
            file.write(data)
    finally:
        with contextlib.suppress(OSError):
            if os.path.isfile(temp_name):
                os.unlink(temp_name)


def save_var(var:object, file_name:str)->None:
    """Pickle `var` to its cache file, atomically (see `_atomic_write`)."""
    _ensure_directory(os.path.dirname(os.path.abspath(file_name)))
    _atomic_write(file_name, pickle.dumps(var))

def load_var(file_name:str)->object:
    with open(file_name, "rb") as file:
        return pickle.load(file)
    
def _ensure_directory(directory:str)->None:
    """Make `directory` exist, tolerating the losers of a concurrent create.

    `os.makedirs(..., exist_ok=True)` is not atomic on Windows: it creates each missing
    segment with its own call and can still raise for a segment another thread just made, so
    two writers that both find a parent missing can fail the one that was only second. The
    levels are therefore raised one segment at a time, and each segment's failure is
    re-checked against the filesystem: if the segment exists now, another writer created
    exactly the directory this call needed, so there is nothing to report.
    """
    if not directory:
        return
    absolute:str = os.path.abspath(directory)
    head, tail = os.path.split(absolute)
    if tail:
        _ensure_directory(head)
    for _attempt in range(8):
        try:
            os.mkdir(absolute)
            return
        except FileExistsError:
            if os.path.isdir(absolute):
                return
        except OSError:
            if os.path.isdir(absolute):
                return
            time.sleep(0.05)
    if os.path.isdir(absolute):
        return
    os.makedirs(absolute, exist_ok=True)


def save_bin(bin:Union[bytes, bytearray], file_name:str)->None:
    """Write a compiled binary to its cache file, atomically.

    The file is one compiled artifact and several processes may be compiling the same
    module at once. A reader could see a half-written file and load a truncated binary
    (`CL_INVALID_KERNEL`), and two writers interleaving their bytes would leave one that is
    a mix of both. The write therefore goes to a temp file in the same directory and is
    moved into place with `os.replace`, which is atomic on every platform this targets -- so
    a reader sees either the previous artifact or the finished new one, never a partial one.
    """
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    _ensure_directory(target_folder)
    _atomic_write(file_name, bytes(bin))

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

def device_limits(device: object) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
    """Best-effort (max_work_group_size, max_work_item_sizes) for a backend device.

    OpenCL devices expose these names directly; CUDA/HIP expose the equivalent
    CUDA attributes (max_threads_per_block / max_block_dim_x..z). Returns
    (None, None) when nothing is available, so callers can fall back to a
    conservative default instead of failing.
    """
    def read(name: str) -> object:
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
            # What comes back is a per-axis sequence on every backend this reads; the cast
            # states that, and the `except` below is what a value of another shape hits.
            sequence = cast("Sequence[object]", value)
            items = tuple(int(v) for v in sequence if int(v) > 0)
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


def blocks_to_global(blocks: Tuple[int, ...], threads: Tuple[int, ...]
                     ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(blocks_per_grid, threads_per_block)`` -> ``(global_work_size, local_work_size)``.

    Every bracket in this package states a launch the way a CUDA or HIP programmer states
    one, on **every** backend: a grid of so many blocks, each holding so many threads.
    OpenCL's own API speaks a different pair -- the total number of work items and the size
    of one work group -- so the translation happens once, here, and nothing downstream has
    to know which spelling the caller used.

    ``global = blocks * threads`` and ``local = threads``. Two consequences follow and both
    are wanted. First, the total always divides by the group size -- the quotient *is* the
    block count -- so a divisibility check has nothing left to reject. Second, a CUDA or HIP
    launch derives ``grid_dims = global // local``, which is ``blocks`` by construction, so
    the three backends agree on the same launch from the same bracket.

    The ranks have to match: two axes cannot say how many threads a three-axis grid holds.
    """
    block_extents:Tuple[int, ...] = tuple(int(b) for b in blocks)
    thread_extents:Tuple[int, ...] = tuple(int(t) for t in threads)
    if len(block_extents) != len(thread_extents):
        raise ValueError(
            f"the work size bracket {(block_extents, thread_extents)} mixes a "
            f"{len(block_extents)}-dimensional block count with a "
            f"{len(thread_extents)}-dimensional thread count; both have to name the same "
            f"number of axes")
    global_size:Tuple[int, ...] = tuple(b * t for b, t in zip(block_extents, thread_extents))
    return global_size, thread_extents


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

# The process-wide default every backend's own flag starts from. A backend module is imported
# lazily -- the call that first needs it is what pulls it in -- so flipping only the flags of
# the modules already loaded left `set_print_info(False)` at the top of a script with no
# effect: the class did not exist yet, and it then defaulted to verbose. Each module now takes
# its initial value from here, which is what makes the order of the two calls irrelevant.
_PRINT_INFO_ENABLED:bool = True


def initial_print_info() -> bool:
    """The process-wide default for a backend's own ``print_info`` flag.

    A backend calls this where it declares the flag, so a module imported *after*
    ``set_print_info(False)`` starts quiet. The flag stays an ordinary class attribute
    afterwards; this only decides what it starts as.
    """
    return _PRINT_INFO_ENABLED


def nvidia_package_libraries(names: List[str]) -> List[str]:
    """Libraries shipped by the pip ``nvidia-*-cu12`` packages, newest first.

    On Linux the CUDA runtime is commonly installed with pip (``nvidia-cuda-runtime-
    cu12``, ``nvidia-cuda-nvrtc-cu12``) instead of a system toolkit. Those wheels
    drop their ``lib*.so.<N>`` files under ``site-packages/nvidia/<component>/lib``,
    which is not on the loader's search path, so a library the driver loads by bare
    SONAME is not found. This returns the matching files so a caller can load one by
    its full path.

    ``names`` are the SONAME stems ("libnvrtc", "libcudart"); nothing is returned on
    a platform where pip does not install CUDA this way.
    """
    try:
        import site
    except Exception:
        return []
    try:
        directories: List[str] = []
        try:
            directories.extend(site.getsitepackages())
        except AttributeError:                  # a virtualenv without that helper
            directories.append(os.path.dirname(os.path.dirname(os.__file__)))
        user_site = getattr(site, "getusersitepackages", None)
        if user_site is not None:
            directories.append(user_site())
    except Exception:
        return []

    found: List[str] = []
    seen: set = set()
    for directory in directories:
        component_dir: str = os.path.join(directory, "nvidia")
        if not os.path.isdir(component_dir):
            continue
        try:
            components = sorted(os.listdir(component_dir))
        except OSError:
            continue
        for component in components:
            binary_dir: str = os.path.join(component_dir, component, "lib")
            if not os.path.isdir(binary_dir):
                continue
            try:
                file_names = sorted(os.listdir(binary_dir), reverse=True)
            except OSError:
                continue
            for file_name in file_names:
                for name in names:
                    if file_name == name + ".so" or file_name.startswith(name + ".so."):
                        candidate: str = os.path.join(binary_dir, file_name)
                        if candidate not in seen:
                            seen.add(candidate)
                            found.append(candidate)
                        break
    return found


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


def settle_future(future:object, error:Optional[BaseException]=None) -> None:
    """Complete an asyncio or concurrent future from any thread.

    Driver completion callbacks run on a driver thread, not on the event loop's.
    Setting the result directly does not wake a loop that is blocked in
    ``select()``, so a plain ``await`` on such a future hangs even though the work
    finished -- only ``asyncio.wait_for`` appeared to work, because it wraps the
    future in a Task whose completion re-schedules the loop. Scheduling the
    completion through the loop keeps a bare await responsive.
    """
    if isinstance(future, asyncio.Future):
        loop = future.get_loop()
        if loop.is_running():
            if error is None:
                loop.call_soon_threadsafe(future.set_result, None)
            else:
                loop.call_soon_threadsafe(future.set_exception, error)
            return
    if error is None:
        future.set_result(None)
    else:
        future.set_exception(error)


def _asyncio_future_of(asyncio_module:object) -> object:
    """An ``asyncio.Future`` bound to a loop that exists, even off the loop's thread.

    ``asyncio.Future()`` demands a running (or current) event loop, and a ``submit``-
    style call may come from a thread that has neither -- the future is then meant for
    a loop the caller awaits on later. The python backend solves this with
    ``_current_loop`` (install a fresh loop when none is set); this is the same rule,
    callable without importing asyncio in the caller. Kept private: it exists so a
    backend's ``async_call`` can spell ``future_type()`` for both future kinds.
    """
    current_loop:object = _current_loop()
    return asyncio_module.Future(loop=current_loop)


def _current_loop() -> object:
    """The running loop, else the thread's current loop, else a fresh installed one.

    The same contract the python backend's own bridge states: ``async_call`` cannot
    be awaited outside a loop anyway, but returning a future bound to an installed
    loop still lets a caller drive it with ``loop.run_until_complete(future)``
    instead of failing at creation time.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    loop:Optional[asyncio.AbstractEventLoop] = None
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None  # no loop is set on this thread (or reading it is no longer allowed)
    if loop is not None and not loop.is_closed():
        return loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def set_print_info(enabled:bool) -> None:
    """Turn the progress logging of every backend on or off, now and from now on.

    Each backend keeps its own ``print_info`` flag (``CL``, ``CUDA``, ``NVRTC``, ``HIP`` and
    ``HipCompiler``), all starting at ``True``, so a single kernel call prints translation,
    compilation, copy and launch lines. This flips them all in one call and only touches
    modules already in ``sys.modules``, so it never pulls in a GPU runtime just to silence
    output::

        from pygpgpu import set_print_info
        set_print_info(False)      # quiet
        set_print_info(True)       # verbose again (the default)

    The call may be made before the backends are imported, which is the usual place for it:
    it also records the value as the process-wide default, and a backend imported later takes
    its flag's initial value from there. Only flipping the loaded modules would have left
    ``set_print_info(False)`` at the top of a script with no effect at all, because the
    backend is imported lazily by the first call that needs it and its class did not exist
    yet -- the script then printed everything it asked to silence.

    A single backend can still be silenced on its own with its own attribute, e.g.
    ``pygpgpu.opencl.driver.CL.print_info = False``; that does not change the default for
    backends imported later, unlike this call. The math-library binders under
    ``pygpgpu/{cuda,hip}/libraries`` keep their own flag and are left alone: what they print
    is a warning about a binding that failed, not progress logging.
    """
    global _PRINT_INFO_ENABLED
    import sys

    _PRINT_INFO_ENABLED = enabled
    for module_name in _PRINT_INFO_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and "print_info" in vars(value):
                setattr(value, "print_info", enabled)
