# The work-size heuristics: one grid, per backend, from the caller's extents.
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, cast



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

# device -> the limits read from it, so a launch does not query the driver for them again
# (see `device_limits`). Keyed by the device object, which every backend compares by handle.
_DEVICE_LIMITS:Dict[object, Tuple[Optional[int], Optional[Tuple[int, ...]]]] = {}


def device_limits(device: object) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
    """Best-effort (max_work_group_size, max_work_item_sizes) for a backend device.

    OpenCL devices expose these names directly; CUDA/HIP expose the equivalent
    CUDA attributes (max_threads_per_block / max_block_dim_x..z). Returns
    (None, None) when nothing is available, so callers can fall back to a
    conservative default instead of failing.

    The answer is remembered per device: reading it is a *driver* round trip on CUDA and
    HIP (`cuDeviceGetAttribute`), the launch asks for it every call, and a device's limits
    do not change. The result is an int and a tuple, so every caller can share it.
    """
    try:
        cached:Optional[Tuple[Optional[int], Optional[Tuple[int, ...]]]] = _DEVICE_LIMITS.get(
            device)
    except TypeError:
        # A device that is not hashable cannot be remembered; reading it again is correct.
        return _read_device_limits(device)
    if cached is not None:
        return cached
    limits:Tuple[Optional[int], Optional[Tuple[int, ...]]] = _read_device_limits(device)
    _DEVICE_LIMITS[device] = limits
    return limits


def _read_device_limits(device: object) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
    """Read a device's limits from the backend, one attribute query each."""
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

