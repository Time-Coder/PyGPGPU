"""One benchmark script: how the four are driven, and what they report.

Every `bench_*.py` is a standalone process that imports **only its own framework**. Two
things are measured from the outside and one from the inside, and all three are asked for:

* the wall time of the first run with the caches cleared -- compilation and JIT included;
* the wall time of the next run, with the caches in place;
* the in-process time of the second call of the same function, which is what a caller
  actually waits for once a program is up.

The first two can only be measured by the process that starts the script, so they belong to
`run_compare.py`; the third is measured here and printed as `CALL1_MS` / `CALL2_MS`.
Importing a second framework in this file would fold its import cost into the numbers, which
is exactly what the separation exists to avoid.
"""
from __future__ import annotations
import argparse
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", choices=("mandelbrot", "laplacian", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=3,
                        help="how many extra calls to time after the first two")
    parser.add_argument("--arch", default=None,
                        help="the backend this framework should pick, when it has a choice")
    parser.add_argument("--startup-only", action="store_true",
                        help="import the framework and initialise its device, then stop -- "
                             "what a process pays before it runs anything")
    parser.add_argument("--timeline", action="store_true",
                        help="print where the process spent its time: import, buffers, the "
                             "first call (which carries the compile or the cache read), and "
                             "the checks")
    return parser.parse_args()


# (label, seconds since the process's own T0), filled by `mark`. Off unless asked for, so a
# normal run pays nothing for the instrumentation.
_TIMELINE: List[Tuple[str, float]] = []


def mark(label: str, since: Optional[float] = None) -> None:
    """Record a milestone. `since` is the call's own `T0`, taken before its imports."""
    _TIMELINE.append((label, time.perf_counter() - (since if since is not None else _T0)))


def set_t0(since: float) -> None:
    """Adopt the script's own start, so every mark shares one time base.

    A script takes its `T0` before importing its framework; this module is imported after
    that, so without this the import mark and the later marks would be offsets from two
    different instants and the first gap would come out negative.
    """
    global _T0
    _T0 = since


_T0: float = time.perf_counter()


def print_timeline() -> None:
    """Report the milestones as offsets from the start of the run, with the gaps between."""
    if not _TIMELINE:
        return
    previous = 0.0
    print("TIMELINE (ms since this script started)")
    for label, offset in _TIMELINE:
        print(f"  {offset * 1000:9.1f}  (+{(offset - previous) * 1000:8.1f})  {label}")
        previous = offset
    sys.stdout.flush()


def _verdict(task: str, actual: np.ndarray, expected: np.ndarray) -> Tuple[bool, str]:
    """Whether a framework's answer is the right computation, and why.

    The two tasks need different tolerances. `laplacian` is a fixed sum of neighbours, so it
    is compared element for element. `mandelbrot` counts iterations of a chaotic recurrence:
    the kernels compute in float32 while the numpy reference uses float64, and near the
    escape boundary the two disagree by much more than one step -- a pixel whose orbit sits
    within a float32 ulp of the boundary can survive a hundred iterations longer. So the
    check is the *share* of pixels that differ at all, which is what separates "the same
    computation, rounded differently" from "a different computation": a swapped axis, a
    wrong constant or a misplaced loop differs in nearly every pixel, while rounding touches
    a fraction of a percent of them.
    """
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} != {expected.shape}"
    import common

    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if task == "laplacian":
        worst = float(difference.max())
        return worst <= 1e-3, f"max |diff| {worst:.3e}"
    differing = int(np.count_nonzero(difference > 0.0))
    total = int(difference.size)
    share = differing / total if total else 0.0
    worst = float(difference.max())
    ok = share <= 0.01
    return ok, (f"{differing}/{total} pixels differ ({share:.3%}), "
                f"max |diff| {worst:.0f} of {common.MANDEL_MAX_ITER}")


def report(task: str, run: Callable[[], object], repeat: int,
           fetch: Optional[Callable[[], np.ndarray]] = None) -> Dict[str, float]:
    """Call `run` once (compiling), then time the next calls, then check the answer.

    The first call is not timed for its kernel: for a JIT framework it carries the whole
    compile, which the caller sees as startup rather than as a launch, and mixing the two
    makes the steady-state number meaningless. It *is* reported as `CALL1_MS` so the
    one-off cost stays visible.

    `fetch` brings the answer back to plain numpy for the check, and is deliberately called
    **outside** the timed region: whether a framework's array is host- or device-resident is
    part of its design, and pulling the bytes over the bus on every iteration would measure
    the bus instead of the kernel. It defaults to `np.asarray`, which is right for the three
    frameworks whose array already is a numpy array.
    """
    import common

    mark(f"{task}: buffers ready")
    started = time.perf_counter()
    run()
    first = (time.perf_counter() - started) * 1000
    mark(f"{task}: first call done (compile or cache read inside)")

    second_started = time.perf_counter()
    run()
    second = (time.perf_counter() - second_started) * 1000
    mark(f"{task}: second call done")

    best = second
    for _ in range(max(0, repeat - 1)):
        again = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - again) * 1000)

    expected = common.reference(task)
    mark(f"{task}: reference loaded")
    actual = np.asarray(fetch() if fetch is not None else run())
    ok, detail = _verdict(task, actual, expected)
    print(f"TASK {task}")
    print(f"CALL1_MS {first:.3f}")
    print(f"CALL2_MS {second:.3f}")
    print(f"CALLBEST_MS {best:.3f}")
    print(f"CHECKSUM {common.checksum(actual):.6f}")
    print(f"EXPECTED {common.checksum(expected):.6f}")
    print(f"CORRECT {'yes' if ok else 'NO'} ({detail})")
    sys.stdout.flush()
    return {"call1": first, "call2": second, "best": best}


def run_tasks(args: argparse.Namespace,
              factory: Callable[[str], Tuple[Callable[[], object],
                                             Optional[Callable[[], np.ndarray]]]]) -> None:
    """Run the requested tasks through one framework's `factory`.

    `factory` answers the timed call and, optionally, the untimed fetch that brings the
    answer home for the check.
    """
    names: Tuple[str, ...] = (("mandelbrot", "laplacian") if args.task == "both"
                              else (args.task,))
    if getattr(args, "startup_only", False):
        # The framework is imported and (for taichi) its runtime started by the time this
        # runs, so the process it belongs to measures exactly what a caller pays before the
        # first kernel exists.
        print("STARTUP_ONLY yes")
        sys.stdout.flush()
        return
    for name in names:
        run, fetch = factory(name)
        report(name, run, args.repeat, fetch)
    if getattr(args, "timeline", False):
        print_timeline()


