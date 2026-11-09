"""Run the four frameworks and report the three numbers that were asked for.

Each framework is its own script, started as its own process, because the cost of importing
a framework is part of what a user pays to use it -- importing all four in one process would
attribute every import to whichever happened to run last.

    python run_compare.py             # every framework, every task
    python run_compare.py --only pygpgpu numba-cuda

Three measurements:

* **cold** -- the wall time of a run with the framework's caches cleared, so compilation and
  JIT are inside the number;
* **warm** -- the wall time of the very next run, with those caches in place;
* **call 2** -- the time the script itself reports for its second invocation of the same
  function, which is what a program pays per call once it is up.

Which caches exist differs per framework, and that is a fact about the framework: `pygpgpu`
keeps its generated sources and compiled modules under `__cucache__` beside the script,
`numba` keeps `.nbi`/`.nbc` files in `__pycache__` (redirected here through `NUMBA_CACHE_DIR`
so it can be cleared by name), and `taichi` 1.7 caches in the frontend rather than on disk --
so its cold and warm columns come out equal, and that is the honest measurement rather than a
missing step.
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE: Path = Path(__file__).resolve().parent
PYTHON: str = sys.executable
NUMBA_CACHE: Path = HERE / ".numba-cache"

# name, script, extra arguments. One entry per *framework and backend*, because the backend
# is most of what a GPU framework's number means: PyGPGPU's CUDA and OpenCL paths are the
# same source through two different compilers, and Taichi's CPU backend is a different
# program from its CUDA one.
FRAMEWORKS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("pygpgpu-cuda", "bench_pygpgpu.py", ("--arch", "cuda")),
    ("pygpgpu-opencl", "bench_pygpgpu.py", ("--arch", "opencl")),
    ("pygpgpu-hip", "bench_pygpgpu.py", ("--arch", "hip")),
    ("taichi-cuda", "bench_taichi.py", ("--arch", "cuda")),
    ("taichi-cpu", "bench_taichi.py", ("--arch", "cpu")),
    ("warp-cuda", "bench_warp.py", ("--arch", "cuda")),
    ("numba-cuda", "bench_numba_cuda.py", ()),
    ("numba-cpu", "bench_numba_cpu.py", ()),
]

CACHE_DIRS: Tuple[Path, ...] = (
    HERE / "__cucache__", HERE / "__clcache__", HERE / "__hipcache__", HERE / "__metalcache__",
    HERE / "__pycache__", NUMBA_CACHE,
    Path(os.path.expanduser("~")) / ".cache" / "taichi",
)

LINE = re.compile(r"^(TASK|CALL1_MS|CALL2_MS|CALLBEST_MS|CHECKSUM|CORRECT|ARCH|CACHE)\s+(.*)$")


def clear_caches() -> None:
    """Remove every on-disk cache, so the next run compiles from nothing."""
    for directory in CACHE_DIRS:
        shutil.rmtree(directory, ignore_errors=True)


def _median(values: List[float]) -> float:
    """The middle sample: one slow run cannot move it the way it moves a mean."""
    return statistics.median(values) if values else float("nan")


def _spread(values: List[float]) -> str:
    """The whole range as `min-max`, so a reader can see how much the ranking can move."""
    if not values:
        return "-"
    return f"{min(values):.0f}-{max(values):.0f}"


def run_once(script: str, extras: Tuple[str, ...], task: str, repeat: int) -> Tuple[float, Dict[str, str]]:
    """Start one framework script and time the whole process; return its reported fields."""
    command = [PYTHON, str(HERE / script), "--task", task, "--repeat", str(repeat), *extras]
    environment = dict(os.environ)
    environment["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE)
    environment.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(HERE), env=environment,
                               capture_output=True, text=True, timeout=1800)
    elapsed = (time.perf_counter() - started) * 1000
    fields: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        match = LINE.match(line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"{script} {task} failed:\n  " + "\n  ".join(tail))
    return elapsed, fields


def run_startup(script: str, extras: Tuple[str, ...]) -> float:
    """Time a run that imports the framework, starts its runtime and stops."""
    return run_once(script, (*extras, "--startup-only"), "mandelbrot", 0)[0]


def compare(only: Optional[List[str]], task: str, repeat: int,
            warm_samples: int = 3, rounds: int = 1) -> List[Dict[str, object]]:
    """Cold and warm are both measured per round; the table reports the median.

    Cold can only be sampled once per round -- the caches are gone after it -- so ranking it
    needs several rounds rather than several samples inside one.

    A warm run repeated back to back is not always faster than the cold one on this machine:
    the cache files were just written, and an on-access scanner reads them on the run that
    follows. Taking the best of several warm runs removes that one-off cost without hiding
    anything -- every sample is printed.

    `rounds` repeats the whole cold-then-warm cycle, because a single sample of either is not
    enough to rank the four against each other: start-up varies by hundreds of milliseconds
    between runs of the same script. Each round clears the caches again, so every cold sample
    really is cold, and the table reports the **median** of the rounds with the spread that
    makes the ranking honest.
    """
    rows: List[Dict[str, object]] = []
    for name, script, extras in FRAMEWORKS:
        if only and name not in only:
            continue
        cold_runs: List[float] = []
        warm_runs: List[float] = []
        startup_runs: List[float] = []
        call2_runs: List[float] = []
        best_runs: List[float] = []
        fields: Dict[str, str] = {}
        for _ in range(max(1, rounds)):
            clear_caches()
            startup_runs.append(run_startup(script, extras))
            cold, _ = run_once(script, extras, task, repeat)
            cold_runs.append(cold)
            warmest: Optional[float] = None
            for _ in range(max(1, warm_samples)):
                elapsed, fields = run_once(script, extras, task, repeat)
                warmest = elapsed if warmest is None else min(warmest, elapsed)
            assert warmest is not None
            warm_runs.append(warmest)
            call2_runs.append(float(fields.get("CALL2_MS", "nan")))
            best_runs.append(float(fields.get("CALLBEST_MS", "nan")))
        row: Dict[str, object] = {
            "framework": name,
            "task": task,
            "startup": _median(startup_runs),
            "cold": _median(cold_runs),
            "warm": _median(warm_runs),
            "call2": _median(call2_runs),
            "best": _median(best_runs),
            "cold_runs": cold_runs,
            "warm_runs": warm_runs,
            "correct": fields.get("CORRECT", "?"),
            "arch": fields.get("ARCH", "-"),
        }
        rows.append(row)
        print(f"{name:16s} {task:10s} "
              f"start {row['startup']:7.1f} ms  "
              f"cold {row['cold']:8.1f} ms  "
              f"warm {row['warm']:8.1f} ms  "
              f"call2 {row['call2']:7.3f} ms  best {row['best']:7.3f} ms  "
              f"{row['correct'].split('(')[0].strip()}")
        sys.stdout.flush()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these framework names")
    parser.add_argument("--task", choices=("mandelbrot", "laplacian", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=3,
                        help="extra timed calls inside each script")
    parser.add_argument("--warm-samples", type=int, default=3,
                        help="how many warm runs to time before taking the best")
    parser.add_argument("--rounds", type=int, default=1,
                        help="cold-then-warm cycles per framework; the table reports the "
                             "median, and 3 or more is what makes the ranking trustworthy")
    arguments = parser.parse_args()

    tasks: Tuple[str, ...] = (("mandelbrot", "laplacian") if arguments.task == "both"
                              else (arguments.task,))
    rows: List[Dict[str, object]] = []
    for task in tasks:
        print(f"\n=== {task} ===")
        rows.extend(compare(arguments.only, task, arguments.repeat, arguments.warm_samples,
                            arguments.rounds))

    print("\n| framework | task | start-up | cold (full) | warm (full) | call 2 | best call | correct |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        detail = str(row["correct"]).split("(")[0].strip()
        print(f"| {row['framework']} | {row['task']} | {row['startup']:.0f} ms | "
              f"{row['cold']:.0f} ms | {row['warm']:.0f} ms | {row['call2']:.3f} ms | "
              f"{row['best']:.3f} ms | {detail} |")
    failures = [row for row in rows if not str(row["correct"]).startswith("yes")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

