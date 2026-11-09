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
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE: Path = Path(__file__).resolve().parent
PYTHON: str = sys.executable
NUMBA_CACHE: Path = HERE / ".numba-cache"

# name, script, extra arguments, the cache paths that keep it warm
FRAMEWORKS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("pygpgpu", "bench_pygpgpu.py", ()),
    ("taichi", "bench_taichi.py", ("--arch", "cuda")),
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


def compare(only: Optional[List[str]], task: str, repeat: int,
            warm_samples: int = 3) -> List[Dict[str, object]]:
    """Cold is one sample (the caches can only be cold once); warm is the best of several.

    A warm run repeated back to back is not always faster than the cold one on this machine:
    the cache files were just written, and an on-access scanner reads them on the run that
    follows. Taking the best of several warm runs removes that one-off cost without hiding
    anything -- every sample is printed.
    """
    rows: List[Dict[str, object]] = []
    for name, script, extras in FRAMEWORKS:
        if only and name not in only:
            continue
        clear_caches()
        cold, _ = run_once(script, extras, task, repeat)
        samples: List[float] = []
        warm_fields: Dict[str, str] = {}
        for _ in range(max(1, warm_samples)):
            elapsed, warm_fields = run_once(script, extras, task, repeat)
            samples.append(elapsed)
        warm = min(samples)
        row: Dict[str, object] = {
            "framework": name,
            "task": task,
            "cold": cold,
            "warm": warm,
            "samples": samples,
            "call2": float(warm_fields.get("CALL2_MS", "nan")),
            "best": float(warm_fields.get("CALLBEST_MS", "nan")),
            "correct": warm_fields.get("CORRECT", "?"),
            "arch": warm_fields.get("ARCH", "-"),
        }
        rows.append(row)
        print(f"{name:11s} {task:10s} cold {cold:8.1f} ms   "
              f"warm {warm:8.1f} ms {[round(s) for s in samples]}   "
              f"call2 {row['call2']:7.3f} ms   correct {row['correct']}")
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
    arguments = parser.parse_args()

    tasks: Tuple[str, ...] = (("mandelbrot", "laplacian") if arguments.task == "both"
                              else (arguments.task,))
    rows: List[Dict[str, object]] = []
    for task in tasks:
        print(f"\n=== {task} ===")
        rows.extend(compare(arguments.only, task, arguments.repeat, arguments.warm_samples))

    print("\n| framework | task | cold script | warm script | call 2 | best call | correct |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        detail = str(row["correct"]).split("(")[0].strip()
        print(f"| {row['framework']} | {row['task']} | {row['cold']:.0f} ms | "
              f"{row['warm']:.0f} ms | {row['call2']:.3f} ms | {row['best']:.3f} ms | "
              f"{detail} |")
    failures = [row for row in rows if not str(row["correct"]).startswith("yes")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
