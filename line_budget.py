# -*- coding: utf-8 -*-
"""The file-size budget: no first-party source file may pass 500 lines.

`docs/file-split-plan.md` records the policy. This script is the part that makes it true: it
walks the first-party sources, applies the exemptions the plan names, and fails when a file is
over the ceiling without being on the allowlist below.

The allowlist is the backlog, and it is deliberately explicit and countable. Finishing a split
means deleting its entry here, in the same commit, so the remaining work is always the length
of the list. An entry records the line count the file had when it was accepted, and a file that
grows past its recorded count fails as well - otherwise an allowlisted file could quietly get
worse while it waits.

Usage:
    python line_budget.py            the check (exit 1 on a violation)
    python line_budget.py --list     also print the warning band (351-500 lines)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterator, List, Tuple

# The hard ceiling, in lines. Everything below is measured in physical lines, so a blank line
# counts: that is what a reviewer scrolls past, which is the thing the ceiling is about.
CEILING: int = 500

# The band the plan calls "warning": allowed, but every file here should be split by the time
# the area around it is touched. Printed by --list, never a failure.
WARNING: int = 350

# Source kinds the budget applies to. Device headers are in scope (`kernel/include/complex.h`
# is 1,000+ lines of our own code); the vendored SDK headers are excluded by path below.
EXTENSIONS: Tuple[str, ...] = (".py", ".cl", ".cu", ".hip", ".metal", ".h")

# Directories that are never walked. `pcpp` is the vendored C preprocessor, the driver
# `include` trees are the vendor SDK headers, and the rest are caches and build output.
SKIP_DIRECTORIES: Tuple[str, ...] = (
    ".git", ".ruff_cache", ".pytest_cache", "__pycache__", "pygpgpu.egg-info",
    "build", "dist", "pcpp", "__clcache__", "__cucache__", "__hipcache__", "__metalcache__",
)

# Path fragments (posix-style, so this works on either platform) that opt a file out.
EXEMPT_FRAGMENTS: Tuple[str, ...] = (
    "/driver/include/",   # the CUDA / OpenCL SDK headers, copied verbatim
    "/pcpp/",             # the vendored preprocessor
)

# Generated stubs are written by the import hooks and by vec_types/helper.py; a stub describes
# one class and cannot be assembled from parts.
EXEMPT_EXTENSIONS: Tuple[str, ...] = (".pyi",)

# The backlog: file -> (recorded line count, why it is still over the ceiling).
# Removing an entry is the act of finishing that split.
ALLOWLIST: Dict[str, Tuple[int, str]] = {
    "pygpgpu/cparser/ShaderBuiltins.py": (16874, "generated; delete-or-split decision pending (plan 5.6)"),
    "pygpgpu/cparser/cparser.py": (1200, "unreachable legacy chain (plan 5.6)"),
    "pygpgpu/kernel/decorator/pykernel_compile.py": (529, "_compile is one flow whose positional signature is pinned by nine test call sites (plan 5.2)"),
    "pygpgpu/kernel/decorator/__init__.py": (330, "wave 2 done: the class is assembled from subject mixins (plan 5.2)"),
    "pygpgpu/kernel/include/complex.h": (1064, "device header; split last (plan 5.3)"),
    "pygpgpu/opencl/driver/clinfo.py": (1940, "phase 5 - data modules (plan 5.4.1)"),
    "pygpgpu/cuda/driver/cuinfo.py": (935, "phase 5 - data modules (plan 5.4.1)"),
    "pygpgpu/cuda/oop/program_parser/struct_builder.py": (280, "phase 5 - struct-builder mixin (plan 5.4.2)"),
    "pygpgpu/hip/oop/program_parser/struct_builder.py": (280, "phase 5 - struct-builder mixin (plan 5.4.2)"),
    "pygpgpu/opencl/oop/program_parser/struct_builder.py": (266, "phase 5 - struct-builder mixin (plan 5.4.2)"),
    "pygpgpu/cparser/ts_parser.py": (827, "phase 7 - mixin split (plan 5.6)"),
    "tests/test_metal_launch_path.py": (1621, "phase 8 - test split (plan 5.7)"),
    "tests/test_metal_codegen.py": (1502, "phase 8 - test split (plan 5.7)"),
    "tests/test_kernel_complex.py": (1385, "phase 8 - test split (plan 5.7)"),
    "tests/test_cuda_libraries.py": (672, "phase 8 - test split (plan 5.7)"),
    "tests/test_hip_libraries.py": (554, "phase 8 - test split (plan 5.7)"),
    "tests/test_kernel_object_arguments.py": (529, "phase 8 - test split (plan 5.7)"),
}


def source_files(root: str) -> Iterator[Tuple[str, str]]:
    """Yield (relative posix path, absolute path) for every file the budget applies to."""
    for directory, sub_directories, names in os.walk(root):
        sub_directories[:] = [name for name in sub_directories if name not in SKIP_DIRECTORIES]
        for name in names:
            if os.path.splitext(name)[1] not in EXTENSIONS:
                continue
            full: str = os.path.join(directory, name)
            relative: str = os.path.relpath(full, root).replace(os.sep, "/")
            if any(fragment in "/" + relative for fragment in EXEMPT_FRAGMENTS):
                continue
            yield relative, full


def line_count(path: str) -> int:
    """Physical lines, counting blank ones: what a reviewer scrolls past."""
    with open(path, "rb") as handle:
        return len(handle.read().splitlines())


def main(argv: List[str]) -> int:
    root: str = os.path.dirname(os.path.abspath(__file__))
    violations: List[str] = []
    grown: List[str] = []
    backlog: List[Tuple[int, str]] = []
    warning: List[Tuple[int, str]] = []

    for relative, full in source_files(root):
        count: int = line_count(full)
        allowed: Tuple[int, str] | None = ALLOWLIST.get(relative)
        if count > CEILING:
            if allowed is None:
                violations.append(f"{count:6}  {relative}  (not on the allowlist)")
            elif count > allowed[0]:
                grown.append(f"{count:6}  {relative}  (allowlisted at {allowed[0]})")
            else:
                backlog.append((count, relative))
        elif count > WARNING:
            warning.append((count, relative))

    backlog.sort(reverse=True)
    warning.sort(reverse=True)

    print(f"line budget: ceiling {CEILING} lines, warning band over {WARNING}")
    print(f"  over the ceiling and allowlisted: {len(backlog)}")
    if backlog:
        print(f"  backlog lines remaining: {sum(count for count, _ in backlog)}")

    for name, entries in (("over the ceiling, not allowlisted", violations),
                          ("allowlisted but grown past its recorded count", grown)):
        if entries:
            print(f"\n{name}:")
            for entry in entries:
                print("  " + entry)

    if "--list" in argv or "-l" in argv:
        print(f"\nwarning band ({len(warning)} files):")
        for count, relative in warning:
            print(f"{count:6}  {relative}")

    if violations or grown:
        print("\nFAILED: see docs/file-split-plan.md (the backlog is ALLOWLIST in line_budget.py).")
        return 1

    stale: List[str] = [name for name in ALLOWLIST if not os.path.isfile(os.path.join(root, name))]
    if stale:
        print("\nallowlist entries for files that no longer exist (remove them):")
        for name in sorted(stale):
            print("  " + name)

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
