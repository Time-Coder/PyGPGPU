"""The synthesized device helpers live in one shared header per cache.

The helpers the translator *synthesizes* -- the vector-operation family, the floor
helpers -- have no python module that owns them, so they used to be written into
every module's generated file that needed them. A kernel whose unit includes
another module's file (a helper of that module did `float3` arithmetic too) then
read the same `vec_sub_float3_float3` definition twice, which the compiler refuses
with "has already been defined" -- found by a real multi-module program whose scene
module calls helpers from three other modules. They now merge into one generated,
guarded header per cache (`include/device_helpers.h`), and the per-module copies
are nothing but a relative include.

A second fact is pinned here: a helper that indexes a multi-dimensional array it
was *handed* takes its extent under the caller's own name, because the launch
resolves shape values against the caller's parameter list -- a `self.grid`
attribute bound into a helper used to leave the helper's local name in
`shapes_used`, which the launch looked up among the kernel's parameters and
raised KeyError on.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import helper_arithmetic_module
from pygpgpu import kernel
from pygpgpu import numpy as gnp
from pygpgpu.kernel import source_paths
from pygf import float3


# ---------------------------------------------------------------------------
# The collision: two modules, one set of synthesized helpers
# ---------------------------------------------------------------------------


@kernel
def shades_in_both_modules(out: np.ndarray):
    """The kernel's own body does float3 arithmetic AND calls the other module's."""
    for i in range(out.shape[0]):
        base = float3(1.0, 2.0, 3.0)
        tint = float3(4.0, 5.0, 6.0)
        mine = (base + (tint - base) * 0.25) % float3(8.0, 8.0, 8.0)
        theirs = helper_arithmetic_module.shade(base, tint, 0.75)
        out[i] = mine.x + theirs.y


def test_the_same_helper_family_in_two_modules_compiles(runner):
    """Before the shared header this unit failed with "already been defined"."""
    out = np.zeros(1, dtype=np.float32)
    runner(shades_in_both_modules, out)
    assert float(out[0]) == pytest.approx(1.75 + 2.125)


def _cache_paths(backend: str) -> "dict[str, str]":
    """The generated files one run of `shades_in_both_modules` produces."""
    source_file: str = __file__
    root: str = source_paths.cache_root(source_file, backend)
    return {"own": source_paths.source_path(source_file, source_file, backend),
            "other": source_paths.source_path(source_file,
                                              helper_arithmetic_module.__file__, backend),
            "header": os.path.join(root, source_paths.INCLUDE_FOLDER,
                                   source_paths.HELPERS_HEADER_NAME)}


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


DEFINITION_HEAD: str = "float3 vec_floor_mod_float3_float3(float3 a, float3 b)\n{"


def test_the_definitions_live_in_the_shared_header_only(gpu_backend):
    """Each module file carries the include and a prototype; the definition is shared.

    The floor wrapper is the probe because it is synthesized on *every* backend
    (OpenCL has native vector arithmetic, so only the floored `%` forces a helper).
    """
    qualified: str = ("__device__ " if gpu_backend in ("cuda", "hip") else "") \
        + "static inline " + DEFINITION_HEAD
    out = np.zeros(1, dtype=np.float32)
    shades_in_both_modules[gpu_backend](out)
    paths = _cache_paths(gpu_backend)
    header_text: str = _read(paths["header"])
    # The definition is in the header, under its own include guard.
    assert "#ifndef __device_helpers_" in header_text
    assert qualified in header_text
    for path in (paths["own"], paths["other"]):
        text: str = _read(path)
        assert DEFINITION_HEAD not in text
        assert "vec_floor_mod_float3_float3(float3 a, float3 b);" in text
        assert "device_helpers.h" in text


# ---------------------------------------------------------------------------
# A helper's array extents are named after the caller's binding
# ---------------------------------------------------------------------------


def grid_total(grid) -> float:
    """A device helper that reads a 2D array it was handed: its width is an extent."""
    total: float = 0.0
    for r in range(5):
        for c in range(6):
            total += float(grid[r, c])
    return total


class Grids:

    def __init__(self) -> None:
        self.grid = gnp.array(np.arange(30, dtype=np.float32).reshape(5, 6))
        self.other = gnp.array(np.arange(30, dtype=np.float32).reshape(5, 6) * 2.0)

    @kernel
    def from_parameter(self, out, grid):
        out[0] = grid_total(grid)

    @kernel
    def from_attribute(self, out):
        out[0] = grid_total(self.grid)

    @kernel
    def from_renamed_parameter(self, out, cells):
        """The helper's parameter is `grid`; the kernel's is `cells`."""
        out[0] = grid_total(cells)

    @kernel
    def from_alias(self, out):
        """`src = self.grid` is a pointer local: the extent follows the array."""
        src = self.grid
        out[0] = grid_total(src)

    @kernel
    def from_two_arrays(self, out):
        """Two call sites, one helper: each site's extent reaches its own call."""
        first: float = grid_total(self.grid)
        second: float = grid_total(self.other)
        out[0] = first + second


GRID_EXPECTED: float = float(np.arange(30, dtype=np.float32).reshape(5, 6).sum())


def test_a_two_dimensional_helper_array_from_a_parameter(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(Grids().from_parameter, out,
           np.arange(30, dtype=np.float32).reshape(5, 6))
    assert float(out[0]) == pytest.approx(GRID_EXPECTED)


def test_a_two_dimensional_helper_array_from_a_self_attribute(runner):
    """The extent's launch-side name is the attribute slot, not the helper's local."""
    out = np.zeros(1, dtype=np.float32)
    runner(Grids().from_attribute, out)
    assert float(out[0]) == pytest.approx(GRID_EXPECTED)


def test_a_two_dimensional_helper_array_from_a_renamed_parameter(runner):
    """The kernel's parameter name need not match the helper's."""
    out = np.zeros(1, dtype=np.float32)
    runner(Grids().from_renamed_parameter, out,
           np.arange(30, dtype=np.float32).reshape(5, 6))
    assert float(out[0]) == pytest.approx(GRID_EXPECTED)


def test_a_two_dimensional_helper_array_from_a_pointer_alias(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(Grids().from_alias, out)
    assert float(out[0]) == pytest.approx(GRID_EXPECTED)


def test_two_call_sites_bind_their_own_extents(runner):
    """One shared helper entry, two arrays: neither site answers the other's extent."""
    out = np.zeros(1, dtype=np.float32)
    runner(Grids().from_two_arrays, out)
    assert float(out[0]) == pytest.approx(GRID_EXPECTED * 3.0)
