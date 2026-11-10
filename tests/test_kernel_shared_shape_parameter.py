"""A `local_array_per_group` shape read from a kernel parameter, and what it costs.

The buffer's size is a literal in the generated source (`__shared__ T name[N]` is a
fixed-size declaration), so a parameter-sized buffer **specializes the kernel**: the value is
baked in at translation time, and two calls with different values compile two kernels whose
definitions coexist in the generated file. The value is part of the cache key and of the C
name's hash, so a second call reuses its own binary instead of launching a grid against a
buffer of the wrong size.

A module constant bakes the same literal every launch, which is the rule `np.ndindex` bounds
already followed; a value the launch cannot know before compiling -- a host-region binding,
an expression -- is refused, with a message that says why and what does work.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import barrier, blockIdx, kernel, local_array_per_group


@kernel(domain="out")
def parameterized(out: np.ndarray, group_size: int):
    bins = local_array_per_group(group_size, np.float32)
    lane = 0
    bins[blockIdx][lane] = float(group_size)
    barrier()
    out[0] = bins[blockIdx][lane]


@kernel(domain="out")
def literal(out: np.ndarray):
    bins = local_array_per_group(64, np.float32)
    bins[blockIdx][0] = 64.0
    barrier()
    out[0] = bins[blockIdx][0]


def test_each_value_compiles_and_runs_its_own_kernel(gpu_backend: str) -> None:
    for size in (32, 128):
        out = np.zeros(1, dtype=np.float32)
        parameterized[gpu_backend][(1, size)](out, size)
        assert out[0] == np.float32(size), (size, out[0])


def test_the_literal_path_is_unchanged(gpu_backend: str) -> None:
    out = np.zeros(1, dtype=np.float32)
    literal[gpu_backend][(1, 64)](out)
    assert out[0] == np.float32(64.0)


def test_a_computed_shape_is_refused_with_its_reason(gpu_backend: str) -> None:
    @kernel(domain="out")
    def computed(out: np.ndarray, width: int):
        size = width * 2
        bins = local_array_per_group(size, np.float32)
        bins[blockIdx][0] = 0.0
        barrier()
        out[0] = bins[blockIdx][0]

    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        computed[gpu_backend][(1, 32)](out, 32)
    message = str(caught.value)
    assert "specializes" in message, message
    assert "int parameter" in message, message


def test_a_non_whole_size_names_the_value_this_launch_carried(gpu_backend: str) -> None:
    """The name resolved and the launch ran -- the *value* is the problem, so say that."""

    @kernel(domain="out")
    def fractional(out: np.ndarray, group_size: float):
        bins = local_array_per_group(group_size, np.float32)
        bins[blockIdx][0] = 0.0
        barrier()
        out[0] = bins[blockIdx][0]

    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        fractional[gpu_backend][(1, 8)](out, 2.5)
    message = str(caught.value)
    assert "whole number" in message, message
    assert "2.5" in message, message


def test_a_zero_size_names_the_value_this_launch_carried(gpu_backend: str) -> None:
    @kernel(domain="out")
    def empty(out: np.ndarray, group_size: int):
        bins = local_array_per_group(group_size, np.float32)
        bins[blockIdx][0] = 0.0
        barrier()
        out[0] = bins[blockIdx][0]

    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        empty[gpu_backend][(1, 8)](out, 0)
    message = str(caught.value)
    assert "positive" in message, message
    assert "carried 0" in message, message


def test_a_module_constant_bakes_the_same_literal_every_launch(gpu_backend: str) -> None:
    module_size = 96

    @kernel(domain="out")
    def from_constant(out: np.ndarray):
        bins = local_array_per_group(module_size, np.float32)
        bins[blockIdx][0] = float(module_size)
        barrier()
        out[0] = bins[blockIdx][0]

    out = np.zeros(1, dtype=np.float32)
    from_constant[gpu_backend][(1, module_size)](out)
    assert out[0] == np.float32(module_size)
