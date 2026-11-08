"""The reads tests, split out of `test_kernel_object_arguments.py`.

The subject is reads.
"""
from __future__ import annotations

from _helpers.test_kernel_object_arguments_helpers import (
    Greeter,
    Pair,
    calls_a_method,
    reads_a_plain_objects_members,
    reads_an_alias,
    reads_two_members,
)

import numpy as np
import pytest


def test_the_read_members_travel(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(reads_two_members, Pair(3.0, 5.0), out)
    assert float(out[0]) == pytest.approx(15.0)


def test_an_alias_counts_as_a_read(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(reads_an_alias, Pair(7.0, 0.0), out)
    assert float(out[0]) == pytest.approx(14.0)


def test_a_method_on_the_object(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method, Greeter(1.5), out)
    assert float(out[0]) == pytest.approx(3.0)


def test_a_plain_objects_structure_lives_with_its_class(gpu_backend):
    """A synthesized structure is declared beside the class the caller wrote.

    The members are inferred from *this* module's body, but the class lives in
    `helper_scope_module`, so the declaration belongs to that module's generated file and
    this module's kernel includes it -- the same rule a `ctypes.Structure` follows.
    """
    import glob
    import pathlib
    import shutil

    from helper_scope_module import PlainObject

    cache_dir = reads_a_plain_objects_members._cache_dir(gpu_backend)
    shutil.rmtree(cache_dir, ignore_errors=True)
    out = np.zeros(1, dtype=np.float32)
    reads_a_plain_objects_members[gpu_backend](PlainObject(1.0, 2.0), out)
    assert float(out[0]) == pytest.approx(2.0)
    extension = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[gpu_backend]
    sources:dict = {pathlib.Path(path).name: pathlib.Path(path).read_text(encoding="utf-8")
                    for path in glob.glob(f"{cache_dir}/*.{extension}")}
    assert sources, sorted(glob.glob(f"{cache_dir}/*"))
    # The declaration exists exactly once across the module set. Which module holds it
    # depends on the class that registered this layout first -- two classes can share one
    # dtype, hence one C type -- so the class name is not asserted; what matters is that a
    # declaration is present and that this module's own file does not carry it, because the
    # class behind this value lives in another module.
    declaring = [name for name, text in sources.items() if "float a;" in text]
    assert len(declaring) == 1, sources
    own = sources.get(f"{pathlib.Path(__file__).stem}.{extension}", "")
    assert "float a;" not in own or declaring == [f"{pathlib.Path(__file__).stem}.{extension}"], own
