"""The generated-source cache compares content, not a clock, and names what it declares.

Two failures this pins, both of which showed up as "the cache did not invalidate":

**Staleness by timestamp.** `append_source` reused a generated file when it was *newer* than
the python source it came from. That assumes a file's timestamp moves forward when its text
changes, and git breaks the assumption: `git checkout` and `git stash pop` restore a file
*and* set its mtime back. A generated file written from newer text then kept looking fresh,
so the old declaration of a kernel stayed beside the new one and the compiler reported
"more than one instance of overloaded function". The file now records a digest of the python
text and a mismatch invalidates it whichever way the timestamps point.

**A name that did not cover what it declares.** The C name's hash was built from the
*argument types* the caller passed, which do not include the extent parameters derived from
the body, the outer-scope values or the shared buffers. A body whose derived signature
changed therefore kept its old name. The rendered parameter list now joins the same hash, so
the name changes with the signature -- and still reads `name_<hash>`, one suffix, not two.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from pygpgpu import (barrier, blockIdx, get_global_id, kernel, zeros_per_group)
from pygpgpu.kernel import source_paths


# ---------------------------------------------------------------------------
# The digest decides staleness, not the timestamps
# ---------------------------------------------------------------------------


def test_a_generated_file_records_the_digest_of_its_source():
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cu")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        pathlib.Path(source_python).write_text("# v1\n", encoding="utf-8")
        digest: str = source_paths.python_source_digest(source_python)
        source_paths.append_source(
            path, [], ["__global__ void k_aa(float* x)\n{\n    x[0] = 1;\n}"],
            source_python, source_digest=digest)
        recorded: str = source_paths._stored_digest(path)
        assert recorded == digest, (recorded, digest)
        # The digest lives beside the source, never in it: the generated text is a
        # translation unit whose blank lines `_split_source` depends on.
        text: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert digest not in text, text


def test_a_changed_source_invalidates_even_when_its_timestamp_is_older():
    """The git case: same stale-clock shape, but the text is what decides."""
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cu")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        source: pathlib.Path = pathlib.Path(source_python)
        source.write_text("# v1\n", encoding="utf-8")
        first_digest: str = source_paths.python_source_digest(source_python)
        source_paths.append_source(
            path, [], ["__global__ void k_aa(float* x)\n{\n    x[0] = 1;\n}"],
            source_python, source_digest=first_digest)

        # Rewrite the text and set its mtime *back*, exactly as `git checkout` does.
        source.write_text("# v2 with a different body\n", encoding="utf-8")
        stamp: float = pathlib.Path(path).stat().st_mtime - 30
        import os
        os.utime(source_python, (stamp, stamp))
        second_digest: str = source_paths.python_source_digest(source_python)
        assert second_digest != first_digest

        source_paths.append_source(
            path, [], ["__global__ void k_bb(float* y)\n{\n    y[0] = 2;\n}"],
            source_python, source_digest=second_digest)
        text: str = pathlib.Path(path).read_text(encoding="utf-8")
        # The file describes the new text, so it was started over: the old kernel is gone.
        assert "k_aa" not in text, text
        assert "k_bb" in text, text


def test_an_unchanged_source_keeps_the_file():
    """The digest must not throw work away when nothing changed."""
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cu")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        pathlib.Path(source_python).write_text("# v1\n", encoding="utf-8")
        digest: str = source_paths.python_source_digest(source_python)
        source_paths.append_source(
            path, [], ["__global__ void k_aa(float* x)\n{\n    x[0] = 1;\n}"],
            source_python, source_digest=digest)
        source_paths.append_source(
            path, [], ["__global__ void k_bb(float* y)\n{\n    y[0] = 2;\n}"],
            source_python, source_digest=digest)
        text: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert "k_aa" in text, text
        assert "k_bb" in text, text


# ---------------------------------------------------------------------------
# The name covers the declaration
# ---------------------------------------------------------------------------


def _kernel_names(backend: str) -> list:
    import glob

    extension: str = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[backend]
    root: str = source_paths.cache_root(__file__, backend)
    names: list = []
    for path in sorted(glob.glob(f"{root}/*.{extension}")):
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if "__global__" in line or "__kernel" in line:
                head: str = line.split("void ")[-1] if "void " in line else line
                names.append(head.split("(")[0].strip())
    return names


@kernel
def named_by_its_declaration(x: np.ndarray, out: np.ndarray, scale: np.float32):
    """`scale` is an argument, so it is part of the signature -- and of the name."""
    for i in range(out.shape[0]):
        out[i] = x[i] * scale


def test_the_kernel_name_keeps_one_hash_suffix(gpu_backend):
    """A name still reads `name_<hash>`: the declaration joins the hash, it is not a second one."""
    import re

    named_by_its_declaration[gpu_backend](np.ones(8, dtype=np.float32),
                                          np.zeros(8, dtype=np.float32), np.float32(2.0))
    names: list = _kernel_names(gpu_backend)
    assert names, "no generated kernel found"
    target: str = [name for name in names if name.startswith("named_by_its_declaration")]
    assert target, names
    for name in target:
        assert re.fullmatch(r"named_by_its_declaration_[0-9a-f]{8}", name), name


@kernel(domain="out")
def declares_shared_memory(out: np.ndarray):
    """A gid-style body that also declares a fixed-size shared buffer.

    Defined at module scope so one source compiles once: a kernel defined inside a test
    function is a *new* source on every run, and OpenCL runs out of resources when a suite
    recompiles it repeatedly.
    """
    lane = get_global_id(0)
    buf = zeros_per_group(64, np.int32)
    buf[blockIdx][lane] = 7
    barrier()
    out[lane] = float(buf[blockIdx][lane])


def test_a_derived_signature_change_changes_the_name(gpu_backend):
    """A kernel whose signature carries derived parameters still gets a name of its own.

    The signature here is not just the caller's argument types: it also declares the extent
    parameters derived from the body (`_out_shape0`). The name covers that declaration, which
    is what keeps a body whose derived signature changed from reusing its old name.
    """
    declares_shared_memory[gpu_backend][(1, 64)](np.zeros(64, dtype=np.float32))
    names: list = [n for n in _kernel_names(gpu_backend)
                   if n.startswith("declares_shared_memory")]
    assert names, "no generated kernel found"
    assert all(name.count("_") >= 1 for name in names), names
