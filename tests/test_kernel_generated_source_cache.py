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
import subprocess
import sys
import tempfile

import numpy as np

from pygpgpu import (barrier, blockIdx, get_global_id, kernel, local_array_per_group)
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
# A definition whose body changed is replaced, not skipped
# ---------------------------------------------------------------------------


def test_a_changed_body_replaces_the_definition_the_file_holds():
    """The caller's text did not move, but the code this package generates for it did.

    A definition's C name hashes its *signature*, so a body change keeps the header line --
    which the old merge compared and then skipped the new render for. The file is what the
    compiler reads, so the old body kept running after the package changed (the python digest
    cannot catch it: the caller's `.py` did not move). The replacement is in place, because
    the order the definitions were written in is the file's shape.
    """
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cu")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        pathlib.Path(source_python).write_text("# v1\n", encoding="utf-8")
        digest: str = source_paths.python_source_digest(source_python)
        before: str = "__global__ void k_aa(float* x)\n{\n    x[0] = 1.0f;\n}"
        after: str = "__global__ void k_aa(float* x)\n{\n    x[0] = 2.0f;\n}"
        sibling: str = "__global__ void k_bb(float* y)\n{\n    y[0] = 3.0f;\n}"
        source_paths.append_source(path, [], [before], source_python, source_digest=digest)
        source_paths.append_source(path, [], [sibling], source_python, source_digest=digest)
        # Same python text, same digest, same header line: only the generated body differs.
        source_paths.append_source(path, [], [after], source_python, source_digest=digest)
        text: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert "x[0] = 2.0f;" in text, text
        assert "x[0] = 1.0f;" not in text, text
        # The definition that did not change is untouched...
        assert "y[0] = 3.0f;" in text, text
        # ...and the refreshed one stays where it was: a definition must not move past one
        # it has to precede.
        assert text.index("k_aa") < text.index("k_bb"), text


def test_a_re_render_of_the_same_definition_is_a_no_op():
    """Idempotence is what keeps the cache from churning on every kernel call."""
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cu")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        pathlib.Path(source_python).write_text("# v1\n", encoding="utf-8")
        digest: str = source_paths.python_source_digest(source_python)
        body: str = "__global__ void k_aa(float* x)\n{\n    x[0] = 1.0f;\n}"
        source_paths.append_source(path, [], [body], source_python, source_digest=digest)
        written: str = pathlib.Path(path).read_text(encoding="utf-8")
        source_paths.append_source(path, [], [body], source_python, source_digest=digest)
        assert pathlib.Path(path).read_text(encoding="utf-8") == written


def test_a_body_holding_a_blank_line_is_not_duplicated():
    """The file is split on blank lines, so a definition's tail is a piece of its own.

    A generated magic loop emits its bounds guard, then a blank line, then the rest of the
    body -- so the file's pieces do not line up with the definitions a render produces (one
    render is one definition, blank lines and all). Comparing a whole rendered definition
    against a single *piece* replaced the head and appended the tail, and every render added
    another copy of that tail until the compiler refused the file.
    """
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cl")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        pathlib.Path(source_python).write_text("# v1\n", encoding="utf-8")
        digest: str = source_paths.python_source_digest(source_python)
        body: str = ("__kernel void k_aa(__global float* out, const int _out_shape0)\n"
                     "{\n"
                     "    const int i = get_global_id(0);\n"
                     "    if (i >= _out_shape0)\n"
                     "    {\n"
                     "        return;\n"
                     "    }\n"
                     "\n"
                     "    out[i] = 1.0f;\n"
                     "}\n")
        source_paths.append_source(path, [], [body], source_python, source_digest=digest)
        once: str = pathlib.Path(path).read_text(encoding="utf-8")
        source_paths.append_source(path, [], [body], source_python, source_digest=digest)
        twice: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert once == twice, twice
        assert twice.count("out[i] = 1.0f;") == 1, twice
        # And a changed body still replaces the whole definition, tail included.
        changed: str = body.replace("1.0f", "2.0f")
        source_paths.append_source(path, [], [changed], source_python, source_digest=digest)
        after: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert after.count("out[i] = 2.0f;") == 1, after
        assert "1.0f" not in after, after


def test_a_body_piece_that_looks_like_a_header_does_not_split_the_definition():
    """A body's middle may start with `while (((b < n)))` -- a header line's shape.

    The file is split on blank lines, so a generated body that holds them becomes several
    pieces. Reading a piece that starts with `while (...)` or `if (...)` as a definition of
    its own gave the body's middle an identity: the definition was then two definitions, and
    refreshing it replaced the first half while the second was appended again -- a file the
    compiler refused with "extraneous closing brace".
    """
    with tempfile.TemporaryDirectory() as directory:
        path: str = str(pathlib.Path(directory) / "mod.cl")
        source_python: str = str(pathlib.Path(directory) / "mod.py")
        pathlib.Path(source_python).write_text("# v1\n", encoding="utf-8")
        digest: str = source_paths.python_source_digest(source_python)
        body: str = ("__kernel void k_aa(__global int* out, const int n)\n"
                     "{\n"
                     "    const int i = get_global_id(0);\n"
                     "    if (i >= n)\n"
                     "    {\n"
                     "        return;\n"
                     "    }\n"
                     "\n"
                     "    int b = i;\n"
                     "\n"
                     "    while (((b < n)))\n"
                     "    {\n"
                     "        out[b] = 1;\n"
                     "        b += 1;\n"
                     "    }\n"
                     "\n"
                     "    if (i == 0)\n"
                     "    {\n"
                     "        out[0] = 2;\n"
                     "    }\n"
                     "}\n")
        source_paths.append_source(path, [], [body], source_python, source_digest=digest)
        once: str = pathlib.Path(path).read_text(encoding="utf-8")
        source_paths.append_source(path, [], [body], source_python, source_digest=digest)
        twice: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert once == twice, twice
        assert twice.count("out[0] = 2;") == 1, twice
        assert twice.count("out[b] = 1;") == 1, twice
        # A changed body still replaces the whole definition, every piece of it.
        changed: str = body.replace("out[0] = 2;", "out[0] = 3;")
        source_paths.append_source(path, [], [changed], source_python, source_digest=digest)
        after: str = pathlib.Path(path).read_text(encoding="utf-8")
        assert after.count("out[0] = 3;") == 1, after
        assert after.count("out[b] = 1;") == 1, after
        assert after.count("}") == body.count("}"), after


def test_merge_source_refreshes_a_changed_definition():
    """`write_source`'s merge applies the same rule: a known definition is compared."""
    existing: str = ("#ifndef __mod__\n#define __mod__\n\n"
                     "__global__ void k_aa(float* x)\n{\n    x[0] = 1.0f;\n}\n\n#endif\n")
    before: str = "__global__ void k_aa(float* x)\n{\n    x[0] = 1.0f;\n}\n"
    after: str = "__global__ void k_aa(float* x)\n{\n    x[0] = 2.0f;\n}\n"
    merged: str = source_paths.merge_source(existing, before)
    assert "x[0] = 1.0f;" in merged
    merged = source_paths.merge_source(merged, after)
    assert "x[0] = 2.0f;" in merged
    assert "x[0] = 1.0f;" not in merged
    # Nothing changed -> the text is handed back untouched.
    assert source_paths.merge_source(merged, after) == merged


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
    buf = local_array_per_group(64, np.int32)
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


# ---------------------------------------------------------------------------
# A structure answers with an identity in both spellings
# ---------------------------------------------------------------------------


def test_a_structure_declaration_has_an_identity_in_both_spellings():
    """A structure is identified by the type it declares, on one line or on several.

    OpenCL writes `typedef struct { ... } Name;`, and the generator emits it both ways. The
    multi-line one is identified by the `} Name;` it closes with; the one-line one has no
    such line, so it answered None -- and a definition with no identity can only be
    *appended*, never matched, which is how a changed structure ends up in a file beside the
    old one.
    """
    one_line: str = "typedef struct { float re; float im; } c64;"
    assert source_paths._definition_identity(one_line) == one_line
    assert source_paths._definition_identity(
        "typedef struct {\n    float re;\n    float im;\n} c64;") == "} c64;"
    # A body's middle is not a declaration, however much it looks like one.
    assert source_paths._definition_identity("if (x < 768)\n{\n    y = 1;\n}") is None


# ---------------------------------------------------------------------------
# A stale file on disk is refreshed by the next launch
# ---------------------------------------------------------------------------


# One kernel whose generated body names a complex constructor, so the file on disk says which
# width the current package generates. It runs in a fresh interpreter, which is what an
# upgrade looks like: the compiled binary cache and the in-process one are both empty, and the
# only thing left of the previous version is the generated source beside the caller's module.
_STALE_PROBE: str = '''
import sys

import numpy as np

from pygpgpu import kernel


@kernel
def writes_a_complex_literal(out):
    for i in range(out.shape[0]):
        out[i] = 3 + 4j


out = np.zeros(2, np.complex64)
writes_a_complex_literal[sys.argv[1]](out)
assert out[0] == np.complex64(3 + 4j), out
'''


def _run_probe(probe: pathlib.Path, backend: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(probe), backend], cwd=str(probe.parent),
                          capture_output=True, text=True)


def _generated_file(directory: pathlib.Path, backend: str) -> pathlib.Path:
    """The one generated source this probe's module owns."""
    extension: str = source_paths.EXTENSIONS[backend]
    found: list = sorted(directory.rglob(f"*.{extension}"))
    assert found, f"no generated .{extension} under {directory}"
    return found[0]


def test_a_stale_generated_file_is_refreshed_by_the_next_launch(tmp_path, gpu_backend):
    """The file is what the compiler reads, so a body this package no longer writes must go.

    The version that skipped a definition whose header line it already knew left the file
    describing code the package had stopped generating -- and nothing moved the python-source
    digest, because the caller's `.py` was untouched. An upgrade is exactly that: a fresh
    interpreter, no compiled binary, and yesterday's generated source sitting on disk. This
    drives it end to end with two subprocesses.
    """
    probe: pathlib.Path = tmp_path / "probe_stale_kernel.py"
    probe.write_text(_STALE_PROBE, encoding="utf-8")

    first = _run_probe(probe, gpu_backend)
    assert first.returncode == 0, first.stderr
    generated: pathlib.Path = _generated_file(tmp_path, gpu_backend)
    fresh: str = generated.read_text(encoding="utf-8")
    assert "c64_make" in fresh, fresh

    # Yesterday's body: the same header line, a different constructor -- which is what the
    # package generated before the python complex width followed `float=`.
    stale: str = fresh.replace("c64_make", "c128_make")
    assert stale != fresh
    generated.write_text(stale, encoding="utf-8")

    second = _run_probe(probe, gpu_backend)
    assert second.returncode == 0, second.stderr
    repaired: str = generated.read_text(encoding="utf-8")
    assert "c128_make" not in repaired, repaired
    assert "c64_make" in repaired, repaired
