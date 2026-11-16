"""A cache tree that survives being moved, and a packaged application with no source.

Both tests here pin the same underlying rule: nothing the cache records may name the
directory it was written in. An absolute path made the generated tree unusable anywhere but
where it was built, so a build-time warm cache could not be shipped inside a packaged
application and the first launch in the packaged process had to translate with no `.py` on
disk -- which `inspect.getsource` cannot do.

Three separate records had to stop naming the build machine:

* ``related_files`` -- the C preprocessor's list of files a parse depended on;
* the parse cache's own key (``md5_of``), which hashed the generated file's absolute path;
* the include guard (``guard_macro``), which hashed it too and therefore made the *text*
  of a moved file differ from the text its digest described.

What replaces them is content: a file's bytes, or its path *inside* the cache tree. That is
also why a moved tree validates at all -- copying rewrites every timestamp, so the older
"newer than the meta file" comparison could invert on a tree that never changed.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from pygpgpu.cparser.related_files import (
    absolute_related_files,
    related_digests,
    related_files_are_current,
    relative_related_files,
)
from pygpgpu.kernel import source_paths


# ---------------------------------------------------------------------------
# The recorded names travel with the tree
# ---------------------------------------------------------------------------


def test_related_files_are_recorded_relative_to_the_cache_folder():
    cache_folder = os.path.join(os.sep, "tmp", "site", "__cucache__")
    generated = os.path.join(cache_folder, "pykernels", "scene.cu")
    recorded = relative_related_files({generated}, os.path.join(cache_folder, "pykernels"))
    assert recorded == {"scene.cu"}, recorded
    # And they resolve back to the same file, so the freshness check still works.
    resolved = absolute_related_files(recorded, os.path.join(cache_folder, "pykernels"))
    assert resolved == {os.path.normpath(generated)}, resolved


def test_a_file_outside_the_cache_stays_absolute(tmp_path: pathlib.Path):
    """An installed package's header is genuinely elsewhere, and must keep working."""
    cache_folder = str(tmp_path / "site" / "pykernels")
    outside = tmp_path / "elsewhere" / "vendor.h"
    recorded = relative_related_files({str(outside)}, cache_folder)
    # Not expressible relative to the cache folder, so it keeps its absolute spelling --
    # with forward slashes, so the record reads the same on every platform.
    assert list(recorded) == [str(outside).replace("\\", "/")], recorded
    assert absolute_related_files(recorded, cache_folder) == {os.path.normpath(str(outside))}


def test_a_moved_tree_still_validates_by_content(tmp_path: pathlib.Path):
    """The reason timestamps cannot decide: a copy rewrites every stamp."""
    original = tmp_path / "a"
    moved = tmp_path / "b"
    original.mkdir()
    (original / "scene.cu").write_text("#ifndef X\n#endif\n", encoding="utf-8")

    cache_folder = str(original)
    recorded = relative_related_files({str(original / "scene.cu")}, cache_folder)
    digests = related_digests({str(original / "scene.cu")}, cache_folder)

    shutil.copytree(original, moved)
    # A copy makes the meta file *older* than the source it describes; content is unchanged.
    meta_mtime = os.path.getmtime(original / "scene.cu") - 60
    assert related_files_are_current(recorded, str(moved), digests, meta_mtime)
    # Changing the content is still caught.
    (moved / "scene.cu").write_text("#ifndef Y\n#endif\n", encoding="utf-8")
    assert not related_files_are_current(recorded, str(moved), digests, meta_mtime)


def test_a_missing_file_invalidates_the_parse(tmp_path: pathlib.Path):
    cache_folder = str(tmp_path)
    (tmp_path / "scene.cu").write_text("void k();\n", encoding="utf-8")
    recorded = relative_related_files({str(tmp_path / "scene.cu")}, cache_folder)
    digests = related_digests({str(tmp_path / "scene.cu")}, cache_folder)
    assert related_files_are_current(recorded, cache_folder, digests, 0)
    (tmp_path / "scene.cu").unlink()
    assert not related_files_are_current(recorded, cache_folder, digests, 0)


# ---------------------------------------------------------------------------
# The include guard and the parse key are path-independent
# ---------------------------------------------------------------------------


def test_the_include_guard_ignores_where_the_tree_sits():
    first = os.path.join(os.sep, "build", "a", "__cucache__", "pykernels", "scene.cu")
    second = os.path.join(os.sep, "dist", "b", "__cucache__", "pykernels", "scene.cu")
    assert source_paths.guard_macro(first) == source_paths.guard_macro(second)
    # Two different modules in one tree still get different guards.
    other = os.path.join(os.sep, "build", "a", "__cucache__", "pykernels", "other.cu")
    assert source_paths.guard_macro(first) != source_paths.guard_macro(other)


def test_the_parse_key_ignores_where_the_tree_sits(tmp_path: pathlib.Path):
    """`md5_of` decides the meta file's name, so it has to match after a move."""
    from pygpgpu.cuda.oop.program_parser import ProgramParser

    first = tmp_path / "a" / "app.cu"
    second = tmp_path / "b" / "app.cu"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text("__global__ void k(float* x)\n{\n    x[0] = 1;\n}\n", encoding="utf-8")
    assert ProgramParser.md5_of([], {}, None, file_name=str(first)) == \
        ProgramParser.md5_of([], {}, None, file_name=str(second))
    # Different content still changes the key: two kernels share one generated file.
    second.write_text("__global__ void j(float* x)\n{\n    x[0] = 2;\n}\n", encoding="utf-8")
    assert ProgramParser.md5_of([], {}, None, file_name=str(first)) != \
        ProgramParser.md5_of([], {}, None, file_name=str(second))


# ---------------------------------------------------------------------------
# The source store: what a frozen process reads instead of a `.py`
# ---------------------------------------------------------------------------


def test_a_function_is_cut_out_of_its_module_text():
    text = (
        "import numpy as np\n"
        "\n"
        "@kernel\n"
        "def scale(a, out, factor):\n"
        "    for i in range(out.shape[0]):\n"
        "        out[i] = a[i] * factor\n"
        "\n"
        "def helper(x):\n"
        "    return x\n"
    )
    found = source_paths.function_source(text, "scale")
    assert found is not None
    lines, start = found
    joined = "".join(lines)
    # The decorator comes along, as `inspect.getsourcelines` would give it.
    assert joined.startswith("@kernel\ndef scale"), joined
    assert "helper" not in joined, joined
    assert start == 3, start


def test_a_qualified_name_finds_a_method_and_not_its_namesake():
    text = (
        "class First:\n"
        "    def run(self, out):\n"
        "        out[0] = 1\n"
        "\n"
        "class Second:\n"
        "    def run(self, out):\n"
        "        out[0] = 2\n"
    )
    found = source_paths.function_source(text, "Second.run")
    assert found is not None
    joined = "".join(found[0])
    assert "out[0] = 2" in joined, joined
    assert "out[0] = 1" not in joined, joined


def test_a_source_file_with_a_byte_order_mark_is_still_parsed(tmp_path: pathlib.Path):
    """A module saved with a BOM is legal python; `ast.parse` rejects the mark itself."""
    path = tmp_path / "mod.py"
    path.write_text("\ufeffdef k(out):\n    out[0] = 1\n", encoding="utf-8")
    from pygpgpu.kernel.source_paths.source_store import _read_text

    text = _read_text(str(path))
    assert text is not None and not text.startswith("\ufeff")
    assert source_paths.function_source(text, "k") is not None


def test_the_store_round_trips_a_module(tmp_path: pathlib.Path, monkeypatch):
    """`store_module_source` then `load_module_source` gives the same text back."""
    monkeypatch.setenv("PYGPGPU_CACHE_DIR", str(tmp_path / "cache"))
    kernel_file = str(tmp_path / "userkern.py")
    text = "from pygpgpu import kernel\n\n@kernel\ndef k(out):\n    out[0] = 1\n"
    written = source_paths.store_module_source(kernel_file, "cuda", "userkern", text)
    assert written and os.path.isfile(written), written
    assert source_paths.load_module_source(kernel_file, "userkern") == text
    # Writing the same text again is a no-op, not a second copy.
    assert source_paths.store_module_source(kernel_file, "cuda", "userkern", text) is None


def test_the_store_is_found_by_module_name_across_backends(tmp_path: pathlib.Path,
                                                           monkeypatch):
    """A tree that ships one backend's folder still serves every backend's launch."""
    monkeypatch.setenv("PYGPGPU_CACHE_DIR", str(tmp_path / "cache"))
    kernel_file = str(tmp_path / "userkern.py")
    text = "def k(out):\n    out[0] = 1\n"
    source_paths.store_module_source(kernel_file, "opencl", "userkern", text)
    # Read back with no backend named: the store searches the whole tree.
    assert source_paths.load_module_source(kernel_file, "userkern") == text


def test_an_unknown_module_is_simply_absent(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("PYGPGPU_CACHE_DIR", str(tmp_path / "cache"))
    assert source_paths.load_module_source(str(tmp_path / "nothing.py"), "nothing") is None


# ---------------------------------------------------------------------------
# The cache folder itself
# ---------------------------------------------------------------------------


def test_the_cache_dir_variable_wins(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setenv("PYGPGPU_CACHE_DIR", str(tmp_path / "choice"))
    root = source_paths.cache_root(str(tmp_path / "app.py"), "cuda")
    assert root == os.path.join(str(tmp_path / "choice"), "__cucache__", "pykernels")


def test_the_default_cache_sits_beside_the_python_file(tmp_path: pathlib.Path,
                                                       monkeypatch):
    monkeypatch.delenv("PYGPGPU_CACHE_DIR", raising=False)
    root = source_paths.cache_root(str(tmp_path / "app.py"), "cuda")
    assert root == os.path.join(str(tmp_path), "__cucache__", "pykernels")


# ---------------------------------------------------------------------------
# End to end: a module whose source is unreadable still launches from the store
# ---------------------------------------------------------------------------


def test_a_kernel_launches_when_its_source_cannot_be_read(gpu_backend):
    """The frozen shape, reproduced without freezing.

    The module is compiled from text written to a temporary file, then that file is deleted
    -- exactly what a frozen application looks like to `inspect.getsource`. The launch must
    still translate, from the source the store kept.
    """
    directory = tempfile.mkdtemp()
    module_file = os.path.join(directory, "storedkern.py")
    module_text = (
        "import numpy as np\n"
        "from pygpgpu import kernel\n"
        "\n"
        "@kernel\n"
        "def stored_scale(a, out, factor):\n"
        "    for i in range(out.shape[0]):\n"
        "        out[i] = a[i] * factor\n"
    )
    with open(module_file, "w", encoding="utf-8") as handle:
        handle.write(module_text)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("storedkern", module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bound = module.stored_scale

        a = np.arange(8, dtype=np.float32)
        out = np.zeros(8, dtype=np.float32)
        # First launch: the `.py` is still there, so the store is filled on the way through.
        bound[gpu_backend](a, out, np.float32(3.0))
        assert np.allclose(out, a * 3.0)

        # Delete it, and drop the reader's caches, so only the store can answer.
        os.unlink(module_file)
        import inspect

        from pygpgpu.kernel.translator.source_text import get_dedented_source

        get_dedented_source.cache_clear()
        assert not os.path.isfile(module_file)
        with pytest.raises(OSError):
            inspect.getsource(bound.raw_func)

        out2 = np.zeros(8, dtype=np.float32)
        bound[gpu_backend](a, out2, np.float32(4.0))
        assert np.allclose(out2, a * 4.0), out2
    finally:
        shutil.rmtree(directory, ignore_errors=True)
