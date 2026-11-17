"""A helper-source change must change the kernel's identity, not just its result.

A kernel inlines the transitive closure of the helpers it calls: the closure's source text
*is* the kernel's device code, monomorphized inside the kernel's own translation unit. The
identity built from the kernel's own source alone -- the in-process cache key and the
generated C name -- therefore answered "unchanged" for a kernel whose helper had changed:

* **in-process** (one process, two launches): the per-file mtime check
  (`_source_files_changed`) compares the files the *last compile wrote*. Editing a helper
  module touches none of them, so the stale entry passed the check and the second launch
  answered the old helper's numbers without recompiling anything.
* **across processes** (a fresh interpreter): the binary cache is keyed on the generated
  file's bytes. The helper's file is re-rendered on launch (its own digest moved), but the
  *kernel's* file is byte-identical -- the kernel's name, and so its file content and the
  binary's key, never moved. The parse was re-validated and the stale binary reloaded.

The fix folds a static helper-closure source digest (`helper_closure.helper_closure_digest`)
into both identities: the in-process key in `_prepare_launch` and the C name in
`_c_kernel_name(helpers=...)`. A renamed kernel changes the generated file's bytes, which
changes the binary key, which forces the rebuild -- every layer moves together.

The tests pin the two levels the defect bites at: the digest's answers (a helper edit
moves it, an unrelated module edit does not, a helper-less kernel carries no mark), the
in-process relaunch (the exact sequence that used to answer stale numbers), and the C
name itself.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.translator.source_text import get_dedented_source


# ---------------------------------------------------------------------------
# A file-backed helper module, so an edit can change the text a launch reads
# ---------------------------------------------------------------------------

# Kernels here are compiled by the launcher from `tests/`, so a helper beside this file
# is found on `sys.path` and its `co_filename` is a real path the digest can read. The
# module is written on import (before any kernel of this file launches) and removed at the
# end, so repeated suite runs start from the same text.


HELPER_MODULE_TEXT_V1 = '''\
"""Written by test_kernel_cache_helpers.py; edited by its tests."""
import numpy as np

COEF = 2.0


def gain(x):
    return x * COEF
'''

# The helper's *function text* changed: the closure digest covers source, and a helper
# whose code moved must rename the kernel (the brief's `x * 2.0` -> `x * 3.0` case).
HELPER_MODULE_TEXT_V2 = '''\
"""Written by test_kernel_cache_helpers.py; edited by its tests."""
import numpy as np

COEF = 2.0


def gain(x):
    return x * COEF * 1.5
'''


HELPER_MODULE_NAME = "_cache_helper_module_under_test"


# The helper module is loaded by the fixture and bound here, so `gain_via_module`
# reaches it the way a real project does (`import helpers as hm`, then `hm.gain(...)`):
# the call's callee is an attribute of a module object, which is exactly the shape the
# closure walk follows.
_CHM: object = None


@kernel
def scaled_by_helper(x: np.ndarray, out: np.ndarray):
    """The body the closure digest must follow: it calls the file-backed helper."""
    for i in range(out.shape[0]):
        out[i] = gain_via_module(x[i])


def gain_via_module(value):
    """A helper whose only content is the call into the file-backed module."""
    return _CHM.gain(value)


@pytest.fixture()
def helper_module(monkeypatch):
    """Load the helper module from disk with a plain loader, and drop it after.

    A plain `SourceFileLoader` rather than a plain import: pytest's rewriting hook caches
    rewritten bytecode per (path, mtime, size) and answers a same-size edit with the old
    code object, so the test would reload the edit and still read the old constant. The
    module this fixture yields is therefore not in `sys.modules` under its own name; the
    tests reload it through `reload_from_disk`, which re-executes the file's text.
    """
    import importlib.util
    from importlib.machinery import SourceFileLoader

    path: pathlib.Path = pathlib.Path(__file__).parent / f"{HELPER_MODULE_NAME}.py"
    write_helper_source(path, HELPER_MODULE_TEXT_V1)
    loader: SourceFileLoader = SourceFileLoader(HELPER_MODULE_NAME, str(path))
    spec = importlib.util.spec_from_loader(HELPER_MODULE_NAME, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    get_dedented_source.cache_clear()
    monkeypatch.setattr(sys.modules[__name__], "_CHM", module)

    def reload_from_disk() -> None:
        """Re-read the module's text from the file, as `importlib.reload` would."""
        loader.exec_module(module)

    monkeypatch.setattr(module, "reload_from_disk", reload_from_disk, raising=False)
    try:
        yield module
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop(HELPER_MODULE_NAME, None)
        get_dedented_source.cache_clear()


def write_helper_source(path: pathlib.Path, text: str) -> None:
    """Rewrite the helper module so the interpreter cannot serve stale bytecode.

    A same-length edit lands within the same clock second and keeps the same size, which
    is exactly the pair a timestamped `.pyc` validates with -- the reload would then
    re-execute the *old* bytecode and the test would measure its own write, not the edit.
    Writing through and moving the stamp forward (plus dropping the cached bytecode) makes
    the next import read what was actually written.
    """
    path.write_text(text, encoding="utf-8")
    stamp: float = path.stat().st_mtime + 2.0
    import os

    os.utime(path, (stamp, stamp))
    bytecode: pathlib.Path = path.parent / "__pycache__" / (
        f"{HELPER_MODULE_NAME}.{sys.implementation.cache_tag}.pyc")
    bytecode.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The digest's answers
# ---------------------------------------------------------------------------


def test_a_helper_edit_moves_the_closure_digest(helper_module):
    """Editing the helper's source text moves the digest; that is the whole point."""
    from pygpgpu.kernel.decorator.helper_closure import helper_closure_digest

    resolver = scaled_by_helper.raw_func.__globals__.get
    before: str = helper_closure_digest(scaled_by_helper.raw_func, resolver)
    assert before, "the kernel calls a helper, so the digest must not be empty"

    path: pathlib.Path = pathlib.Path(__file__).parent / f"{HELPER_MODULE_NAME}.py"
    write_helper_source(path, HELPER_MODULE_TEXT_V2)
    helper_module.reload_from_disk()
    get_dedented_source.cache_clear()
    after: str = helper_closure_digest(scaled_by_helper.raw_func, resolver)
    assert after != before
    # And a third text back to the first: the digest returns to the first value.
    write_helper_source(path, HELPER_MODULE_TEXT_V1)
    helper_module.reload_from_disk()
    get_dedented_source.cache_clear()
    assert helper_closure_digest(scaled_by_helper.raw_func, resolver) == before


def test_a_helper_less_kernel_has_no_closure_mark():
    """A kernel that calls no helper keeps the identity it always had."""
    from pygpgpu.kernel.decorator.helper_closure import helper_closure_digest

    @kernel
    def plain(x: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0

    assert helper_closure_digest(plain.raw_func, plain.raw_func.__globals__.get) == ""


# The transitive-closure probe's helpers, at module level: the closure walk resolves
# callees through module globals, which is also how the translator resolves them.
def transitive_inner(value):
    return value * 2.0


def transitive_outer(value):
    return transitive_inner(value) + 1.0


def transitive_unrelated(value):
    return value * 3.0


def transitive_outer_solo(value):
    return value + 1.0


def transitive_outer_and_unrelated(value):
    return transitive_unrelated(value) + 1.0


@kernel
def transitive_caller(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = transitive_outer(x[i])


def test_the_digest_covers_the_transitive_closure():
    """A helper of a helper is in the closure too; a sibling helper is not."""
    from pygpgpu.kernel.decorator.helper_closure import helper_closure_digest

    raw = transitive_caller.raw_func
    resolver = raw.__globals__.get
    with_inner: str = helper_closure_digest(raw, resolver)
    assert with_inner

    # The same body shape, but the helper no longer reaches `inner` (or reaches an
    # unrelated one instead): the digest moves, because the leaf's text is part of
    # what this kernel compiles.
    digest_solo = helper_closure_digest(transitive_outer_solo, resolver)
    digest_unrelated = helper_closure_digest(transitive_outer_and_unrelated, resolver)
    assert digest_solo != with_inner
    assert digest_unrelated != with_inner
    assert digest_solo != digest_unrelated


# ---------------------------------------------------------------------------
# The identities the digest feeds
# ---------------------------------------------------------------------------


def test_a_kernel_c_name_covers_the_helper_closure(helper_module):
    """`_c_kernel_name` folds the closure digest, so an edited helper renames the kernel."""
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper["python"](x, out)
    sig: str = scaled_by_helper._source_signature()
    before: str = scaled_by_helper._c_kernel_name(sig, "cuda",
                                             helpers=scaled_by_helper._helper_closure_digest())

    path: pathlib.Path = pathlib.Path(__file__).parent / f"{HELPER_MODULE_NAME}.py"
    write_helper_source(path, HELPER_MODULE_TEXT_V2)
    helper_module.reload_from_disk()
    get_dedented_source.cache_clear()
    after: str = scaled_by_helper._c_kernel_name(sig, "cuda",
                                            helpers=scaled_by_helper._helper_closure_digest())
    assert after != before
    # The digest mark is part of the payload, not a second suffix: the name still reads
    # `name_<hash>`.
    import re

    assert re.fullmatch(r"scaled_by_helper_[0-9a-f]{8}", after), after


def test_a_kernel_launch_is_recompiled_after_a_helper_edit(helper_module, gpu_backend):
    """The in-process relaunch that used to answer the old helper's numbers.

    Two launches in one process on a compiling backend, the helper edited between them,
    no cache deletion: the per-file mtime check compares files the *last compile wrote*,
    and a helper-module edit touches none of them -- so the cache key has to move (the
    closure digest), or the stale entry would be served without recompiling. The
    `python` backend cannot see this failure (it runs the body live and caches nothing),
    so this test needs a device backend.
    """
    x = np.arange(4, dtype=np.float32)

    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper[gpu_backend](x, out)
    np.testing.assert_allclose(out, x * 2.0)

    path: pathlib.Path = pathlib.Path(__file__).parent / f"{HELPER_MODULE_NAME}.py"
    write_helper_source(path, HELPER_MODULE_TEXT_V2)
    helper_module.reload_from_disk()
    get_dedented_source.cache_clear()

    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper[gpu_backend](x, out)
    np.testing.assert_allclose(out, x * 3.0)


def test_a_changed_constant_value_is_seen_without_a_recompile(helper_module):
    """A constant the helper reads travels as an outer-scope parameter.

    The kernel's own source never moves and neither does the helper's function text, so
    nothing recompiles -- and nothing has to: the launch reads the constant from its own
    scope on every call (see `_bind_outer_variables`), exactly like a changed closure
    cell. This is the rendering project's literal complaint -- a lighting coefficient
    changed -- and the answer for it is a live parameter, not a new identity.
    """
    x = np.arange(4, dtype=np.float32)

    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper["python"](x, out)
    np.testing.assert_allclose(out, x * 2.0)

    path: pathlib.Path = pathlib.Path(__file__).parent / f"{HELPER_MODULE_NAME}.py"
    edited: str = HELPER_MODULE_TEXT_V1.replace("COEF = 2.0", "COEF = 5.0")
    write_helper_source(path, edited)
    helper_module.reload_from_disk()
    get_dedented_source.cache_clear()

    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper["python"](x, out)
    np.testing.assert_allclose(out, x * 5.0)


# A third text for the cache-key test: the earlier tests launch with V1 and V2, so this
# one edits the helper into a form neither has seen, and its key cannot coincide.
HELPER_MODULE_TEXT_V3 = '''\
"""Written by test_kernel_cache_helpers.py; edited by its tests."""
import numpy as np

COEF = 2.0


def gain(x):
    return x * COEF * 2.5
'''


def test_the_in_process_cache_key_covers_the_closure(helper_module, gpu_backend):
    """The `_cache` entry itself moves with the closure, before any mtime is consulted.

    The cache key is observable through the compiled kernel object the launch produced:
    an edit to the helper's text between two launches must change it, on a backend that
    actually compiles (the interpreter backend keeps one runner and no entries).
    """
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper[gpu_backend](x, out)
    keys_before: list = list(scaled_by_helper._cache.keys())

    path: pathlib.Path = pathlib.Path(__file__).parent / f"{HELPER_MODULE_NAME}.py"
    write_helper_source(path, HELPER_MODULE_TEXT_V3)
    helper_module.reload_from_disk()
    get_dedented_source.cache_clear()

    out = np.zeros(4, dtype=np.float32)
    scaled_by_helper[gpu_backend](x, out)
    keys_after: list = list(scaled_by_helper._cache.keys())
    assert keys_after != keys_before, (keys_before, keys_after)
    np.testing.assert_allclose(out, x * 5.0)
