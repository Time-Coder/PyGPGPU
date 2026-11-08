# Writing a generated file: the digest sidecar and the atomic replace.
from __future__ import annotations
from .locks import (
    _SOURCE_WRITE_POLL,
    _SOURCE_WRITE_TIMEOUT,
    _ensure_directory,
    write_lock,
)
from .merge import (
    _canonical_header,
    _merge_definitions,
    _split_source,
    assemble_source,
    merge_source,
)
import contextlib
import hashlib
import os
import threading
import time
from typing import List, Optional



def _write_source_file(path:str, source:str)->None:
    """Replace one generated file's contents, surviving a transient external hold.

    A plain `open(path, "w")` fails with `PermissionError` when *another process* holds the
    destination with a non-shared handle. Measured on Windows: a same-process read handle
    does **not** block a plain write, but it does block `os.replace`; an exclusive handle
    held by another process blocks both. Since the generated source is rewritten hundreds of
    times while a module's kernels compile, a real-time scanner or an indexer taking the file
    for a moment lands inside one of those windows and the whole launch fails -- which is
    the intermittent error this code exists to remove.

    So the write is tolerant rather than optimistic: it tries `os.replace` over a temp file
    (which is what makes a reader never see a half-written file), and on `PermissionError`
    it retries for a bounded time -- alternating the atomic form with a plain overwrite,
    because the two fail for *different* kinds of holder. Only when the hold outlives the
    whole budget does the error surface, and then it names the file and the holder hint.
    """
    temp_name:str = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    deadline:float = time.monotonic() + _SOURCE_WRITE_TIMEOUT
    payload:bytes = source.encode("utf-8")
    last_error:Optional[PermissionError] = None
    while True:
        try:
            with open(temp_name, "wb") as temp_file:
                temp_file.write(payload)
            os.replace(temp_name, path)
            return
        except PermissionError as error:
            last_error = error
        try:
            # A plain overwrite is refused by a *different* set of holders than `os.replace`
            # is, so trying both each pass covers whichever one is in the way.
            with open(path, "wb") as file:
                file.write(payload)
            return
        except PermissionError as error:
            last_error = error
        finally:
            with contextlib.suppress(OSError):
                if os.path.isfile(temp_name):
                    os.unlink(temp_name)
        if time.monotonic() >= deadline:
            raise PermissionError(
                f"cannot write {path}: another process held it for more than"
                f" {_SOURCE_WRITE_TIMEOUT:.0f}s (a real-time scanner or an indexer holding a"
                f" generated file is the usual cause). Last error: {last_error}") from None
        time.sleep(_SOURCE_WRITE_POLL)



def append_source(path: str, header: List[str], definitions: List[str],
                  source_python: str, source_digest: str = "") -> str:
    """Write definitions into a generated file, keeping what the file already holds.

    One python module owns one generated file, and it is compiled once per kernel, so each
    call contributes the definitions it rendered and must not drop the ones an earlier call
    wrote. Python functions in one module have distinct names, so the union cannot collide;
    a definition the file already has is skipped, which is what makes a recompiled kernel
    idempotent.

    The whole read-merge-write is one critical section: two threads or two processes
    compiling different kernels of this module reach this function with the same path, and
    an unserialized writer would base its merge on a file that no longer holds what the
    other one added. See `write_lock`.
    """
    with write_lock(path):
        return _append_source_locked(path, header, definitions, source_python, source_digest)



def _append_source_locked(path: str, header: List[str], definitions: List[str],
                          source_python: str, source_digest: str) -> str:
    """`append_source`'s body, with the file's locks already held."""
    existing: Optional[str] = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        except OSError:
            existing = None

    if existing is not None:
        # The digest sidecar is the authority when it exists: it describes the python
        # text that wrote this file, so a matching digest means nothing changed even
        # when the timestamps say the source is newer (`git checkout` sets the mtime
        # back, and two files written inside one clock tick compare equal). The clock
        # is only consulted when no digest was ever recorded -- a file from before the
        # sidecar existed, or one written without a digest -- and even then it only
        # decides the ambiguous direction: an equal timestamp is not evidence that the
        # source changed, so the file is kept.
        stored: str = _stored_digest(path)
        if source_digest and stored:
            target_is_newer = stored == source_digest
        else:
            try:
                target_is_newer = (
                    os.path.isfile(source_python)
                    and os.path.getmtime(path) >= os.path.getmtime(source_python))
            except OSError:
                target_is_newer = False
        if not target_is_newer:
            # The file describes python text that has since changed: start it over.
            existing = None

    kept_definitions: List[str] = []
    kept_header: List[str] = []
    if existing is not None:
        kept_header, kept_definitions = _split_source(existing, path)

    # A definition the file already holds is kept **unless this render writes it
    # differently**: the name hashes the signature, so a package-side change to the
    # generated body keeps the name (see `_merge_definitions`). Nothing is skipped here even
    # when every definition is already known: the header may still gain a quaternion width
    # the earlier run did not need.
    kept_definitions, _ = _merge_definitions(kept_definitions, definitions)

    merged_header: List[str] = (list(kept_header) if existing is not None else [])
    for piece in header:
        cleaned = piece.strip("\n")
        if cleaned:
            merged_header.append(cleaned)
    merged_header = _canonical_header(merged_header)
    source: str = assemble_source(merged_header, kept_definitions, path)
    _ensure_directory(os.path.dirname(path))
    # An unchanged file is not rewritten. Every rewrite opens a window in which a scanner
    # or an indexer can hold the destination exclusively, and the module is re-rendered on
    # every kernel call -- 265 rewrites of one file in a single test module, most of them
    # byte-identical. Skipping the no-op write removes those windows entirely, and the
    # digest is still refreshed because it is a separate, tolerant write.
    if existing != source:
        _write_source_file(path, source)
    _write_digest(path, source_digest)
    return path



DIGEST_SUFFIX: str = ".source-digest"



def digest_sidecar(path:str)->str:
    """Where a generated file's source digest is kept.

    A sidecar rather than a comment inside the source: the generated text is a
    carefully-assembled translation unit whose blank lines are significant (`_split_source`
    splits it on a blank line), so threading one more line through it risked changing how a
    later write re-reads the file. A `.source-digest` file next to the source cannot affect
    what the compiler sees.
    """
    return path + DIGEST_SUFFIX



def _write_digest(path:str, digest:str)->None:
    """Record the digest beside a generated file, but only when it actually changed.

    An unconditional rewrite moves the sidecar's mtime on every kernel call, including the
    calls that changed nothing: measured across repeated launches of one module, the sidecar
    advanced while the generated source it describes stayed byte-identical. The sidecar is
    not in a parser's `related_files` list, so this does not invalidate a cache by itself --
    but it is file churn inside the critical section that also writes the source, and a
    freshness test that compares mtimes is decided by such small gaps. Writing only on a
    change keeps the sidecar's timestamp honest: it moves exactly when the digest it holds
    moves.
    """
    if not digest:
        return
    sidecar:str = digest_sidecar(path)
    if _stored_digest(path) == digest:
        return
    try:
        with open(sidecar, "w", encoding="utf-8", newline="\n") as file:
            file.write(digest + "\n")
    except OSError:
        pass



def _stored_digest(path:str, source:str="")->str:
    """The digest recorded beside a generated file, or "" when there is none."""
    try:
        with open(digest_sidecar(path), "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""



def python_source_digest(source_python:str)->str:
    """A digest of the python text that produced a generated file.

    The freshness test used to be mtime alone, which assumes a file's timestamp moves
    forward when its text changes. Git breaks that assumption: `git checkout` and
    `git stash pop` restore a file *and* set its mtime back, so a generated file written
    from newer text kept looking "newer than the source" and was reused -- leaving the old
    declaration of a kernel beside the new one, and the compiler reported "more than one
    instance of overloaded function". Reading the text is what makes the test about
    content rather than about a clock.
    """
    try:
        with open(source_python, "r", encoding="utf-8") as file:
            return hashlib.md5(file.read().encode("utf-8")).hexdigest()
    except OSError:
        return ""



def write_source(path: str, source: str, source_python: str,
                 keep_definitions: bool = True) -> str:
    """Write one module's generated source, honouring the append/overwrite rule.

    A target file **newer** than the python file it came from is kept and the new
    definitions are appended to it: the file was written after that python text, so it
    holds work this run should not throw away. An **older** target is replaced outright, it
    describes python text that has since changed.

    `keep_definitions` makes an overwrite keep the definitions the file already holds, which
    is what one python module needs: two kernels defined beside each other live in one
    source file, and each is compiled on its own call, so the second call must not erase the
    first kernel. Python functions in one module have distinct names, so adding them cannot
    collide.

    Like `append_source`, this is a read-modify-write over a file several threads and
    processes may be writing, so it runs under the same locks.
    """
    with write_lock(path):
        return _write_source_locked(path, source, source_python, keep_definitions)



def _write_source_locked(path: str, source: str, source_python: str,
                         keep_definitions: bool) -> str:
    """`write_source`'s body, with the file's locks already held."""
    _ensure_directory(os.path.dirname(path))
    existing: Optional[str] = None
    if os.path.isfile(path):
        existing = None
        try:
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        except OSError:
            existing = None

    if existing is not None and os.path.isfile(source_python):
        try:
            target_is_newer: bool = (os.path.getmtime(path)
                                     > os.path.getmtime(source_python))
        except OSError:
            target_is_newer = False
        if target_is_newer:
            merged: str = merge_source(existing, source)
            if merged != existing:
                _write_source_file(path, merged)
            return path

    if existing is not None and keep_definitions:
        # The file is this module's whole device code, and this call renders only one
        # kernel plus the helpers it reaches. Keep what the file already holds as well, so
        # a module with two kernels keeps both.
        combined: str = merge_source(existing, source)
        if combined != existing:
            _write_source_file(path, combined)
        return path

    if existing != source or existing is None:
        _write_source_file(path, source)
    return path
