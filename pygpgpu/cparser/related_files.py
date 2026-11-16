# Where a parsed program's `related_files` are recorded, relative to the program itself.
"""Names for the generated files a parse depended on, spelled so they move with the cache.

`related_files` is the set of files a C preprocessor pass touched: the generated source and
every header it `#include`d, recorded so a later process can tell whether the parse is still
current. Absolute paths made that record unusable anywhere but the directory it was written
in -- the whole `__xxcache__` tree could not be copied to another machine, and a cache
warmed at build time could not be shipped inside a packaged application, because the
recorded paths named the build machine.

The record is therefore stored **relative to the directory of the program it describes**
(the cache folder the generated sources live in), which is the unit that moves: the
`__xxcache__` directory travels as one piece. A path that cannot be expressed relative to
that directory is stored absolute, which is correct for a file genuinely elsewhere (an
installed package's header) and keeps the old behaviour for it.
"""
from __future__ import annotations
import hashlib
import os
from typing import Dict, Iterable, Optional, Set



def relative_related_files(related_files:Iterable[str], cache_folder:str)->Set[str]:
    """The recorded form of `related_files`, relative to `cache_folder` where possible.

    Forward slashes are used on every platform: the recorded spelling has to read the same
    whichever machine wrote it, and `os.path.relpath` emits backslashes on Windows.
    """
    root:str = os.path.normcase(os.path.abspath(cache_folder)) if cache_folder else ""
    recorded:Set[str] = set()
    for path in related_files:
        if not path:
            continue
        absolute:str = os.path.abspath(path)
        if root:
            try:
                if os.path.normcase(absolute).startswith(root + os.sep) \
                        or os.path.normcase(absolute) == root:
                    absolute = os.path.relpath(absolute, cache_folder)
            except ValueError:
                # A different drive on Windows: `relpath` refuses, and an absolute path is
                # the only correct spelling.
                pass
        recorded.add(absolute.replace("\\", "/"))
    return recorded



def absolute_related_files(related_files:Iterable[str], cache_folder:str)->Set[str]:
    """The usable form of a recorded set: every entry resolved against `cache_folder`.

    An entry that is already absolute (a header outside the cache) is kept as it is, so a
    record written before this change -- or one holding a genuinely external file -- still
    resolves.
    """
    resolved:Set[str] = set()
    for path in related_files:
        if not path:
            continue
        candidate:str = path.replace("/", os.sep)
        if not os.path.isabs(candidate):
            candidate = os.path.join(cache_folder, candidate)
        resolved.add(os.path.normpath(candidate))
    return resolved



def recorded_related_files(related_files:Iterable[str], program_file:str)->Set[str]:
    """The form a parser stores: relative to the generated source's own directory.

    The base is the *generated file's directory*, not the parser's `_cache_folder`
    attribute -- that one is spelled with a trailing `/__xxcache__` for the meta file's
    name, and using it here would nest a second `__xxcache__` segment into every entry.
    One helper for this and `resolved_related_files` keeps the three parsers from having to
    repeat the reasoning (and the `dirname(abspath(...))` call) a second and third time.
    """
    return relative_related_files(related_files, os.path.dirname(os.path.abspath(program_file)))



def resolved_related_files(related_files:Iterable[str], program_file:str)->Set[str]:
    """The usable form a parser loads: every entry resolved against the source's directory."""
    return absolute_related_files(related_files,
                                  os.path.dirname(os.path.abspath(program_file)))



def file_content_digest(path:str)->str:
    """The md5 of one file's bytes, or "" when it cannot be read."""
    try:
        with open(path, "rb") as file:
            return hashlib.md5(file.read()).hexdigest()
    except OSError:
        return ""



def related_digests(related_files:Iterable[str], cache_folder:str)->Dict[str, str]:
    """The content digest of every recorded file, keyed by its recorded (relative) name.

    A digest rather than the file's timestamp, for the reason the generated source's own
    freshness check already uses one: a timestamp is not preserved by anything that moves
    files. Copying a cache tree (`shutil.copytree`, an archive extraction, a packaging step
    that ships the tree inside the application) rewrites every stamp, so the recorded
    "newer than the meta file" comparison can invert and a perfectly current cache is
    re-parsed -- and, in the packaged case, recompiled on every start. Content is what the
    check is actually about.
    """
    digests:Dict[str, str] = {}
    for recorded in relative_related_files(related_files, cache_folder):
        absolute:str = absolute_related_files([recorded], cache_folder).pop()
        digests[recorded] = file_content_digest(absolute)
    return digests



def related_files_are_current(recorded:Iterable[str], cache_folder:str,
                              stored_digests:Optional[Dict[str, str]],
                              meta_mtime:float)->bool:
    """Whether every recorded file still exists and still holds what the parse read.

    Content decides when a digest was recorded for the file. Timestamps are the fallback,
    for a meta file written before digests were recorded: there the old rule stands (a file
    missing, or newer than the meta, invalidates the parse).

    An empty digest means the file could not be read. That is "not current" rather than
    "unchanged": the fresh set of related files always holds the generated source itself,
    which exists by the time this runs.
    """
    for recorded in recorded:
        absolute:str = next(iter(absolute_related_files([recorded], cache_folder)))
        if not os.path.isfile(absolute):
            return False
        stored:Optional[str] = (stored_digests or {}).get(recorded)
        if stored:
            if file_content_digest(absolute) != stored:
                return False
            continue
        if meta_mtime and os.path.getmtime(absolute) > meta_mtime:
            return False
    return True

