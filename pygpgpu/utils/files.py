# The cache directory, the atomic write, and the digests the cache is keyed on.
from __future__ import annotations
import contextlib
import hashlib
import re
import os
import pickle
import threading
import time
from collections.abc import Iterable
from typing import Union
import numpy as np



def product(values:Iterable[int])->int:
    """Product of an iterable (kept local so callers need no math import)."""
    total:int = 1
    for value in values:
        total *= value
    return total


def modify_time(file_path:str)->float:
    if not os.path.isfile(file_path):
        return 0
    
    return os.path.getmtime(file_path)

def md5sum(data_or_path:Union[str, np.ndarray, bytes, bytearray])->str:
    md5_hash = hashlib.md5()

    if isinstance(data_or_path, str):
        chunk_size:int = 4096
        
        with open(data_or_path, "rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                md5_hash.update(chunk)
    elif isinstance(data_or_path, (bytes, bytearray)):
        md5_hash.update(data_or_path)

    elif isinstance(data_or_path, np.ndarray):
        md5_hash.update(data_or_path.tobytes())
    
    return md5_hash.hexdigest()

def md5sums(content:str)->str:
    return md5sum(content.encode("utf-8"))

def cat(file_name: str) -> str:
    # charset_normalizer ships with imageio, but it was never a declared
    # dependency and importing it at module scope made this module -- and
    # therefore the whole package -- fail to import wherever it is absent.
    # It is only needed to guess the encoding of an unreadable file, so import
    # it on demand and fall back to UTF-8 when it is not installed.
    try:
        from charset_normalizer import from_path
    except ImportError:
        with open(file_name, "r", encoding="utf-8", errors="replace") as file:
            return file.read()
    return str(from_path(file_name).best())
    
def echo(content:str, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "w", encoding='utf-8') as file:
        file.write(content)

def _atomic_write(file_name:str, data:bytes)->None:
    """Write `data` to `file_name` in a way no concurrent reader can observe half-done.

    The write goes to a temp file in the same directory and is moved into place with
    `os.replace`, which is atomic on every platform this targets, so a reader sees either
    the previous file or the finished new one. The temp name carries both the pid and the
    thread id: two writers in one process and two processes must never share a temp file, or
    the replace would install one writer's bytes over the other's unfinished ones.

    `os.replace` can still refuse on Windows with a sharing violation when another thread
    happens to hold the destination open for a read at that instant -- the name is in use,
    not the data. That is transient by construction, so it is retried briefly; a failure
    that outlives the retries falls back to a plain overwrite, which is the pre-existing
    behaviour rather than a new error.
    """
    temp_name:str = f"{file_name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temp_name, "wb") as file:
            file.write(data)
        for _attempt in range(8):
            try:
                os.replace(temp_name, file_name)
                return
            except PermissionError:
                # A reader has the destination open; let it finish and try again.
                time.sleep(0.05)
        with open(file_name, "wb") as file:
            file.write(data)
    finally:
        with contextlib.suppress(OSError):
            if os.path.isfile(temp_name):
                os.unlink(temp_name)


def save_var(var:object, file_name:str)->None:
    """Pickle `var` to its cache file, atomically (see `_atomic_write`)."""
    _ensure_directory(os.path.dirname(os.path.abspath(file_name)))
    _atomic_write(file_name, pickle.dumps(var))

def load_var(file_name:str)->object:
    with open(file_name, "rb") as file:
        return pickle.load(file)
    
def _ensure_directory(directory:str)->None:
    """Make `directory` exist, tolerating the losers of a concurrent create.

    `os.makedirs(..., exist_ok=True)` is not atomic on Windows: it creates each missing
    segment with its own call and can still raise for a segment another thread just made, so
    two writers that both find a parent missing can fail the one that was only second. The
    levels are therefore raised one segment at a time, and each segment's failure is
    re-checked against the filesystem: if the segment exists now, another writer created
    exactly the directory this call needed, so there is nothing to report.
    """
    if not directory:
        return
    absolute:str = os.path.abspath(directory)
    head, tail = os.path.split(absolute)
    if tail:
        _ensure_directory(head)
    for _attempt in range(8):
        try:
            os.mkdir(absolute)
            return
        except FileExistsError:
            if os.path.isdir(absolute):
                return
        except OSError:
            if os.path.isdir(absolute):
                return
            time.sleep(0.05)
    if os.path.isdir(absolute):
        return
    os.makedirs(absolute, exist_ok=True)


def save_bin(bin:Union[bytes, bytearray], file_name:str)->None:
    """Write a compiled binary to its cache file, atomically.

    The file is one compiled artifact and several processes may be compiling the same
    module at once. A reader could see a half-written file and load a truncated binary
    (`CL_INVALID_KERNEL`), and two writers interleaving their bytes would leave one that is
    a mix of both. The write therefore goes to a temp file in the same directory and is
    moved into place with `os.replace`, which is atomic on every platform this targets -- so
    a reader sees either the previous artifact or the finished new one, never a partial one.
    """
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    _ensure_directory(target_folder)
    _atomic_write(file_name, bytes(bin))

def load_bin(file_name:str)->bytes:
    with open(file_name, "rb") as file:
        return file.read()

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    illegal_chars = r'[\\/:*?"<>|]'
    safe_name = re.sub(illegal_chars, '_', filename)
    
    safe_name = safe_name.strip().strip('.')
    
    if not safe_name:
        safe_name = "unnamed"
    
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    if safe_name.upper() in reserved_names:
        safe_name = f"_{safe_name}_"
    
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name

def complete_basename(file_name:str):
    return '.'.join(os.path.basename(file_name).split('.')[:-1])
