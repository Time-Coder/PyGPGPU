"""The helpers the split test files share.

Each helper keeps the name it had, so a test reads the same either way.
"""
from __future__ import annotations

from __future__ import annotations
import pytest
from pygpgpu.cuda.libraries import (
    driver_for,
)


def _library_available(name:str)->bool:
    """Whether a library's DLL can be loaded on this machine."""
    try:
        driver_for(name).dll()
        return True
    except OSError:
        return False


def _require_library(name:str)->None:
    """Skip the test when the library is not installed here."""
    if not _library_available(name):
        pytest.skip(f"the {name} library is not installed on this machine")


