# The import hook: finder + loader for handwritten kernel modules.
from __future__ import annotations
from .source import _source_paths_for
from .wrapper import _MergedModuleLoader
from .tables import _BACKEND_EXTENSIONS
import sys
import threading
from typing import Dict, List, Optional



class kernel_import:
    """The import hook: `kernel_import.install()`, then import the module."""

    _lock = threading.Lock()
    _installed:bool = False
    _config:Dict[str, object] = {
        "type_checked": False,
        "generate_pyi": True,
        # Per-backend build options; None keeps each backend's own default.
        "options": {"cuda": None, "hip": None, "opencl": None},
    }

    @staticmethod
    def install()->None:
        """Install the finder. Idempotent, and it goes in front of the backend hooks."""
        with kernel_import._lock:
            if kernel_import._installed:
                return
            sys.meta_path.insert(0, _KernelImportFinder())
            kernel_import._installed = True

    @staticmethod
    def uninstall()->None:
        with kernel_import._lock:
            for i in range(len(sys.meta_path) - 1, -1, -1):
                if isinstance(sys.meta_path[i], _KernelImportFinder):
                    del sys.meta_path[i]
            kernel_import._installed = False

    @staticmethod
    def config(type_checked:bool=False, generate_pyi:bool=True,
               options:Optional[Dict[str, object]]=None)->None:
        """Configure the hook. `options` maps backend -> BuildOptions (None = default)."""
        with kernel_import._lock:
            kernel_import._config["type_checked"] = type_checked
            kernel_import._config["generate_pyi"] = generate_pyi
            if options is not None:
                kernel_import._config["options"].update(options)


class _KernelImportFinder:
    """The meta-path finder: resolve a name to same-named .cl/.cu/.hip sources."""

    def find_spec(self, fullname:str, path:Optional[List[str]] = None,
                  target:object = None)->Optional[object]:
        import importlib.util
        sources:Dict[str, str] = _source_paths_for(fullname, path)
        if not sources:
            return None
        config:Dict[str, object] = kernel_import._config
        loader = _MergedModuleLoader(fullname, sources, config["options"],
                                     config["type_checked"], config["generate_pyi"])
        # The module's file location is whichever source comes first in the probe order;
        # a merged module is a leaf, so it carries no `__path__`.
        first_source:str = next(sources[backend] for backend, _ in _BACKEND_EXTENSIONS
                                if backend in sources)
        return importlib.util.spec_from_file_location(fullname, first_source, loader=loader)
