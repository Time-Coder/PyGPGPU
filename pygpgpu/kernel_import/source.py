# Reading a handwritten .cl/.cu/.hip module: paths, parsers, includes.
from __future__ import annotations
from .tables import _BACKEND_EXTENSIONS, _PARSER_MODULES, _work_size_item
import importlib
import sys
import os
import threading
from typing import Dict, List, Optional, Tuple, Union



def _source_paths_for(fullname:str, path:Optional[List[str]])->Dict[str, str]:
    """The backend sources one module name resolves to: {"cuda": path, ...}.

    `fullname` may be dotted (`kernel_import.vecops`); the file path is the module path
    joined with each backend's extension. When `path` is given (a parent package's
    `__path__`, which is how `from kernel_import import vecops` arrives) only the module's
    own name is appended to each of those directories; otherwise the full dotted path is
    spelled under each `sys.path` entry. The first directory holding any of the extensions
    wins as a whole, so a module's backends always come from one place.
    """
    parts:list = fullname.split(".")
    if path is None:
        # A top-level name: walk sys.path the way the backend hooks do.
        search:List[str] = list(sys.path)
    else:
        # A parent package's `__path__`: python has already descended into the package.
        search:List[str] = list(path)
        parts = parts[-1:]

    for directory in search:
        if not directory:
            continue
        file_path:str = directory.replace("\\", "/")
        for part in parts:
            file_path += f"/{part}"

        found:Dict[str, str] = {}
        for backend, extension in _BACKEND_EXTENSIONS:
            candidate:str = file_path + extension
            if os.path.isfile(candidate):
                found[backend] = candidate
        if found:
            return found
    return {}


class _BackendSource:
    """One backend's source: its parsed metadata and the programs built from it.

    Parsing happens when the merged module is imported (cheap, cache-backed) so structures
    and signatures are known without a device; a `Program` is built on first use per
    device, because that is where the backend's compiler and context belong.
    """

    def __init__(self, backend:str, file_name:str, options:object, type_checked:bool):
        self.backend:str = backend
        self.file_name:str = file_name
        self.options:object = options
        self.type_checked:bool = type_checked
        self.parser:object = self._parse()
        self._lock = threading.Lock()
        self._programs:Dict[str, object] = {}
        self._kernels:Dict[str, object] = {}
        # The per-source compiler object each backend needs: NVRTC for CUDA, the clang
        # wrapper for HIP, and nothing extra for OpenCL (a context compiles directly).
        self._compiler:object = None

    def _parse(self)->object:
        # Importing `pygpgpu.<backend>` initializes that backend's driver, whose init()
        # records -- but does not raise -- a runtime the machine lacks (see the driver
        # layer). Nothing special is needed here: a parse is pure tree-sitter over a
        # file, and the driver's own failure surfaces when a call is actually made.
        module:object = importlib.import_module(_PARSER_MODULES[self.backend], "pygpgpu")
        parser:object = module.ProgramParser()
        parser.parse(self.file_name, options=self.options)
        return parser

    def kernel_infos(self)->Dict[str, object]:
        return self.parser.kernel_infos

    def struct_types(self)->Dict[str, type]:
        return self.parser._struct_types

    # ---------- kernels ----------

    def kernel(self, name:str, device_sel:Optional[str],
               bracket:Optional[Union[int, str, Tuple[int, ...]]])->object:
        """The backend's own `Kernel`, bound to a device and to the bracket's work size.

        The device is resolved through the decorator's resolver, so every spelling the
        `@kernel` bracket accepts works here too. ``bracket`` is what the caller wrote:
        a CUDA ``(blocks, threads)`` pair, a bare item count, or None for automatic.
        """
        device:object = self._device(device_sel)
        device_key:str = getattr(device, "unique_key", None) or str(device)
        key:str = f"{device_key}|{name}"
        with self._lock:
            kernel:object = self._kernels.get(key)
            if kernel is None:
                kernel = self._program(device)[name]
                self._kernels[key] = kernel
        if self.backend == "opencl":
            # A program belongs to a platform, and a kernel handle to the whole context;
            # the device it runs on is per-call state on the Kernel itself. Every other
            # backend gets its device from the context its Program was built in.
            kernel = kernel[device]
        if bracket is not None:
            kernel = kernel[_work_size_item(bracket)]
        return kernel

    def _device(self, device_sel:Optional[str])->object:
        from ..kernel.decorator import _resolve_device
        device:object
        device, _key = _resolve_device(self.backend, device_sel)
        return device

    def _program(self, device:object)->object:
        """This backend's `Program` for one device, compiled on first use."""
        backend:str = self.backend
        if backend == "opencl":
            from ..opencl.oop.program import Program
            platform_key:str = str(id(device.platform))
            program:Optional[object] = self._programs.get(platform_key)
            if program is None:
                program = Program(device.platform.default_context,
                                  type_checked=self.type_checked,
                                  program_parser=self.parser)
                self._programs[platform_key] = program
            return program
        if backend == "cuda":
            from ..cuda.oop.program import Program
            device_key:str = getattr(device, "unique_key", None) or str(device)
            program = self._programs.get(device_key)
            if program is None:
                device.default_context.make_current()
                program = Program(self.file_name, options=self.options,
                                  type_checked=self.type_checked,
                                  nvrtc_program=self._nvrtc(device))
                self._programs[device_key] = program
            return program
        if backend == "hip":
            from ..hip.oop.program import Program
            device_key = getattr(device, "unique_key", None) or str(device)
            program = self._programs.get(device_key)
            if program is None:
                device.default_context.make_current()
                program = Program(self.file_name, options=self.options,
                                  type_checked=self.type_checked,
                                  compiler=self._hip_compiler())
                self._programs[device_key] = program
            return program
        if backend == "metal":
            from ..metal.oop.program import Program
            device_key = getattr(device, "unique_key", None) or str(device)
            program = self._programs.get(device_key)
            if program is None:
                program = Program(self.file_name, options=self.options,
                                  type_checked=self.type_checked, device=device)
                self._programs[device_key] = program
            return program
        raise ValueError(f"unknown backend '{backend}'")

    def _nvrtc(self, device:object)->object:
        """The NVRTC program for this source, parsed once and compiled per device."""
        from ..cuda.oop.nvrtc_program import NVRTCProgram
        if self._compiler is None:
            self._compiler = NVRTCProgram(self.file_name, options=self.options,
                                          program_parser=self.parser)
        return self._compiler

    def _hip_compiler(self)->object:
        """The HIP compiler for this source, parsed once and compiled per architecture."""
        from ..hip.oop.hip_compiler import HipCompiler
        if self._compiler is None:
            self._compiler = HipCompiler(self.file_name, options=self.options,
                                         program_parser=self.parser)
        return self._compiler
