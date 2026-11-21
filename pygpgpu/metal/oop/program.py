"""A compiled Metal library, and the pipeline states its kernels run on.

The compile step is **offline**: the generated MSL is handed to the `metal`
toolchain (`xcrun metal -c` then `xcrun metallib`), and the `.metallib` it
produces is loaded with `newLibraryWithData:error:`. That mirrors the
relationship NVRTC has with nvcc only superficially -- the runtime compiler
(`newLibraryWithSource:options:error:`) was the previous spelling here, and it
was replaced because of what the probes in `metal_codegen.barrier` recorded: a
python process's `MTLCompilerService` session compiles under a different
default language version than a native binary's, and the same source that
compiles cleanly in a compiled program fails in-process with "undeclared
identifier" errors that name the *dialect*, not the code. The offline toolchain
has no such split: one compiler, one dialect, every caller. The cost is an
`xcrun` subprocess per compile (cached by the source digest beside the `.metal`
file), and the shape below (library -> function -> pipeline) is unchanged.

A pipeline state is per function, so `Program.kernels` hands out one `MetalKernel`
per entry point, and the kernel owns its pipeline.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_size_t, c_void_p
from typing import TYPE_CHECKING, Dict, List, Optional

from ..driver.metal import MetaMetal
from ..driver.metal import _wait_ready
from ..driver.objc import ObjC
from .kernel import MetalKernel
# Re-exported so `metal.oop.program.Kernel` answers the same name every other
# backend's `program` module does.
from .kernel import Kernel as Kernel  # noqa: F401
from .compile_options import CompileOptions, from_mapping
from .program_info import KernelInfo
from .program_parser import ProgramParser

if TYPE_CHECKING:
    from .context import Context
    from .devices import Device

# libdispatch's `dispatch_data_create`/`dispatch_get_global_queue`, reached through
# libSystem (which re-exports them) so no extra dylib is loaded. A `dispatch_data_t`
# is what `newLibraryWithData:error:` takes.
#
# Resolved **lazily**, on the first compile: this module is imported by the shared
# compile path on every platform (`_backend_compile`, `kernel_import.source`), and a
# module-level `CDLL` of a macOS-only dylib raised `FileNotFoundError` at import time
# on Windows and Linux -- breaking `import pygpgpu.metal.oop.program` for a caller who
# was only parsing or dispatching another backend. The driver's contract is that a
# missing runtime is recorded and surfaces at first *use*, so the handle is built
# where it is used and only there.
_LIB_SYSTEM:"Optional[ctypes.CDLL]" = None


def _lib_system()->"ctypes.CDLL":
    """libSystem, loaded once on first use and cached for the process."""
    global _LIB_SYSTEM
    if _LIB_SYSTEM is None:
        library:ctypes.CDLL = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        library.dispatch_get_global_queue.restype = c_void_p
        library.dispatch_get_global_queue.argtypes = [c_size_t, c_size_t]
        library.dispatch_data_create.restype = c_void_p
        library.dispatch_data_create.argtypes = [c_void_p, c_size_t, c_void_p, c_void_p]
        _LIB_SYSTEM = library
    return _LIB_SYSTEM


def _compile_offline(source:str, file_name:str,
                     options:Optional[CompileOptions])->bytes:
    """The `.metallib` the offline toolchain builds for one MSL source.

    `xcrun metal -c` translates the source to AIR, `xcrun metallib` bundles the
    AIR into a library. Both read the source from a temp file (the toolchain has
    no stdin path), and both fail loudly: a nonzero status or an empty library
    carries the toolchain's own stderr into the error the caller sees.
    """
    import subprocess
    import tempfile
    import os

    # The generated sources `#include` the shared device-helpers header relative to
    # the cache root (`include/device_helpers.h`), so the compile runs with the
    # source's own directory as the working directory and the cache root on the
    # include path -- the same resolution the runtime compiler's relative lookup
    # gave the include.
    source_dir:str = os.path.dirname(os.path.abspath(file_name))
    cache_root:str = os.path.dirname(source_dir)
    with tempfile.TemporaryDirectory(prefix="pygpgpu-metal-") as workdir:
        # The file name is what the compiler prints in its diagnostics, so the
        # caller's own name is what appears there.
        source_path:str = os.path.join(workdir, os.path.basename(file_name))
        with open(source_path, "w", encoding="utf-8") as source_file:
            source_file.write(source)
        air_path:str = os.path.join(workdir, "out.air")
        lib_path:str = os.path.join(workdir, "out.metallib")

        if MetaMetal.print_info:
            print(f"building {os.path.basename(file_name)} ... ", end="", flush=True)
        translate:str = subprocess.run(
            ["xcrun", "metal", "-c", source_path, "-o", air_path,
             "-fno-fast-math",
             "-I", cache_root, "-I", source_dir],
            capture_output=True, text=True, timeout=300,
            cwd=source_dir if os.path.isdir(source_dir) else None)
        if MetaMetal.print_info:
            print("done." if translate.returncode == 0 else "FAILED", flush=True)
        if translate.returncode != 0 or not os.path.isfile(air_path):
            # Keep the exact text the compiler rejected, beside the *cache* rather than
            # beside the temp copy: the `with` block deletes the temp directory on the way
            # out, so a file written next to `source_path` would be gone by the time the
            # reader followed the error's own line numbers. `file_name` IS the cache file
            # the text came from, so naming it in the message is enough -- the reader opens
            # the file the compile read, with the diagnostics' lines intact.
            raise RuntimeError(
                f"compiling {file_name} failed (xcrun metal): "
                f"{(translate.stderr or translate.stdout).strip()}")
        bundle:str = subprocess.run(
            ["xcrun", "metallib", air_path, "-o", lib_path],
            capture_output=True, text=True, timeout=300)
        if bundle.returncode != 0 or not os.path.isfile(lib_path):
            raise RuntimeError(
                f"compiling {file_name} failed (xcrun metallib): "
                f"{(bundle.stderr or bundle.stdout).strip()}")
        with open(lib_path, "rb") as library_file:
            return library_file.read()


class Program:
    """One `MTLLibrary` built from a source file, plus the layouts of its kernels."""

    def __init__(self, file_name:str, includes:Optional[List[str]]=None,
                 defines:Optional[Dict[str, str]]=None, type_checked:bool=False,
                 device:object=None, options:Optional[CompileOptions]=None,
                 **build_options:object)->None:
        # The parser reads the signature the launch binds arguments to; it runs
        # before the compile so a source the scanner cannot read is refused with
        # the scanner's own message rather than the compiler's.
        # The MSL compile settings are NOT swallowed: `options` used to arrive as a
        # keyword argument and land in `**build_options`, which silently dropped it
        # while every other backend honoured its own. `from_mapping` reads the MSL
        # settings out of whatever the caller passed, and answers None -- the
        # compiler's defaults, which is what passing nil means -- when the object
        # carries none (an OpenCL `BuildOptions`, for instance). The offline
        # toolchain maps `language_version` onto `metal -std=`; the settings the
        # toolchain has no spelling for are honoured by the runtime compile fallback
        # (see `_compile_offline`'s docstring in compile_options).
        self.compile_options:Optional[CompileOptions] = from_mapping(options)
        self._parser:ProgramParser = ProgramParser()
        self._parser.parse(file_name, includes, defines)
        self._file_name:str = file_name
        self._type_checked:bool = type_checked
        self._kernels:Dict[str, MetalKernel] = {}

        _wait_ready()
        context = (device.default_context if device is not None
                   else _default_context())
        self._context:"Context" = context
        self._device:"Device" = context.device

        library_id:int = self._load(self._parser.clean_code, self.compile_options)
        if not library_id:
            raise RuntimeError(f"Metal could not compile {file_name}")
        self._library_id:int = library_id

    def _load(self, source:str, options:Optional[CompileOptions])->int:
        """Compile offline and wrap the result as an `MTLLibrary`.

        The metallib bytes travel through `dispatch_data_create` (whose no-op
        destructor borrows the caller's bytes) and `newLibraryWithData:error:`.
        `fastMathEnabled` has no offline spelling in the toolchain flags the
        generated sources need, so a caller that configured it is refused rather
        than silently ignored -- the honest answer to "this option has no
        offline path".
        """
        if options is not None and (options.fast_math is not None
                                    or options.math_mode is not None
                                    or options.macros is not None):
            raise RuntimeError(
                "the offline Metal toolchain does not accept fastMath/mathMode/"
                "macros settings; drop them from the compile options or use the "
                "runtime compiler backend")
        library_bytes:bytes = _compile_offline(source, self._file_name, options)

        holder = ctypes.create_string_buffer(library_bytes, len(library_bytes))
        # The no-op destructor (a literal NULL dispatch block) borrows the bytes;
        # `holder` outlives the call, and Metal copies what it needs into the
        # MTLLibrary it returns.
        lib_system:ctypes.CDLL = _lib_system()
        data_id:int = lib_system.dispatch_data_create(
            ctypes.cast(holder, c_void_p), c_size_t(len(library_bytes)),
            lib_system.dispatch_get_global_queue(0, 0), c_void_p(0))
        if not data_id:
            raise RuntimeError(f"Metal could not wrap the library for {self._file_name}")
        try:
            error = c_void_p(0)
            library = ObjC.send(
                self._device.id, "newLibraryWithData:error:",
                (c_void_p, POINTER(c_void_p)),
                c_void_p, data_id, byref(error))
            if not library and error:
                message:str = ObjC.describe(int(error.value or 0))
                raise RuntimeError(
                    f"loading the compiled library for {self._file_name} failed: {message}")
            return int(library or 0)
        finally:
            ObjC.release(int(data_id))

    @property
    def id(self)->int:
        return self._library_id

    @property
    def context(self)->"Context":
        return self._context

    @property
    def device(self)->"Device":
        return self._device

    @property
    def file_name(self)->str:
        return self._file_name

    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._parser.kernel_infos

    @property
    def kernel_names(self)->List[str]:
        return list(self.kernel_infos.keys())

    @property
    def structs(self)->Dict[str, type]:
        return self._parser.structs

    @property
    def kernels(self)->Dict[str, MetalKernel]:
        self._fetch_kernels()
        return self._kernels

    def _fetch_kernels(self)->None:
        if self._kernels:
            return
        for name, info in self.kernel_infos.items():
            kernel = MetalKernel(self, name)
            kernel.args = info.args
            # The attribute parameters are declared by the device signature but filled
            # in by the dispatch, so they are not among the bindable arguments; the
            # launch needs their names to know the signature's tail is not its own.
            kernel.dispatch_parameters = list(getattr(info, "dispatch_parameters", ()))
            kernel.type_checked = self._type_checked
            self._kernels[name] = kernel

    def __getitem__(self, name:str)->MetalKernel:
        if name in self.kernel_infos:
            return self.kernels[name]
        raise KeyError(name)

    def __getattr__(self, name:str)->MetalKernel:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.kernel_infos:
            return self.kernels[name]
        raise AttributeError(f"the program {self._file_name} declares no kernel {name!r}")


def _default_context()->"Context":
    """The context a Program compiles for when the caller did not name a device."""
    from .devices import Devices
    return Devices[0].default_context
