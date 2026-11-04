"""A compiled Metal library, and the pipeline states its kernels run on.

The compile step is `newLibraryWithSource:options:error:`, the runtime compiler
Metal ships with. That mirrors the relationship NVRTC has with nvcc: a `@kernel`
compiles at first call, and the result is cached beside the source (see
`pygpgpu/kernel/source_paths.py`). An offline `xcrun metal` build would remove the
compile cost from a launch and is a later step; the shape below
(library -> function -> pipeline) is exactly what such a build would produce, so
switching to it would replace only `Program.__init__`.

A pipeline state is per function, so `Program.kernels` hands out one `MetalKernel`
per entry point, and the kernel owns its pipeline.
"""
from __future__ import annotations

from ctypes import POINTER, byref, c_char_p, c_void_p
from typing import Any, Dict, List, Optional

from ..driver.metal import _wait_ready
from ..driver.objc import ObjC
from .kernel import MetalKernel
# Re-exported so `metal.oop.program.Kernel` answers the same name every other
# backend's `program` module does.
from .kernel import Kernel as Kernel  # noqa: F401
from . import compile_options as compile_options_module
from .compile_options import CompileOptions, from_mapping
from .program_info import KernelInfo
from .program_parser import ProgramParser


class Program:
    """One `MTLLibrary` built from a source file, plus the layouts of its kernels."""

    def __init__(self, file_name:str, includes:Optional[List[str]]=None,
                 defines:Optional[Dict[str, Any]]=None, type_checked:bool=False,
                 device:Any=None, options:Any=None, **build_options:Any)->None:
        # The parser reads the signature the launch binds arguments to; it runs
        # before the compile so a source the scanner cannot read is refused with
        # the scanner's own message rather than the compiler's.
        # The MSL compile settings are NOT swallowed: `options` used to arrive as a
        # keyword argument and land in `**build_options`, which silently dropped it
        # while every other backend honoured its own. `from_mapping` reads the MSL
        # settings out of whatever the caller passed, and answers None -- the
        # compiler's defaults, which is what passing nil means -- when the object
        # carries none (an OpenCL `BuildOptions`, for instance).
        self.compile_options:Optional[CompileOptions] = from_mapping(options)
        self._parser:ProgramParser = ProgramParser()
        self._parser.parse(file_name, includes, defines)
        self._file_name:str = file_name
        self._type_checked:bool = type_checked
        self._kernels:Dict[str, MetalKernel] = {}

        _wait_ready()
        context = (device.default_context if device is not None
                   else _default_context())
        self._context:Any = context
        self._device:Any = context.device

        library_id = self._compile(self._parser.clean_code, self.compile_options)
        if not library_id:
            raise RuntimeError(f"Metal could not compile {file_name}")
        self._library_id:int = library_id

    def _compile(self, source:str,
                 options:Optional[CompileOptions]=None)->int:
        """Build an `MTLLibrary` from MSL source.

        `options` is the `MTLCompileOptions` object `compile_options.build` produced,
        or 0 for the compiler's defaults. Passing nil (which is what this used to do
        unconditionally) means "the defaults" and leaves a kernel that needs
        IEEE-exact math no way to ask for it.

        Both Objective-C objects this creates are +1 references the callee owns --
        the options object and, on failure, the `NSError` -- so each is released
        here rather than left to the pool. The library itself is +1 and is owned by
        the `Program` (released by `MTLObject.__del__`).
        """
        error = c_void_p(0)
        options_id:int = compile_options_module.build(options)
        try:
            library = ObjC.send(
                self._device.id, "newLibraryWithSource:options:error:",
                (c_char_p, c_void_p, POINTER(c_void_p)),
                c_void_p, source.encode("utf-8"), options_id, byref(error))
            if not library and error:
                # The description is read *before* the object is released below:
                # `describe` sends a message to it, and a released object may be gone.
                message:str = ObjC.describe(int(error.value or 0))
                raise RuntimeError(
                    f"compiling {self._file_name} failed: {message}")
            return int(library or 0)
        finally:
            # Both objects are +1 references this call owns: the compiler copies
            # whatever it needs from the options, and the `NSError` is only read for
            # its description above.
            compile_options_module.release_compile_options(options_id)
            if error:
                ObjC.release(int(error.value or 0))

    @property
    def id(self)->int:
        return self._library_id

    @property
    def context(self)->Any:
        return self._context

    @property
    def device(self)->Any:
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
    def structs(self)->Dict[str, Any]:
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

    def __getattr__(self, name:str)->Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.kernel_infos:
            return self.kernels[name]
        raise AttributeError(f"the program {self._file_name} declares no kernel {name!r}")


def _default_context()->Any:
    """The context a Program compiles for when the caller did not name a device."""
    from .devices import Devices
    return Devices[0].default_context
