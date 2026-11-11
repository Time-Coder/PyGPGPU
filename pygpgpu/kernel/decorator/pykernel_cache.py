# The cache folder, the backend compile, and the launch-argument helpers.
from __future__ import annotations
import os
from typing import (
    List,
    Optional,
    Tuple,
)
from ..translator import (
    is_ptr,
)
from .. import source_paths
from .cuda_link import (
    _compile_linked_cuda,
)
from .host_regions import (
    _declares_a_magic_loop,
    _positional_argument_names,
)


class _PykernelCacheMixin:

    def _cache_dir(self, backend:str)->str:
        """The pykernels folder this kernel's generated sources live in.

        It is the shared root the per-module files sit under, so the quaternion header can
        be placed there once and every module's source reach it.
        """
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        return source_paths.cache_root(source_file, backend)


    def _backend_compile(self, source_path:str, c_name:str, backend:str, device_obj:object,
                         linked:bool=False):
        if backend == "opencl":
            from ...opencl.oop.program import Program
            program:object = Program(device_obj.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        if backend == "hip":
            from ...hip.oop.program import Program as HIPProgram
            from ...hip.oop.hip_compiler import HipCompiler
            device_obj.default_context.make_current()
            compiler = HipCompiler(source_path)
            program:object = HIPProgram(source_path, compiler=compiler)
            return program.kernels[c_name]

        if backend == "metal":
            from ...metal.oop.program import Program as MetalProgram
            program:object = MetalProgram(source_path, device=device_obj)
            return program.kernels[c_name]

        from ...cuda.oop.program import Program as CUProgram

        device_obj.default_context.make_current()
        if linked:
            # A kernel that launches another kernel needs *linked* device code: the child is
            # resolved through the device runtime, which NVRTC does not link in (and cuLink
            # on Windows will not read cudadevrt.lib). nvcc performs that link, so only such
            # a source pays for it; everything else stays on NVRTC.
            return _compile_linked_cuda(source_path, c_name, device_obj)

        program:object = CUProgram(source_path)
        return program.kernels[c_name]


    def _build_launch_args(self, args:List[object], arg_types:List[object], shape_sources:List[Tuple[int, int]])->List[object]:
        launch_args:List[object] = list(args)
        for param_index, dim in shape_sources:
            launch_args.append(int(args[param_index].shape[dim]))
        return launch_args


    @staticmethod
    def _source_files_changed(launch_meta:object)->bool:
        """Whether a generated file this entry was built from changed on disk.

        The in-process `_cache` outlives the compile, so a file damaged *after* the entry
        was built -- a previous package version's shape, a crashed write, a truncated
        copy, an indexer's partial save -- would be served by every later launch forever:
        the digest sidecar describes the python, not the generated text, so it cannot see
        any of these. The entry records one stat per file it was built from; a change
        invalidates the entry and the next launch re-renders, whose merge (and whose
        build-failure rewrite, see _compile_kernels) restores the current shape.
        """
        mtimes:object = getattr(launch_meta, "get", lambda key, default=None: None)("source_mtimes", None)
        if not mtimes:
            return False
        for path, recorded in mtimes.items():
            try:
                if os.stat(path).st_mtime_ns != recorded:
                    return True
            except OSError:
                return True
        return False


    def _bind_images(self, full_values:List[object], backend:str)->None:
        """Create the device objects for every Image argument, before compilation.

        Each backend decides what an image becomes (a texture for reading, a surface
        for writing); the objects are cached on the Image so later launches reuse
        them. Only CUDA needs this ahead of compilation; OpenCL images are created
        per launch without that ordering constraint.
        """
        from ...image import Image

        for value in full_values:
            if not isinstance(value, Image):
                continue
            binding = value._bindings.get(backend)
            if binding is not None:
                continue
            if backend == "cuda":
                value.bind_cuda()


    def _apply_declared_domain(self, translator:object)->None:
        """Record the array `@kernel(domain=...)` named as the iteration domain.

        The parameter has to exist and be an array: only a pointer carries the shape
        dimensions a launch is derived from, and a scalar has none to give. Naming a
        nonexistent parameter is the failure a string-valued decorator argument invites --
        renaming the parameter leaves the string behind -- so it is reported with the ones
        that do exist.

        A magic loop already fixes the domain, so a kernel carrying both would have two
        declarations of the same thing; that is checked here before translation, because the
        loop scan is what records `magic_loop_present`.
        """
        name:str = self._domain
        parameter_names:List[str] = _positional_argument_names(self.raw_func)
        if name not in parameter_names:
            available:str = ", ".join(parameter_names) or "none"
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) names no parameter of this "
                f"function (its parameters are: {available}). The domain is the array whose "
                f"shape declares the iteration domain, spelled as the parameter's own name")
        if _declares_a_magic_loop(self.raw_func):
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) declares the iteration domain, "
                f"but the body already declares one with a magic loop "
                f"(`for i in range(x.shape[0])` or `np.ndindex`); keep exactly one of them")
        index:Optional[int] = None
        for position, (parameter, _) in enumerate(translator.params):
            if parameter == name:
                index = position
                break
        if index is None or not is_ptr(translator.params[index][1]):
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) requires an array parameter; "
                f"'{name}' is not an array, so it has no shape to derive a work size from")
        translator.declared_work_arrays = [name]
