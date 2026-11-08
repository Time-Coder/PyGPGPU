# The nvcc path for a kernel that launches another kernel.
from __future__ import annotations
import os
import pathlib
from typing import (
    Dict,
    List,
    Optional,
)



def _cuda_toolkit() -> Optional[pathlib.Path]:
    """The CUDA Toolkit directory, or None when nvcc cannot be found."""
    import shutil

    for variable in ("CUDA_PATH", "CUDA_HOME"):
        value:str = os.environ.get(variable, "")
        if value:
            candidate:pathlib.Path = pathlib.Path(value) / "bin" / (
                "nvcc.exe" if os.name == "nt" else "nvcc")
            if candidate.exists():
                return candidate
    # Linux installs ship /usr/local/cuda as a symlink to the versioned toolkit;
    # checking it keeps a toolkit without nvcc on PATH discoverable.
    if os.name != "nt":
        candidate:pathlib.Path = pathlib.Path("/usr/local/cuda/bin/nvcc")
        if candidate.exists():
            return candidate

    found:Optional[str] = shutil.which("nvcc")
    return pathlib.Path(found) if found else None


def _cuda_tool(nvcc:pathlib.Path, tool:str) -> Optional[pathlib.Path]:
    """Another executable shipped beside nvcc, or None when it is not installed."""
    import shutil

    tool_name:str = tool if os.name == "nt" else pathlib.Path(tool).stem
    candidate:pathlib.Path = nvcc.parent / tool_name
    if candidate.exists():
        return candidate

    found:Optional[str] = shutil.which(tool_name)
    return pathlib.Path(found) if found else None


class _LinkedImage:
    """A loaded module plus the metadata `Kernel` needs, for an nvcc-linked source.

    `Program` normally owns an NVRTC program and reads the kernel arguments from its
    parser. A linked build comes from nvcc instead, so only the two attributes `Kernel`
    actually touches are provided: the module handle and the per-kernel argument layout,
    which is parsed from the source the same way NVRTC's parser would.
    """

    def __init__(self, module:object, kernel_infos:Dict[str, object],
                 context:object, device:object)->None:
        self.id = module
        self.kernel_infos:Dict[str, object] = kernel_infos
        # `Kernel` reaches back into its program for these during a launch, so a linked
        # build has to answer them like an NVRTC one does.
        self.context:object = context
        self.device:object = device
        self.prefix:str = "nvcc"


def _compile_linked_cuda(source_path:str, c_name:str, device_obj:object)->object:
    """Compile a source that launches a kernel from device code, through nvcc.

    NVRTC cannot produce a loadable image for such a source: the child kernel is resolved
    through the device runtime, which has to be linked in, and the only tool on this
    platform that performs that link is nvcc. Everything else keeps using NVRTC, so the
    cost -- a subprocess and a dependency on the toolkit -- is paid only here.
    """
    import subprocess
    from ctypes import byref, c_void_p, cast, create_string_buffer

    from ...cuda.driver.cuda import CUDA
    from ...cuda.oop.kernel import Kernel
    from ...cuda.oop.program_parser import ProgramParser
    from ...exceptions import CompileError

    nvcc:Optional[pathlib.Path] = _cuda_toolkit()
    if nvcc is None:
        raise SyntaxError(
            "this kernel launches another kernel from device code, which needs CUDA's "
            "dynamic parallelism and therefore a device-code link. NVRTC cannot perform "
            "that link, so nvcc is required: set CUDA_PATH to a CUDA Toolkit install (or "
            "put nvcc on PATH) and try again. Kernels that do not nest a launch keep "
            "working without it")

    arch:str = (f"sm_{device_obj.compute_capability_major}"
                f"{device_obj.compute_capability_minor}")
    image_path:pathlib.Path = pathlib.Path(source_path).with_suffix(".cubin")
    # nvcc compiles `.cu` as C++, where `__global__ void f(...)` gets a mangled symbol, and
    # the host looks the kernel up by the plain name it generated. NVRTC exposes the plain
    # one; nvcc does not, so the entry points are given C linkage here -- only for this
    # build, since a plain NVRTC source must keep the name it already resolves.
    #
    # `-cubin` is not enough: for a source that launches a kernel, nvcc leaves the parent
    # out of the image until the device link runs, and only the *host* link performs that
    # one (a `-cubin` build lists the child alone). So the tools go through a throwaway
    # executable and the device image is extracted from it, which is the one route that
    # produces a parent that `cuModuleGetFunction` can find. That needs a `main`, which the
    # generated source does not have, so the copy gets a stub -- nothing calls it.
    linked_source:pathlib.Path = image_path.with_suffix(".linked.cu")
    text:str = pathlib.Path(source_path).read_text(encoding="utf-8")
    linked_source.write_text(
        text.replace("__global__ void ", 'extern "C" __global__ void ')
        + "\n\nint main() { return 0; }\n", encoding="utf-8")
    executable:pathlib.Path = image_path.with_suffix(
        ".linked.exe" if os.name == "nt" else ".linked")
    command:List[str] = [
        str(nvcc), "-rdc=true", f"-arch={arch}", "-x", "cu",
        "-o", str(executable), str(linked_source),
    ]
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0 or not executable.exists():
        detail:str = (finished.stderr or finished.stdout or "").strip()
        raise CompileError(
            f"nvcc could not build {pathlib.Path(source_path).name} as linked device code "
            f"(it launches a kernel from device code, which needs dynamic parallelism):\n"
            f"{detail[:1200]}")

    extractor:Optional[pathlib.Path] = _cuda_tool(nvcc, "cuobjdump.exe")
    if extractor is None:
        raise CompileError(
            "cuobjdump was not found beside nvcc; it is needed to take the linked device "
            "image out of an nvcc build")
    extract_dir:pathlib.Path = image_path.with_suffix(".linked")
    extract_dir.mkdir(exist_ok=True)
    extracted = subprocess.run(
        [str(extractor), "-xelf", "all", str(executable)],
        capture_output=True, text=True, cwd=str(extract_dir))
    images:List[pathlib.Path] = sorted(extract_dir.glob("*.cubin"))
    if not images:
        detail = (extracted.stderr or extracted.stdout or "").strip()
        raise CompileError(
            f"the linked image could not be extracted from the nvcc build:\n{detail[:900]}")
    image_path.write_bytes(images[0].read_bytes())

    device_obj.default_context.make_current()
    module = c_void_p()
    blob:bytes = image_path.read_bytes()
    buffer = create_string_buffer(blob, len(blob))
    code:int = CUDA.cuModuleLoadData(byref(module), cast(buffer, c_void_p))
    if code != 0:
        raise CompileError(f"the linked image for {pathlib.Path(source_path).name} could "
                           f"not be loaded: {code}")

    # The argument layout comes from the same parser the NVRTC path uses, so the launch
    # binds parameters identically; only the image itself is nvcc's.
    parser:ProgramParser = ProgramParser()
    parser.parse(source_path, None, None, None)
    kernel_infos:Dict[str, object] = dict(parser.kernel_infos)
    program:object = _LinkedImage(module, kernel_infos, device_obj.default_context, device_obj)
    kernel:object = Kernel(program, c_name)
    # `Kernel` starts with an empty argument layout and a `Program` fills it while fetching
    # its kernels; a linked build has no such program, so the parsed layout is handed over.
    kernel.args = kernel_infos[c_name].args
    kernel.type_checked = False
    return kernel
