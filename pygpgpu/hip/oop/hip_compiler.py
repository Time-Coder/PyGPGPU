"""HIP source compiler: hipRTC is broken on this machine (errors with an empty log), so we spawn ROCm's own clang instead.

Compiling needs C++ standard library headers (hip's clang wrapper force-includes cstdlib etc.),
so we auto-discover the MSVC / Windows SDK include paths (any drive letter),
overridable with the PYGPGPU_HIP_INCLUDES environment variable (os.pathsep-separated).
"""
from __future__ import annotations

import glob
import os
import subprocess
from typing import Any, Dict, List, Optional

from ...utils import modify_time
from .program_parser import ProgramParser


class HipCompiler:

    print_info:bool = True
    check_error:bool = True

    _include_dirs:Optional[List[str]] = None

    def __init__(self, file_name:str="", includes:Optional[List[str]]=None,
                 defines:Optional[Dict[str, Any]]=None, options:Any=None,
                 program_parser:Optional[ProgramParser]=None):
        self._objects:Dict[str, bytes] = {}   # (md5, arch) -> code object bytes

        if program_parser is None:
            self._program_parser:ProgramParser = ProgramParser()
            self._program_parser.parse(file_name, includes, defines, options)
        else:
            self._program_parser:ProgramParser = program_parser

    # ---------- include path discovery ----------

    @classmethod
    def include_dirs(cls)->List[str]:
        if cls._include_dirs is not None:
            return cls._include_dirs

        env = os.environ.get("PYGPGPU_HIP_INCLUDES", "")
        if env:
            cls._include_dirs = [p for p in env.split(os.pathsep) if p]
            return cls._include_dirs

        drives = [f"{d}:" for d in "CDEFG" if os.path.isdir(f"{d}:/")]
        dirs:List[str] = []
        msvc_dirs:Dict[str, str] = {}
        patterns = ["%s/Microsoft/Visual Studio/*/*/VC/Tools/MSVC/*/include",
                    "%s/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/include",
                    "%s/Program Files/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/include",
                    "%s/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/include"]
        for drive in drives:
            for pattern in patterns:
                for found in glob.glob(pattern % drive):
                    msvc_dirs[drive] = found   # keep the last match per drive (highest glob lexicographic version)
        dirs.extend(msvc_dirs.values())

        sdk_dirs:Dict[str, List[str]] = {}
        for drive in drives:
            for sdkver in glob.glob(f"{drive}/Windows Kits/10/Include/*"):
                sdk_dirs.setdefault(drive, [])
                for sub in ("ucrt", "um", "shared"):
                    path = os.path.join(sdkver, sub)
                    if os.path.isdir(path):
                        sdk_dirs[drive].append(path)
        # Prefer the SDK on the same drive as MSVC
        for drive in msvc_dirs:
            if drive in sdk_dirs:
                dirs.extend(sdk_dirs.pop(drive))
        for remain in sdk_dirs.values():
            dirs.extend(remain)

        cls._include_dirs = dirs
        return dirs

    @staticmethod
    def clang_path()->str:
        from ..driver.hip import MetaHIP
        for root in MetaHIP.rocm_roots():
            candidate = os.path.join(root, "bin", "clang.exe")
            if os.path.isfile(candidate):
                return candidate
        return "clang.exe"

    # ---------- compilation ----------

    def code_object(self, device:Any)->bytes:
        """Compile for the device architecture (gfx1036 etc.) and return code object bytes hipModuleLoadData can load directly."""
        arch:str = device.arch
        parser:ProgramParser = self._program_parser
        key:str = f"{parser.md5}|{arch}"
        cached = self._objects.get(key)
        if cached is not None:
            return cached

        cache_dir:str = os.path.join(parser.cache_folder, device.unique_key)
        os.makedirs(cache_dir, exist_ok=True)
        clean_path:str = os.path.join(cache_dir, f"{parser.base_name}_{parser.md5}_clean.hip")
        obj_path:str = os.path.join(cache_dir, f"{parser.base_name}_{parser.md5}_{arch}.o")

        clean_code:str = '#include <hip/hip_runtime.h>\nextern "C" {\n#line 1\n' + parser.clean_code + "\n}"
        existing:Optional[str] = None
        if os.path.isfile(clean_path):
            with open(clean_path, "r", encoding="utf-8") as file:
                existing = file.read()
        if existing != clean_code:
            with open(clean_path, "w", encoding="utf-8", newline="\n") as file:
                file.write(clean_code)

        newest_mtime = parser.newest_mtime
        if (not os.path.isfile(obj_path)
                or (newest_mtime and modify_time(clean_path) > modify_time(obj_path))):
            # Use clang when hipRTC fails on this path: the extern "C" wrapper is already in the parser-generated clean_code
            # --cuda-device-only: emits an unbundled raw AMDGCN code object,
            # which hipModuleLoadData can load directly (-x hip defaults to a host COFF with an embedded fatbin, which reports image invalid)
            cmd:List[str] = [
                self.clang_path(),
                "--cuda-device-only",
                "-x", "hip",
                "-c", clean_path, "-o", obj_path,
                f"--offload-arch={arch}",
                "-O2", "-w",
            ]
            rocm_include = os.path.join(os.path.dirname(os.path.dirname(self.clang_path())), "include")
            if os.path.isdir(rocm_include):
                cmd.extend(["-isystem", rocm_include])
            for inc in self.include_dirs():
                cmd.extend(["-isystem", inc])

            if self.print_info:
                print(f"compiling {parser.base_name} for {arch} ... ", end="", flush=True)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                if self.print_info:
                    print("error.", flush=True)
                message = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(
                    f"HIP compilation failed for {arch}:\n{message[-2000:]}\n"
                    f"(include dirs: {self.include_dirs()}; "
                    f"set PYGPGPU_HIP_INCLUDES to override)")
            if self.print_info:
                print("done.", flush=True)

        with open(obj_path, "rb") as file:
            data:bytes = file.read()
        self._objects[key] = data
        return data

    # ---------- attributes the cuda Program expects (forwarded to the parser) ----------

    @property
    def file_name(self)->str:
        return self._program_parser.file_name

    @property
    def base_name(self)->str:
        return self._program_parser.base_name

    @property
    def parser(self)->ProgramParser:
        return self._program_parser

    @property
    def includes(self):
        return self._program_parser.includes

    @property
    def defines(self):
        return self._program_parser.defines

    @property
    def options(self):
        return self._program_parser.options

    @property
    def options_ptr(self):
        return self._program_parser.options_ptr

    @property
    def n_options(self)->int:
        return self._program_parser.n_options

    @property
    def clean_code(self)->str:
        return self._program_parser.clean_code

    @property
    def line_map(self):
        return self._program_parser.line_map

    @property
    def related_files(self):
        return self._program_parser.related_files

    @property
    def kernel_infos(self):
        return self._program_parser.kernel_infos

    @property
    def structs(self):
        return self._program_parser._struct_types

    @property
    def newest_mtime(self):
        return self._program_parser.newest_mtime
