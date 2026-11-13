from __future__ import annotations
import re
import os
import sys
import copy
import json
import pickle
from ctypes import c_char_p, c_void_p, Structure, _Pointer, pointer
from ctypes import Union as ctypes_union
from typing import Tuple, TYPE_CHECKING, Dict, List, Optional, Set, Union
from ....cparser import CPreprocessor
from ....cparser.ts_parser import TSParserMixin
from ...driver import (
    NVRTC,
    CUInfo,
    ptr_ptr_char
)
from ....utils import md5sums, save_var, modify_time, load_var
from ....kernel import structs as kernel_structs
from ..build_options import BuildOptions
from ..program_info import KernelInfo, ArgInfo, StructInfo, VarInfo
import ctypes
import numpy as np
from .... import numpy as gnp
from ....cparser.ts_parser import make_parser
from .struct_builder import _StructBuilderMixin


if TYPE_CHECKING:
    from ....cparser.ts_parser import ParseMeta, ParsedVar


try:
    import tree_sitter_cuda37 as tscuda
except ImportError:  # pragma: no cover - original distribution fallback
    import tree_sitter_cuda as tscuda


class ProgramParser(_StructBuilderMixin, TSParserMixin):

    
    @property
    def n_options(self)->int:
        return self._n_options

    _parser = make_parser(tscuda.language())


    def __init__(self):
        self._file_name:str = ""
        self._cache_folder:str = ""
        self._clean_code:str = ""
        self._includes:List[str] = []
        self._defines:Dict[str, object] = {}
        self._options:BuildOptions = BuildOptions()
        self._options_ptr:Optional[ptr_ptr_char]=None
        self._n_options:int = 0
        self._line_map:Dict[int, str] = {}
        self._related_files:Set[str] = set()
        self._kernel_infos:Dict[str, KernelInfo] = {}
        self._struct_infos:Dict[str, StructInfo] = {}
        self._struct_types:Dict[str, type] = {}
        self._newest_mtime:Union[float, bool] = False


    def parse(self, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, object]]=None, options:Optional[BuildOptions]=None)->Union[float, bool]:
        self._file_name:str = file_name
        self._base_name:str = os.path.basename(file_name)
        self._cache_folder:str = (os.path.normcase(os.path.dirname(os.path.abspath(file_name)))
                                  .replace("\\", "/") + "/__cucache__")
        self._includes:List[str] = copy.deepcopy(includes) if includes is not None else []
        self._defines:Dict[str, object] = copy.deepcopy(defines) if defines is not None else {}
        self._options:BuildOptions = copy.deepcopy(options) if options is not None else BuildOptions()
        options_list:List[str] = self._options.to_list()

        self._n_options:int = len(options_list)
        self._options_ptr:ptr_ptr_char = (c_char_p * self._n_options)(*[option.encode("utf-8") + b"\0" for option in options_list])

        module_path:str = ""
        for path in sys.path:
            if not path:
                continue

            module_file_path:str = self._file_name
            pos_point:int = module_file_path.rfind(".")
            if pos_point != -1:
                module_file_path = module_file_path[:pos_point]

            try:
                module_path:str = os.path.relpath(module_file_path, path).replace("\\", "/")
            except ValueError:
                continue

            if module_path.startswith("./"):
                module_path = module_path[2:]

            if not module_path.startswith("../"):
                break

        self._module_path:str = module_path.replace("/", ".")

        try:
            newest_mtime = self._load()
        except:
            newest_mtime = False

        if not newest_mtime:
            if NVRTC.print_info:
                print(f"parsing {self.base_name} ... ", end="", flush=True)

            self._preprocess(file_name, includes, defines)
            self._parse()
            self._save()
            if NVRTC.print_info:
                print("done. ", flush=True)
        else:
            if NVRTC.print_info:
                print(f"load {self.base_name}'s meta info from cache.", flush=True)

        self._newest_mtime = newest_mtime
        
        return newest_mtime


    def format_error(self, error_message:Union[str, bytes])->str:
        if isinstance(error_message, bytes):
            error_message = error_message.decode("utf-8", errors="replace")

        def replace_handler(match:re.Match):
            old_line_number:int = int(match.group(1) or match.group(2))
            mapping = self._line_map.get(old_line_number)
            if mapping is None:
                # A line NVRTC reports that the preprocessor's map does not carry
                # (the `#line 1` wrapper's own lines) stays as it was.
                return match.group(0)
            file_name, new_line_number = mapping
            return f"{file_name}:{new_line_number}"

        # NVRTC spells the program name two ways: `<kernel>(111)` -- a location in
        # parentheses -- and `<kernel>:111` (some compilers, continuation contexts).
        # The map keys on the number either way; only the spelling differs, and a
        # pattern for the colon form alone left every parenthesized location --
        # which is the ordinary one, warnings included -- untouched.
        return re.sub(r'<kernel>(?::(\d+)|\((\d+)\))', replace_handler,
                      error_message.strip("\r\n"))


    def _save(self)->None:
        meta:"ParseMeta" = {
            "clean_code": self._clean_code,
            "related_files": self._related_files,
            "line_map": self._line_map,
            "struct_infos": self._struct_infos,
            "kernel_infos": self._kernel_infos
        }
        save_var(meta, self._meta_file_name)


    def generate_pyi(self)->None:
        pyi_file_name = self._file_name.replace(".cu", ".pyi")

        if self._newest_mtime and modify_time(pyi_file_name) > self._newest_mtime:
            return

        content = """from typing import override, overload, Tuple
import ctypes
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.cuda.oop.kernel import Kernel

"""

        for struct_info in self._struct_infos.values():
            content += struct_info.declare()

        for kernel_name, kernel_info in self._kernel_infos.items():
            args_declare = kernel_info.args_declare(True)
            content += f"""
class Kernel_{kernel_name}(Kernel):

    @override
    def __call__(self, {args_declare})->None: ...

    @override
    def submit(self, {args_declare})->concurrent.futures.Future: ...

    @override
    def async_call(self, {args_declare})->asyncio.Future: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_{kernel_name}: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_{kernel_name}: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_{kernel_name}: ...

    @overload
    def __getitem__(self, device:str)->Kernel_{kernel_name}: ...

"""

        content += "\n"

        for kernel_name in self._kernel_infos:
            content += f"{kernel_name}: Kernel_{kernel_name}\n"

        with open(pyi_file_name, "w") as out_file:
            out_file.write(content)


    def _load(self)->Union[bool, float]:
        meta_mtime = modify_time(self._meta_file_name)
        if modify_time(self._file_name) > meta_mtime:
            return False
        
        newest_mtime:float = 0
        try:
            meta:"ParseMeta" = load_var(self._meta_file_name)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError):
            # No usable cache (see the OpenCL counterpart): re-parse instead of failing.
            return False
        for related_file in meta["related_files"]:
            if not os.path.isfile(related_file):
                return False

            related_file_mtime = modify_time(related_file)
            if related_file_mtime > newest_mtime:
                newest_mtime = related_file_mtime

            if related_file_mtime > meta_mtime:
                return False
            
        self._clean_code = meta["clean_code"]
        self._related_files = meta["related_files"]
        self._line_map = meta["line_map"]
        self._kernel_infos = meta["kernel_infos"]
        self._struct_infos = meta["struct_infos"]
        for struct_info in self._struct_infos.values():
            self._create_struct_type(struct_info)

        return newest_mtime


    @property
    def newest_mtime(self)->Union[float, bool]:
        return self._newest_mtime


    @property
    def file_name(self)->str:
        return self._file_name

    
    @property
    def base_name(self)->str:
        return self._base_name

    
    @property
    def includes(self)->List[str]:
        return self._includes

    
    @property
    def defines(self)->Dict[str, object]:
        return self._defines

    
    @property
    def options(self)->BuildOptions:
        return self._options

    
    @property
    def options_ptr(self)->ptr_ptr_char:
        return self._options_ptr


    @property
    def clean_code(self)->str:
        return self._clean_code


    @property
    def line_map(self)->Dict[int, str]:
        return self._line_map

    
    @property
    def related_files(self)->Set[str]:
        return self._related_files

    
    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._kernel_infos


    def _preprocess(self, file_name:str, includes:List[str], defines:Dict[str, object]):
        # The source's own directory is on the include path: a `#include "..."` in a
        # generated source names a file beside it (the package's own header lives in the
        # cache's `include/` folder, and a user's header is copied there too).
        search_paths:List[str] = list(includes) if includes else []
        source_directory:str = os.path.dirname(os.path.abspath(file_name))
        if source_directory not in search_paths:
            search_paths.append(source_directory)
        (
            self._clean_code,
            self._line_map,
            self._related_files
        ) = CPreprocessor.macros_expand_file(file_name, search_paths, defines)
        self._struct_infos.clear()
        self._struct_types.clear()
        self._kernel_infos.clear()


    def _parse(self):
        self._ts_parse()
        for struct_info in self._struct_infos.values():
            self._create_struct_type(struct_info)
        self._analyze_kernels()


    def _new_struct_info(self, name:str, is_union:bool=False)->StructInfo:
        return StructInfo(name=name, is_union=is_union)


    def _new_kernel_info(self, name:str)->KernelInfo:
        return KernelInfo(name=name)


    def _new_var_info(self, desc:"ParsedVar")->VarInfo:
        return VarInfo(
            name=desc["name"],
            type_str=desc["type_str"],
            array_shape=desc["array_shape"],
            type_qualifiers=desc["type_qualifiers"],
            kernel_flags=desc["kernel_flags"]
        )


    def _new_arg_info(self, parent:KernelInfo, desc:"ParsedVar")->ArgInfo:
        return ArgInfo(
            parent=parent,
            name=desc["name"],
            type_str=desc["type_str"],
            array_shape=desc["array_shape"],
            type_qualifiers=desc["type_qualifiers"],
            kernel_flags=desc["kernel_flags"]
        )


    def _is_mem_arg(self, arg:object)->bool:
        return arg.is_ptr


    @property
    def _meta_file_name(self)->str:
        return f"{self._cache_folder}/{self._base_name}_{self.md5}_ts2.meta"

    
    @property
    def md5(self)->str:
        # The file itself belongs to the key: several kernels share one generated
        # file (every kernel a module defines), and each is compiled on its own call.
        return ProgramParser.md5_of(self._includes, self._defines, self._options,
                                    file_name=self._file_name)

    
    @property
    def cache_folder(self)->str:
        return self._cache_folder


    @staticmethod
    def md5_of(includes:Optional[List[str]]=None, defines:Optional[Dict[str, object]]=None, options:Optional[BuildOptions]=None, file_name:str="")->str:
        if includes is None:
            includes = []

        if defines is None:
            defines = {}

        if options is None:
            options = BuildOptions()
        
        clean_includes:List[str] = []
        for include in includes:
            include = os.path.normcase(os.path.abspath(include)).replace("\\", "/")
            if include not in clean_includes:
                clean_includes.append(include)
                
        content = {
            "includes": clean_includes,
            "defines": defines,
            "options": str(options)
        }
        if file_name:
            content["file_name"] = (os.path.normcase(os.path.abspath(file_name))
                                    .replace("\\", "/"))
            # The source's *content* belongs to the key: several kernels can share one
            # generated file (every kernel a python module defines), and each is compiled
            # on its own call, so a key that named only the path would let the first
            # kernel's binary answer for the rest -- the module then lacks their symbols.
            try:
                with open(file_name, "r", encoding="utf-8") as file:
                    content["source"] = md5sums(file.read())
            except OSError:
                pass

        return md5sums(json.dumps(content, separators=(',', ':'), indent=None))


__all__ = [
    "__init__",
    "_access_qualifiers",
    "_address_qualifiers",
    "_apply_structure_pointers",
    "_create_struct_type",
    "_is_mem_arg",
    "_load",
    "_make_getter",
    "_make_setter",
    "_member_numpy_dtype",
    "_meta_file_name",
    "_new_arg_info",
    "_new_kernel_info",
    "_new_struct_info",
    "_new_var_info",
    "_parse",
    "_parser",
    "_pointer_members",
    "_preprocess",
    "_record_dtype",
    "_save",
    "_structure_bytes",
    "_type_qualifier_flags",
    "_type_qualifier_flags_of",
    "base_name",
    "cache_folder",
    "clean_code",
    "defines",
    "file_name",
    "format_error",
    "generate_pyi",
    "includes",
    "is_up_to_date",
    "kernel_infos",
    "line_map",
    "md5",
    "md5_of",
    "newest_mtime",
    "options",
    "options_ptr",
    "parse",
    "related_files",
]
