from __future__ import annotations
import re
import os
import copy
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from ctypes import c_char_p, c_void_p
from typing import Optional, Dict, Any, List, Set, Union, Iterator, TYPE_CHECKING, Tuple

from ...kernel_parser import CPreprocessor
from ..runtime import (
    CL,
    cl_kernel_arg_address_qualifier,
    cl_kernel_arg_access_qualifier,
    cl_kernel_arg_type_qualifier,
    cl_mem_flags
)
from ...utils import md5sums, save_var, modify_time, load_var
from .build_options import BuildOptions

if TYPE_CHECKING:
    from .buffer import Buffer
    from .command_queue import CommandQueue


class ArgInfo:

    def __init__(self, name: str, type_str: str, address_qualifier: cl_kernel_arg_address_qualifier, access_qualifier: cl_kernel_arg_access_qualifier, type_qualifiers: cl_kernel_arg_type_qualifier):
        self.name = name
        self.type_str = type_str
        self.address_qualifier = address_qualifier
        self.access_qualifier = access_qualifier
        self.type_qualifiers = type_qualifiers
        self.value: Any = None
        self.buffer: Optional[Buffer] = None
        self.__buffers: Dict[Tuple[int, cl_mem_flags], List[Buffer]] = defaultdict(list)
        self.__busy_buffers: Set[Buffer] = set()

    @property
    def is_ptr(self)->bool:
        return (self.type_str[-1] == "*")
    
    @property
    def base_type_str(self)->str:
        return (self.type_str[:-1] if self.is_ptr else self.type_str)
    
    @property
    def need_read_back(self)->bool:
        return (
            self.is_ptr and
            self.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_GLOBAL and
            self.access_qualifier != cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_READ_ONLY and
            not (self.type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_CONST)
        )
    
    @property
    def readonly(self)->bool:
        return (
            self.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_CONSTANT or
            self.access_qualifier == cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_READ_ONLY or
            self.type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_CONST
        )

    def use_buffer(self, data:np.ndarray, cmd_queue:CommandQueue):
        if self.readonly:
            flags = cl_mem_flags.CL_MEM_READ_ONLY
        else:
            flags = cl_mem_flags.CL_MEM_READ_WRITE

        buffer_key = (data.nbytes, flags)

        used_buffer = None
        for buffer in self.__buffers[buffer_key]:
            if buffer not in self.__busy_buffers:
                used_buffer = buffer
                break

        if used_buffer is None:
            used_buffer = cmd_queue.context.create_buffer(data, flags)
            self.__buffers[buffer_key].append(used_buffer)
        else:
            used_buffer.set_data(data, cmd_queue)

        self.__busy_buffers.add(used_buffer)
        return used_buffer
    
    def unuse_buffer(self, buffer:Buffer):
        self.__busy_buffers.remove(buffer)


class KernelInfo:

    def __init__(self, name:str):
        self.name: str = name
        self.args: Dict[str, ArgInfo] = {}


class KernelParser:

    __kernel_def_pattern:Optional[re.Pattern] = None
    __address_qualifier_pattern:Optional[re.Pattern] = None
    __access_qualifier_pattern:Optional[re.Pattern] = None
    __type_qualifier_pattern:Optional[re.Pattern] = None
    __arg_name_pattern:Optional[re.Pattern] = None

    def __init__(self):
        self._file_name:str = ""
        self._clean_code:str = ""
        self._includes:List[str] = []
        self._defines:Dict[str, Any] = {}
        self._options:BuildOptions = BuildOptions()
        self._options_ptr:c_char_p = c_char_p(str(self._options).encode("utf-8"))
        self._line_map:Dict[int, str] = {}
        self._related_files:Set[str] = set()
        self._kernel_infos:Dict[str, KernelInfo] = {}

    def parse(self, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None)->Union[float, bool]:
        self._file_name:str = file_name
        self._base_name:str = os.path.basename(file_name)
        self._includes:List[str] = copy.deepcopy(includes) if includes is not None else []
        self._defines:Dict[str, Any] = copy.deepcopy(defines) if defines is not None else {}
        self._options:Dict[str, Any] = copy.deepcopy(options) if options is not None else BuildOptions()
        self._options_ptr:c_char_p = c_char_p(str(self._options).encode("utf-8"))

        try:
            newest_mtime = self._load()
        except:
            newest_mtime = False

        if not newest_mtime:
            if CL.print_info:
                print(f"parsing {self.base_name} ... ", end="", flush=True)

            self._preprocess(file_name, includes, defines)
            self._parse()
            self._save()

            if CL.print_info:
                print(f"done. ", flush=True)
        else:
            if CL.print_info:
                print(f"load {self.base_name}'s meta info from cache.", flush=True)

        return newest_mtime

    def format_error(self, error_message:str)->str:
        def replace_handler(match:re.Match):
            old_line_number = int(match.group(1))
            file_name, new_line_number = self._line_map[old_line_number]
            return f"{file_name}:{new_line_number}"

        return re.sub(r'<kernel>:(\d+)', replace_handler, error_message.strip("\r\n"))

    def _save(self)->None:
        meta:Dict[str, Any] = {
            "clean_code": self._clean_code,
            "related_files": self._related_files,
            "line_map": self._line_map,
            "kernel_infos": self._kernel_infos
        }
        save_var(meta, self._meta_file_name)

    def _load(self)->Union[bool, float]:
        meta_mtime = modify_time(self._meta_file_name)
        if modify_time(self._file_name) > meta_mtime:
            return False
        
        newest_mtime:float = 0
        meta:Dict[str, Any] = load_var(self._meta_file_name)
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
        return newest_mtime

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
    def defines(self)->Dict[str, Any]:
        return self._defines
    
    @property
    def options(self)->BuildOptions:
        return self._options
    
    @property
    def options_ptr(self)->c_char_p:
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

    def _preprocess(self, file_name:str, includes:List[str], defines:Dict[str, Any]):
        (
            self._clean_code,
            self._line_map,
            self._related_files
        ) = CPreprocessor.macros_expand_file(file_name, includes, defines)
        self._kernel_infos.clear()

    def _parse(self):
        kernel_matches:List[re.Match] = self._find_kernel_defs(self._clean_code)
        for kernel_match in kernel_matches:
            kernel_name:str = kernel_match.group(1)
            self._kernel_infos[kernel_name] = KernelInfo(name=kernel_name)

            args_str:str = kernel_match.group(2)
            args_items:List[str] = args_str.split(",")
            for arg_item in args_items:
                arg_item = arg_item.strip("\r\n\t ")
                address_qualifier = self._find_address_qualifier(arg_item)
                access_qualifier = self._find_access_qualifier(arg_item)
                type_qualifiers = self._find_type_qualifiers(arg_item)
                arg_name:str = self._find_arg_name(arg_item)
                arg_type:str = self._find_arg_type(arg_item)
                self._kernel_infos[kernel_name].args[arg_name] = ArgInfo(
                    name=arg_name,
                    type_str=arg_type,
                    address_qualifier=address_qualifier,
                    access_qualifier=access_qualifier,
                    type_qualifiers=type_qualifiers
                )
    
    @property
    def _cache_folder(self)->str:
        self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        return self_folder + "/__clcache__"
    
    @property
    def _meta_file_name(self)->str:
        return f"{self._cache_folder}/{self._base_name}_{self.md5}.meta"
    
    @property
    def md5(self)->str:
        return KernelParser.md5_of(self._file_name, self._includes, self._defines, self._options)

    @staticmethod
    def md5_of(file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None)->str:
        if includes is None:
            includes = []

        if defines is None:
            defines = {}

        if options is None:
            options = BuildOptions()
        
        file_name = os.path.abspath(file_name).replace("\\", "/")
        clean_includes:List[str] = []
        for include in includes:
            include = os.path.abspath(include).replace("\\", "/")
            if include not in clean_includes:
                clean_includes.append(include)
                
        content = {
            "file_name": file_name,
            "includes": clean_includes,
            "defines": defines,
            "options": str(options)
        }
        return md5sums(json.dumps(content, separators=(',', ':'), indent=None))

    @property
    def _kernel_def_pattern(self)->re.Pattern:
        if KernelParser.__kernel_def_pattern is None:
            KernelParser.__kernel_def_pattern = re.compile(r"^\s*__kernel\s+void\s+([a-zA-Z_]\w*)\((.*?)\)", re.MULTILINE | re.DOTALL)

        return KernelParser.__kernel_def_pattern
    
    @property
    def _address_qualifier_pattern(self)->re.Pattern:
        if KernelParser.__address_qualifier_pattern is None:
            KernelParser.__address_qualifier_pattern = re.compile(r"\b(__global|__local|__private|__constant|global|local|private|constant)\b")

        return KernelParser.__address_qualifier_pattern
    
    @property
    def _access_qualifier_pattern(self)->re.Pattern:
        if KernelParser.__access_qualifier_pattern is None:
            KernelParser.__access_qualifier_pattern = re.compile(r"\b(__read_only|__write_only|__read_write|read_only|write_only|read_write)\b")

        return KernelParser.__access_qualifier_pattern
    
    @property
    def _type_qualifier_pattern(self)->re.Pattern:
        if KernelParser.__type_qualifier_pattern is None:
            KernelParser.__type_qualifier_pattern = re.compile(r"\b(const|restrict|volatile|pipe)\b")

        return KernelParser.__type_qualifier_pattern
    
    @property
    def _arg_name_pattern(self)->re.Pattern:
        if KernelParser.__arg_name_pattern is None:
            KernelParser.__arg_name_pattern = re.compile(r"\b([a-zA-Z_]\w*)\b(?=\W*$)")

        return KernelParser.__arg_name_pattern

    def _find_kernel_defs(self, code:str)->Iterator[re.Match]:
        return self._kernel_def_pattern.finditer(code)
    
    def _find_address_qualifier(self, arg_code:str)->cl_kernel_arg_address_qualifier:
        address_qualifier = cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_PRIVATE
        match:re.Match = self._address_qualifier_pattern.search(arg_code)
        if match:
            match_str:str = match.group(1)
            match_str:str = match_str.replace("__", "")
            address_qualifier = getattr(cl_kernel_arg_address_qualifier, f"CL_KERNEL_ARG_ADDRESS_{match_str.upper()}")

        return address_qualifier
    
    def _find_access_qualifier(self, arg_code:str)->cl_kernel_arg_access_qualifier:
        access_qualifier = cl_kernel_arg_access_qualifier.CL_KERNEL_ARG_ACCESS_NONE
        match:re.Match = self._access_qualifier_pattern.search(arg_code)
        if match:
            match_str:str = match.group(1)
            match_str:str = match_str.replace("__", "")
            access_qualifier = getattr(cl_kernel_arg_access_qualifier, f"CL_KERNEL_ARG_ACCESS_{match_str.upper()}")

        return access_qualifier
    
    def _find_type_qualifiers(self, arg_code:str)->cl_kernel_arg_type_qualifier:
        type_qualifiers = cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_NONE
        matches:Iterator[re.Match] = self._type_qualifier_pattern.finditer(arg_code)
        for match in matches:
            match_str:str = match.group(1)
            type_qualifiers |= getattr(cl_kernel_arg_type_qualifier, f"CL_KERNEL_ARG_TYPE_{match_str.upper()}")

        return type_qualifiers
    
    def _find_arg_name(self, arg_code:str)->str:
        arg_name:str = ""
        match:re.Match = self._arg_name_pattern.search(arg_code)
        if match:
            arg_name:str = match.group(1)

        return arg_name
    
    def _find_arg_type(self, arg_code:str)->str:
        arg_code = self._address_qualifier_pattern.sub("", arg_code)
        arg_code = self._access_qualifier_pattern.sub("", arg_code)
        arg_code = self._type_qualifier_pattern.sub("", arg_code)
        arg_code = self._arg_name_pattern.sub("", arg_code)
        arg_code = re.sub(r"\s{2,}", " ", arg_code)
        arg_code = re.sub(r"\s+\*", "*", arg_code)
        arg_code = arg_code.strip("\r\t\n ")
        return arg_code