from __future__ import annotations
import re
import os
import copy
import json
from ctypes import c_char_p, Structure, POINTER
from typing import Optional, Dict, Any, List, Set, Union, Iterator, Tuple

from ...kernel_parser import CPreprocessor
from ..runtime import (
    CL,
    cl_kernel_arg_address_qualifier,
    cl_kernel_arg_access_qualifier,
    cl_kernel_arg_type_qualifier,
    CLInfo
)
from ...utils import md5sums, save_var, modify_time, load_var
from .build_options import BuildOptions
from .kernel_info import KernelInfo, ArgInfo, StructInfo, VarInfo


class KernelParser:

    _struct_def_pattern1:re.Pattern = re.compile(r"^\s*struct\s+(?P<struct_name>[a-zA-Z_]\w*)\{(?P<body>.*?)\}\s*;", re.MULTILINE | re.DOTALL)
    _struct_def_pattern2:re.Pattern = re.compile(r"^\s*typedef\s+struct(?P<struct_name>\s+[a-zA-Z_]\w*)?\{(?P<body>.*?)\}\s*(?P<type_name>[a-zA-Z_]\w*)\s*;", re.MULTILINE | re.DOTALL)
    _kernel_def_pattern:re.Pattern = re.compile(r"^\s*(__kernel|kernel)\s+void\s+(?P<name>[a-zA-Z_]\w*)\((?P<body>.*?)\)", re.MULTILINE | re.DOTALL)
    _address_qualifier_pattern:re.Pattern = re.compile(r"\b(__global|__local|__private|__constant|global|local|private|constant)\b")
    _access_qualifier_pattern:re.Pattern = re.compile(r"\b(__read_only|__write_only|__read_write|read_only|write_only|read_write)\b")
    _type_qualifier_pattern:re.Pattern = re.compile(r"\b(const|restrict|volatile|pipe)\b")
    _arg_name_pattern:re.Pattern = re.compile(r"\b([a-zA-Z_]\w*(\[[^\]]*\])*)(?=\W*$)")
    _array_shape_pattern:re.Pattern = re.compile(r'\[\s*(\d+)\s*\]')

    def __init__(self):
        self._file_name:str = ""
        self._cache_folder:str = ""
        self._clean_code:str = ""
        self._includes:List[str] = []
        self._defines:Dict[str, Any] = {}
        self._options:BuildOptions = BuildOptions()
        self._options_ptr:c_char_p = c_char_p(str(self._options).encode("utf-8"))
        self._line_map:Dict[int, str] = {}
        self._related_files:Set[str] = set()
        self._kernel_infos:Dict[str, KernelInfo] = {}
        self._struct_infos:Dict[str, StructInfo] = {}
        self._newest_mtime:Union[float, bool] = False

    def parse(self, file_name:str, includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None)->Union[float, bool]:
        self._file_name:str = file_name
        self._base_name:str = os.path.basename(file_name)
        self._cache_folder:str = os.path.dirname(os.path.abspath(file_name)).replace("\\", "/") + "/__clcache__"
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

        self._newest_mtime = newest_mtime
        
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
            "struct_infos": self._struct_infos,
            "kernel_infos": self._kernel_infos
        }
        save_var(meta, self._meta_file_name)

    def generate_pyi(self)->None:
        pyi_file_name = self._file_name.replace(".cl", ".pyi")

        if self._newest_mtime and modify_time(pyi_file_name) > self._newest_mtime:
            return

        content = """from typing import override, overload, Tuple
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.opencl import *

"""

        for kernel_name, kernel_info in self._kernel_infos.items():
            args_declare = kernel_info.args_declare(True)
            content += f"""
class Kernel_{kernel_name}(KernelWrapper):

    @override
    def __call__(self, {args_declare})->None: ...

    @override
    def submit(self, {args_declare})->concurrent.futures.Future: ...
    
    @override
    def async_call(self, {args_declare})->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_{kernel_name}: ...

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
        self._struct_infos.clear()
        self._kernel_infos.clear()

    def _create_struct_type(self, struct_info:StructInfo)->type:
        if struct_info.struct_type is not None:
            return struct_info.struct_type

        fields = []
        dtype = []
        for member in struct_info.members.values():
            if member.type_str in CLInfo.basic_types:
                fields.append((member.name, CLInfo.basic_types[member.type_str]))
                dtype.append((member.name, CLInfo.basic_types[member.type_str]))
            elif member.is_ptr and member.base_type_str in CLInfo.basic_types:
                fields.append((member.name, POINTER(CLInfo.basic_types[member.type_str])))
            else:
                struct_type = self._create_struct_type(self._struct_infos[member.base_type_str])
                if member.is_ptr:
                    fields.append((member.name, POINTER(struct_type)))
                else:
                    fields.append((member.name, struct_type))
                    dtype.append((member.name, struct_type.dtype))

        struct_info.struct_type = type(struct_info.name, (Structure,), {
            "_fields_": fields,
            "dtype": dtype
        })
        return struct_info.struct_type

    def _parse(self):
        struct_matches:List[re.Match] = self._find_struct_defs(self._clean_code)
        for struct_match in struct_matches:
            if "type_name" in struct_match:
                struct_name:str = struct_match["type_name"]
            else:
                struct_name:str = "struct " + struct_match["struct_name"]
            struct_info = StructInfo(name=struct_name)
            self._struct_infos[struct_name] = struct_info

            members_str:str = struct_match["body"]
            members_items:List[str] = members_str.split(";")
            for member_item in members_items:
                member_item = member_item.strip("\r\n\t ")
                submember_items = member_item.split(",")
                first_member_item = submember_items[0]
                address_qualifier = self._find_address_qualifier(first_member_item)
                access_qualifier = self._find_access_qualifier(first_member_item)
                type_qualifiers = self._find_type_qualifiers(first_member_item)
                member_name:str = self._find_arg_name(first_member_item)
                member_type:str = self._find_arg_type(first_member_item)
                member_name, member_shape = self._extract_array_shape(member_name)
                struct_info[member_name] = VarInfo(
                    name=member_name,
                    type=member_type,
                    array_shape=member_shape,
                    address_qualifier=address_qualifier,
                    access_qualifier=access_qualifier,
                    type_qualifiers=type_qualifiers
                )
                for i in range(1, len(submember_items)):
                    member_name:str = self._find_arg_name(submember_items[i]).split(",")
                    struct_info[member_name] = VarInfo(
                        name=member_name,
                        type=member_type,
                        address_qualifier=address_qualifier,
                        access_qualifier=access_qualifier,
                        type_qualifiers=type_qualifiers
                    )

        for struct_info in self._struct_infos.values():
            self._create_struct_type(struct_info)

        kernel_matches:Iterator[re.Match] = self._find_kernel_defs(self._clean_code)
        for kernel_match in kernel_matches:
            kernel_name:str = kernel_match["name"]
            kernel_info = KernelInfo(name=kernel_name)
            self._kernel_infos[kernel_name] = kernel_info

            args_str:str = kernel_match["body"]
            args_items:List[str] = args_str.split(",")
            for arg_item in args_items:
                arg_item = arg_item.strip("\r\n\t ")
                address_qualifier = self._find_address_qualifier(arg_item)
                access_qualifier = self._find_access_qualifier(arg_item)
                type_qualifiers = self._find_type_qualifiers(arg_item)
                arg_name:str = self._find_arg_name(arg_item)
                arg_type:str = self._find_arg_type(arg_item)
                arg_name, arg_shape = self._extract_array_shape(arg_name)
                kernel_info.args[arg_name] = ArgInfo(
                    parent=kernel_info,
                    name=arg_name,
                    type_str=arg_type,
                    array_shape=arg_shape,
                    address_qualifier=address_qualifier,
                    access_qualifier=access_qualifier,
                    type_qualifiers=type_qualifiers
                )
    
    @property
    def _meta_file_name(self)->str:
        return f"{self._cache_folder}/{self._base_name}_{self.md5}.meta"
    
    @property
    def md5(self)->str:
        return KernelParser.md5_of(self._includes, self._defines, self._options)
    
    @property
    def cache_folder(self)->str:
        return self._cache_folder

    @staticmethod
    def md5_of(includes:Optional[List[str]]=None, defines:Optional[Dict[str, Any]]=None, options:Optional[BuildOptions]=None, file_name:str="")->str:
        if includes is None:
            includes = []

        if defines is None:
            defines = {}

        if options is None:
            options = BuildOptions()
        
        clean_includes:List[str] = []
        for include in includes:
            include = os.path.abspath(include).replace("\\", "/")
            if include not in clean_includes:
                clean_includes.append(include)
                
        content = {
            "includes": clean_includes,
            "defines": defines,
            "options": str(options)
        }
        if file_name:
            content["file_name"] = os.path.abspath(file_name).replace("\\", "/")

        return md5sums(json.dumps(content, separators=(',', ':'), indent=None))

    def _find_struct_defs(self, code:str)->List[re.Match]:
        structs = list(self._struct_def_pattern1.finditer(code))
        structs.extend(list(self._struct_def_pattern2.finditer(code)))
        return structs

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
    
    def _extract_array_shape(self, content: str)->Tuple[str, Tuple[int, ...]]:
        end_pos = content.find("[")
        if end_pos == -1:
            name = content
        else:
            name = content[:end_pos].strip()

        matches = re.findall(self._array_shape_pattern, content)
        
        if not matches:
            shape = ()
        else:
            shape = tuple(int(x) for x in matches)

        return name, shape
        