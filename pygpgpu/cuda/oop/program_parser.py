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

if TYPE_CHECKING:
    from ...cparser.ts_parser import ParseMeta, ParsedVar

from ...cparser import CPreprocessor
from ...cparser.ts_parser import TSParserMixin
from ..driver import (
    NVRTC,
    CUInfo,
    ptr_ptr_char
)
from ...utils import md5sums, save_var, modify_time, load_var
from ...kernel import structs as kernel_structs
from .build_options import BuildOptions
from .program_info import KernelInfo, ArgInfo, StructInfo, VarInfo

import ctypes

import numpy as np
from ... import numpy as gnp

# Prefer the dual-version tree-sitter-cuda37 distribution: it returns a native
# tree_sitter.Language on both the legacy tree-sitter 0.20.x line (Python 3.7)
# and modern 0.21+, whereas the original tree-sitter-cuda's language() yields a
# raw PyCapsule that newer tree-sitter's Parser() rejects. Fall back to it when
# the cuda37 package is not installed.
try:
    import tree_sitter_cuda37 as tscuda
except ImportError:  # pragma: no cover - original distribution fallback
    import tree_sitter_cuda as tscuda

from ...cparser.ts_parser import make_parser


class ProgramParser(TSParserMixin):

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
            old_line_number = int(match.group(1))
            file_name, new_line_number = self._line_map[old_line_number]
            return f"{file_name}:{new_line_number}"

        return re.sub(r'<kernel>:(\d+)', replace_handler, error_message.strip("\r\n"))

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
    def n_options(self)->int:
        return self._n_options

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

    @staticmethod
    def _make_getter(name:str):
        def getter(self):
            return getattr(self, f"_{self.__class__.__name__}_data_{name}")
        
        return getter
    
    @staticmethod
    def _make_setter(name:str):
        def setter(self, value):
            setattr(self, f"_{self.__class__.__name__}_data_{name}", value)

        return setter
    
    @staticmethod
    def _structure_bytes(value)->bytes:
        """The raw bytes of one record value, from any of the shapes a caller passes.

        A `ctypes` instance converts through `bytes()`, which is what backs it. A numpy
        record does not: `bytes(record)` goes through the buffer protocol, and a dtype whose
        members overlap (a union layout) or whose offsets do not ascend is refused there with
        "dtypes with overlapping or out-of-order fields are not representable as buffers".
        `tobytes()` copies the element's raw memory instead, so it answers for every layout.
        """
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, np.void):
            return value.tobytes()
        return bytes(value)

    @staticmethod
    def _apply_structure_pointers(value, cmd_queue):
        if isinstance(value, Structure) and hasattr(value, "apply_pointers"):
            value.apply_pointers(cmd_queue)
            return
        
        if isinstance(value, (list, tuple)):
            for sub_value in value:
                ProgramParser._apply_structure_pointers(sub_value, cmd_queue)

    @staticmethod
    def _record_dtype(entries:List[object], is_union:bool)->np.dtype:
        """The numpy record for one parsed declaration, laid out by its kind.

        A structure's members follow one another, which is what `np.dtype(entries)` already
        does. A union's do not: they all sit at offset zero and the record is as wide as the
        widest member. `np.dtype` sorts a field list by member name, so the offsets cannot be
        inferred from it -- they have to be stated, which is the only way a hand-built record
        can express a union (see `structs.is_union_dtype`, which reads them back).
        """
        if not is_union:
            return np.dtype(entries)
        names:List[str] = []
        formats:List[object] = []
        widest:int = 0
        for entry in entries:
            name, base = entry[0], np.dtype(entry[1])
            shape:Tuple[int, ...] = entry[2] if len(entry) > 2 else ()
            count:int = 1
            for extent in shape or ():
                count *= int(extent)
            names.append(name)
            formats.append((base, shape) if shape else base)
            widest = max(widest, int(base.itemsize) * count)
        return np.dtype({"names": names, "formats": formats,
                         "offsets": [0] * len(names), "itemsize": widest})


    @staticmethod
    def _member_numpy_dtype(base_type:object)->object:
        """The numpy dtype describing one member's storage.

        A class-level `dtype` is the answer only when it really is a dtype: a class that
        declares `dtype` as a property (pygf's quaternion and complex records do) hands back
        the descriptor on attribute access, and numpy rejects it. The ctypes class itself
        describes the layout, so that is the fallback.
        """
        candidate:object = getattr(base_type, "dtype", None)
        if isinstance(candidate, np.dtype):
            return candidate
        if isinstance(base_type, type) and issubclass(base_type, ctypes.Structure):
            return np.dtype(base_type)
        return base_type

    def _create_struct_type(self, struct_info:StructInfo)->type:
        struct_name:str = struct_info.name
        if struct_name in self._struct_types:
            return self._struct_types[struct_name]
        
        _fields_ = []
        dtype = []
        pointer_types = {}
        for member in struct_info.members.values():
            fixed_class:Optional[type] = kernel_structs.fixed_member_class(
                member.base_type_str)
            if fixed_class is not None:
                # A quaternion or a complex value: its declaration lives in a fixed header,
                # so it is not one of the parsed structures and has no entry in the type
                # tables -- the class the header's layout describes is the member's type.
                base_type = fixed_class
            elif member.base_type_str in CUInfo.basic_types:
                base_type = CUInfo.basic_types[member.base_type_str]
            elif member.base_type_str in self._struct_infos:
                # A structure the source declared wins over a vector of the same
                # layout: the dtypes are equal, but the alignments are not, and a
                # padded structure no longer matches the host's packed bytes.
                base_type = self._create_struct_type(
                    self._struct_infos[member.base_type_str])
            elif member.base_type_str in CUInfo.vec_types:
                # A vector member is a type of its own, not a registered structure: its
                # numpy dtype describes it for the host-side field list.
                base_type = CUInfo.vec_types[member.base_type_str]
            else:
                base_type = self._create_struct_type(self._struct_infos[member.base_type_str])

            member_type = base_type
            for n_elements in reversed(member.array_shape):
                member_type *= n_elements

            if member.is_ptr:
                _fields_.append((f"_{struct_info.name}_ptr_{member.name}", c_void_p))
                dtype.append((member.name, np.uint64))
                pointer_types[member.name] = member_type
            else:
                _fields_.append((member.name, member_type))
                # A vector member contributes its structured dtype (a ctypes class has no
                # numpy dtype of its own), so the record layout matches the C structure.
                dtype.append((member.name, self._member_numpy_dtype(base_type),
                              member.array_shape))

        dtype = self._record_dtype(dtype, struct_info.is_union)

        def __init__(self, *args, **kwargs):
            for field in self._fields_:
                if issubclass(field[1], _Pointer):
                    ori_name:str = field[0].replace(f"_{self.__class__.__name__}_ptr_", "")
                    setattr(self, f"_{self.__class__.__name__}_data_{ori_name}", None)

            used_kwargs = {}
            for i in range(len(args)):
                used_kwargs[_fields_[i][0]] = args[i]

            used_kwargs.update(kwargs)
            keys = list(used_kwargs.keys())
            for key in keys:
                if self._members[key].is_ptr:
                    setattr(self, f"_{self.__class__.__name__}_data_{key}", used_kwargs[key])
                    del used_kwargs[key]

            # `ctypes.Structure.__init__` only accepts a Structure instance, so a class
            # built on the Union base has to store the members through attribute sets --
            # one member per keyword, which is exactly what the structure branch does
            # through its constructor.
            if issubclass(type(self), Structure):
                Structure.__init__(self, **used_kwargs)
            else:
                for key, item in used_kwargs.items():
                    setattr(self, key, item)

        def apply_pointers(self, cmd_queue)->List[Dict[str, object]]:
            result = []
            for field in self._fields_:
                if issubclass(field[1], _Pointer):
                    ori_name:str = field[0].replace(f"_{self.__class__.__name__}_ptr_", "")
                    data_name:str = f"_{self.__class__.__name__}_data_{ori_name}"
                    data = getattr(self, data_name)
                    var_info:VarInfo = self._members[ori_name]
                    content_type = self._pointer_types[ori_name]
                    
                    if isinstance(data, np.ndarray):
                        used_value = data
                        if data.dtype != content_type:
                            used_value = data.astype(content_type)
                    else:
                        ProgramParser._apply_structure_pointers(data, cmd_queue)
                        used_value = np.array(data, dtype=content_type)

                    if not used_value.flags['C_CONTIGUOUS']:
                        used_value = np.ascontiguousarray(used_value)

                    buffer, event = var_info.use_buffer(cmd_queue, used_value)
                    if var_info.need_read_back and not isinstance(data, gnp.ndarray):
                        result.append({
                            "event": event,
                            "arg_info": var_info,
                            "value": data,
                            "mem_obj": buffer
                        })
                    else:
                        result.append({
                            "event": event
                        })

                    setattr(self, f"_{self.__class__.__name__}_ptr_{ori_name}", pointer(buffer.id))
                    
                if issubclass(field[1], Structure) and hasattr(field[1], "apply_pointers"):
                    result.extend(getattr(self, field[0]).apply_pointers(cmd_queue))

            return result

        type_body = {
            '__module__': self._module_path,
            "_fields_": _fields_,
            "dtype": dtype,
            "_members": struct_info.members,
            "_pointer_types": pointer_types,

            "apply_pointers": apply_pointers,
            "__init__": __init__,
            "__repr__": lambda self: f"{struct_name}({', '.join([member.name + '=' + str(getattr(self, member.name)) for member in self._members.values()])})"
        }

        for field in _fields_:
            if issubclass(field[1], _Pointer):
                ori_name:str = field[0].replace(f"_{self.__class__.__name__}_ptr_", "")
                type_body[ori_name] = property(self._make_getter(ori_name), self._make_setter(ori_name))

        # A union gives every member offset zero, and a type() over a Structure base
        # would lay them out one after another -- the device type would then be twice
        # as wide as the value the caller hands over. The dtype already says which it
        # is (see structs.is_union_dtype), so the base follows it.
        from ...kernel.structs import is_union_dtype  # noqa: PLC0415 (import cycle)

        base_type = ctypes_union if is_union_dtype(dtype) else Structure
        struct_type = type(struct_name, (base_type,), type_body)
        self._struct_types[struct_name] = struct_type

        return struct_type

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
