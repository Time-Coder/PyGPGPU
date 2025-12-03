from __future__ import annotations
from typing import Tuple, Any, Dict, List, Optional

import numpy as np

from ..driver import CUInfo
from .mem_object import MemObject


class VarInfo:

    def __init__(self, name: str, type_str: str, array_shape: Tuple[int, ...], type_qualifiers: List[str]):
        self.name = name
        self.type_str = type_str
        self.array_shape = array_shape
        self.type_qualifiers = type_qualifiers
        self.value: Any = None
        self.mem_obj: Optional[MemObject] = None

    @property
    def is_ptr(self)->bool:
        return (self.type_str[-1] == "*")
    
    @property
    def base_type_str(self)->str:
        return (self.type_str[:-1] if self.is_ptr else self.type_str)
    
    @property
    def type_annotation(self)->str:
        base_type_str = self.base_type_str
        if self.is_ptr:
            dtype_name = base_type_str
            if base_type_str in CUInfo.scalar_types:
                dtype_name = "np." + np.dtype(CUInfo.scalar_types[base_type_str]).name
            return f"NDArray[{dtype_name}]"
        else:
            if base_type_str in ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']:
                return 'int'
            elif base_type_str in ['float', 'double']:
                return 'float'
            else:
                return base_type_str
    
    @property
    def readonly(self)->bool:
        return ("const" in self.type_qualifiers)
    
    @property
    def writeonly(self)->bool:
        return False
    
    @property
    def need_read_back(self)->bool:
        return (self.is_ptr and "const" not in self.type_qualifiers)
    

class ArgInfo(VarInfo):

    def __init__(self, parent:KernelInfo, name: str, type_str: str, array_shape:Tuple[int, ...], type_qualifiers: List[str]):
        VarInfo.__init__(self, name, type_str, array_shape, type_qualifiers)
        self.parent = parent
    
    def check_type(self, value:Any)->None:
        base_type_str = self.base_type_str
        if self.is_ptr:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be np.ndarray, got {value.__class__.__name__}.")
            
            dtype = None
            dtype_name = base_type_str
            if base_type_str in CUInfo.scalar_types:
                dtype = np.dtype(CUInfo.scalar_types[base_type_str])
                dtype_name = "np." + dtype.name
            elif base_type_str in CUInfo.vec_types:
                dtype = CUInfo.vec_types[base_type_str]

            if dtype is not None:
                if value.dtype != dtype:
                    raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be NDArray[{dtype_name}], got NDArray[np.{value.dtype.name}].")
        else:
            base_type = None
            if base_type_str in CUInfo.basic_types:
                base_type = CUInfo.basic_types[base_type_str]

            if base_type_str in CUInfo.alter_types:
                base_type = CUInfo.alter_types[base_type_str]

            same_type = (value.__class__.__name__ == base_type_str if base_type is None else isinstance(value, base_type))

            if not same_type:
                raise TypeError(f"{self.parent.signature()}'s argument '{self.name}' must be {base_type_str}, got {value.__class__.__name__}.")


class StructInfo:

    def __init__(self, name:str):
        self.name: str = name
        self.members: Dict[str, VarInfo] = {}

    def declare(self)->str:
        result = f"""
class {self.name}(ctypes.Structure):

    def __init__(self, {", ".join([member.name + ": " + member.type_annotation for member in self.members.values()])})->None: ...

"""
        for member in self.members.values():
            result += f"    {member.name}: {member.type_annotation}\n"

        result += "\n"

        return result


class KernelInfo:

    def __init__(self, name:str):
        self.name: str = name
        self.args: Dict[str, ArgInfo] = {}

    def args_declare(self, with_annotation:bool=False)->str:
        if with_annotation:
            return ", ".join([arg.name + ": " + arg.type_annotation for arg in self.args.values()])
        else:
            return ", ".join(list(self.args.keys()))

    def signature(self, with_annotation:bool=False)->str:
        return self.name + "(" + self.args_declare(with_annotation) + ")"
