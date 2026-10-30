from __future__ import annotations
from typing import Tuple, Any, Dict, List, Optional

import numpy as np
from ... import numpy as gnp
from ...constants import KernelFlags

from ..driver import CUInfo, CUDA
from .mem_object import MemObject
from .stream import Stream
from .buffer import Buffer
from .event import Event


class VarInfo:

    # The CUDA path writes unsigned types in their full C spelling (NVRTC does not define
    # uchar/ushort/uint/ulong), while the driver tables are keyed by the OpenCL-style short
    # names. Mapping here keeps every downstream lookup working on one spelling.
    _TYPE_ALIASES:Dict[str, str] = {
        "unsigned char": "uchar",
        "unsigned short": "ushort",
        "unsigned int": "uint",
        "unsigned long long": "ulong",
        "unsigned long": "ulong",
        # `long long` is how the 64-bit signed type is spelled for NVRTC (see
        # _CUDA_TYPE_NAMES); the driver tables key it as `long`.
        "long long": "long",
    }

    def __init__(self, name: str, type_str: str, array_shape: Tuple[int, ...], type_qualifiers: List[str], kernel_flags:Optional[KernelFlags]=None):
        self.name = name
        # Normalise the spelling while keeping any trailing pointer marker.
        stripped:str = type_str.strip()
        pointer:str = ""
        if stripped.endswith("*"):
            stripped = stripped[:-1].rstrip()
            pointer = "*"
        self.type_str = self._TYPE_ALIASES.get(stripped, stripped) + pointer
        self.array_shape = array_shape
        self.type_qualifiers = type_qualifiers
        self.kernel_flags = (KernelFlags.Readed | KernelFlags.Writed) if kernel_flags is None else kernel_flags
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
    def writeonly(self)->bool:
        return False
    
    @property
    def readonly(self)->bool:
        """A texture is read-only by definition; a surface follows the analysis.

        CUDA separates the two object kinds precisely so that a sampler and a
        write target are distinguishable: a texture object can only be sampled, while
        a surface object exists to be written. Declaring a texture read-only here
        means the static analysis never has to infer it, and a surface keeps whatever
        read/write verdict the analysis reaches for its uses in the kernel body.
        """
        if self.base_type_str == "cudaTextureObject_t":
            return True
        return ("const" in self.type_qualifiers)

    @property
    def need_read_back(self)->bool:
        # A texture is an opaque read-only object, not memory the kernel writes into,
        # so it never takes part in the upload/read-back flow. A surface can be
        # written, so it keeps the usual rule.
        if self.base_type_str == "cudaTextureObject_t":
            return False
        if self.base_type_str == "cudaSurfaceObject_t":
            return bool(self.kernel_flags & KernelFlags.Writed)
        return (self.is_ptr and bool(self.kernel_flags & KernelFlags.Writed))
    
    def use_buffer(self, stream:Stream, data:np.ndarray)->Tuple[Buffer, Optional[Event]]:
        if isinstance(data, gnp.ndarray):
            used_buffer, event = data._to_device(stream, self.name, kernel_flags=self.kernel_flags)
        else:
            used_buffer:Buffer = stream.context.get_buffer(data.nbytes)
            used_buffer.kernel_flags = self.kernel_flags
            event = used_buffer.set_data(stream, data)

        if event is not None and CUDA.print_info:
            print(f"copy data to device for argument '{self.name}'")
        
        return used_buffer, event
    

class ArgInfo(VarInfo):

    def __init__(self, parent:KernelInfo, name: str, type_str: str, array_shape:Tuple[int, ...], type_qualifiers: List[str], kernel_flags:Optional[KernelFlags]=None):
        VarInfo.__init__(self, name, type_str, array_shape, type_qualifiers, kernel_flags)
        self.parent = parent
    
    def check_type(self, value:Any)->None:
        base_type_str = self.base_type_str
        # A texture or surface handle is neither a numpy array nor a scalar: it is an
        # opaque object the kernel reads or writes, so it is accepted as-is. The
        # backend validates the handle itself when the argument is bound.
        from .texture import TextureObject
        from .surface import SurfaceObject
        from ..driver import cudaTextureObject_t, cudaSurfaceObject_t
        if isinstance(value, (TextureObject, cudaTextureObject_t,
                              SurfaceObject, cudaSurfaceObject_t)):
            return
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
