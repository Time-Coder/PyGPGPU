from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union, Any
from ctypes import c_uint, sizeof
import enum

if TYPE_CHECKING:
    from ctypes import Array, _Pointer, c_void_p

# A host-side pointer as the ctypes layers use it: an explicit c_void_p, a raw address, a
# bytes-like buffer, or a ctypes Array/_Pointer. Declared once so the three backend stacks
# agree (Buffers construct such values in several places).
#   The names after "from __future__ import annotations" are strings, so nothing here is
#   evaluated at runtime -- ctypes.Array/_Pointer are not subscriptable.
HostPointer = Union["c_void_p", int, bytes, bytearray, "Array[Any]", "_Pointer[Any]", None]


class Constant:

    __instances:Dict[Union[str, int], Constant] = {}

    def __init__(self, name:str, value:Any):
        self.__name:str = name
        Constant.__instances[name] = self

    @property
    def name(self)->str:
        return self.__name
    
    @staticmethod
    def get(key:int)->Constant:
        return Constant.__instances[key]


class IntConstant(int, Constant):

    def __new__(cls, name:str, value:int):
        instance = int.__new__(cls, value)
        return instance
    
    @property
    def value(self)->int:
        return int(self)
    
    def __str__(self)->str:
        return f"{self.name}({int.__str__(self)})"
    
    def __repr__(self)->str:
        return str(self)


class FloatConstant(float, Constant):

    def __new__(cls, name:str, value:float):
        instance = float.__new__(cls, value)
        return instance
    
    @property
    def value(self)->float:
        return float(self)
    
    def __str__(self)->str:
        return f"{self.name}({float.__str__(self)})"
    
    def __repr__(self)->str:
        return str(self)
    

class IntEnum(enum.IntEnum):

    def __str__(self)->str:
        return f"{self.name}({self.value})"
    
    def __repr__(self)->str:
        return str(self)
    
    @classmethod
    def dtype(cls):
        return c_uint
    
    @classmethod
    def size(cls):
        return sizeof(cls.dtype())
    

class IntFlag(enum.IntFlag):

    def __str__(self)->str:
        if len(list(self)) < 2:
            return f"{self.name}({self.value})"
        else:
            return enum.Flag.__str__(self)
    
    def __repr__(self)->str:
        return str(self)
    
    @classmethod
    def dtype(cls):
        return c_uint
    
    @classmethod
    def size(cls):
        return sizeof(cls.dtype())
    

class KernelFlags(IntFlag):
    Readed = (1 << 0)
    Writed = (1 << 1)