from ctypes import c_void_p, sizeof, _SimpleCData, LittleEndianStructure, c_size_t, byref, c_char
from typing import Dict, Any, get_args
from abc import ABC, abstractmethod

from ..runtime import CL, IntEnum, IntFlag, cl_bool, cl_uint


class CLObject(ABC):

    def __init__(self, id:c_void_p):
        self._id = id
        self._info:Dict[str, Any] = {}

    @property
    def id(self)->c_void_p:
        return self._id
    
    def __getattr__(self, name:str)->Any:
        if name in self._info:
            return self._info[name]

        key = None
        if hasattr(self._info_enum, f"{self._prefix}_{name.upper()}"):
            key = getattr(self._info_enum, f"{self._prefix}_{name.upper()}")
        elif hasattr(self._info_enum, f"{self._prefix}_{name.upper()}_KHR"):
            key = getattr(self._info_enum, f"{self._prefix}_{name.upper()}_KHR")
        elif hasattr(self._info_enum, f"CL_{name.upper()}"):
            key = getattr(self._info_enum, f"CL_{name.upper()}")
        elif hasattr(self._info_enum, f"CL_{name.upper()}_KHR"):
            key = getattr(self._info_enum, f"CL_{name.upper()}_KHR")

        if key is None:
            raise AttributeError(f"'Platform' object has no attribute '{name}'")

        self._info[name] = self._fetch_info(key)

        return self._info[name]
    
    def __eq__(self, other:CLObject)->bool:
        return (self.__class__ == other.__class__ and self._id == other._id)
    
    def _fetch_info(self, key)->Any:
        result_size = c_size_t()
        self._get_info_func(self._id, key, 0, None, byref(result_size))

        result_bytes = (c_char * result_size.value)()
        self._get_info_func(self._id, key, result_size, result_bytes, None)

        cls = self._info_types_map[key]
        return self._parse_result(result_bytes, cls)

    def __repr__(self)->str:
        return f"{self.__class__.__name__}('{self.name}')"

    @property
    @abstractmethod
    def _prefix(self)->str:
        pass

    @property
    @abstractmethod
    def _get_info_func(self)->CL.Func:
        pass

    @property
    @abstractmethod
    def _info_types_map(self)->Dict[IntEnum, type]:
        pass

    @property
    @abstractmethod
    def _info_enum(self)->type:
        pass

    @staticmethod
    def _parse_result(buffer:bytes, cls:type):
        if cls.__name__.startswith("List"):
            args = get_args(cls)
            ele_cls = args[0]
            if issubclass(ele_cls, (IntFlag, IntEnum)):
                step:int = ele_cls.size()
            else:
                step:int = sizeof(ele_cls)
            n:int = len(buffer) // step
            result = []
            offset:int = 0
            for i in range(n):
                value = CLObject._parse_result(buffer[offset:offset+step], ele_cls)
                result.append(value)
                offset += step
            return result

        if issubclass(cls, _SimpleCData):
            return cls.from_buffer_copy(buffer).value
        elif issubclass(cls, LittleEndianStructure):
            return cls.from_buffer_copy(buffer)
        elif issubclass(cls, IntEnum):
            return cls(cls.dtype().from_buffer_copy(buffer).value)
        elif issubclass(cls, IntFlag):
            return cls(cls.dtype().from_buffer_copy(buffer).value)
        elif issubclass(cls, str):
            return buffer.value.decode("utf-8")
        elif issubclass(cls, bytes):
            return buffer.raw
        elif issubclass(cls, cl_bool):
            return bool(cl_uint.from_buffer_copy(buffer).value)