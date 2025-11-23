from __future__ import annotations

from ctypes import c_void_p, sizeof, _SimpleCData, Structure, c_size_t, pointer, c_char
from typing import Dict, Any, get_args
from abc import ABC, abstractmethod
import weakref

from ..runtime import CL, IntEnum, IntFlag, cl_bool, cl_uint, CLInfo


class CLObject(ABC):

    def __init__(self, id:c_void_p):
        if isinstance(id, int):
            id = c_void_p(id)
            
        self._id = id
        self._info:Dict[str, Any] = {}
        self._finalizer = weakref.finalize(self, self._release, self._id)

    @property
    def id(self)->c_void_p:
        return self._id
    
    def __hash__(self):
        if isinstance(self._id, c_void_p):
            return self._id.value
        else:
            return self._id

    def __getattr__(self, name:str)->Any:
        if name in self._info:
            return self._info[name]

        key = None
        info_enum = self._info_enum()
        if hasattr(info_enum, f"{self._prefix()}_{name.upper()}"):
            key = getattr(info_enum, f"{self._prefix()}_{name.upper()}")
        elif hasattr(info_enum, f"{self._prefix()}_{name.upper()}_KHR"):
            key = getattr(info_enum, f"{self._prefix()}_{name.upper()}_KHR")
        elif hasattr(info_enum, f"CL_{name.upper()}"):
            key = getattr(info_enum, f"CL_{name.upper()}")
        elif hasattr(info_enum, f"CL_{name.upper()}_KHR"):
            key = getattr(info_enum, f"CL_{name.upper()}_KHR")

        if key is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        value = self._fetch_info(key)
        if key not in CLInfo.no_cached_info:
            self._info[name] = value

        return value
    
    def __eq__(self, other:CLObject)->bool:
        return (self.__class__ == other.__class__ and self.id == other.id)
    
    def _fetch_info(self, key)->Any:
        get_info_func = self._get_info_func()

        result_size = c_size_t()
        get_info_func(self.id, key, 0, None, pointer(result_size))

        result_bytes = (c_char * result_size.value)()
        get_info_func(self.id, key, result_size, result_bytes, None)

        info_types_map = self._info_types_map()
        cls = info_types_map[key]
        return self._parse_result(result_bytes, cls)

    def __repr__(self)->str:
        try:
            result = f"{self.__class__.__name__}('{self.name}')"
        except:
            _id = self.id
            if not isinstance(_id, int):
                _id = _id.value

            result = f"{self.__class__.__name__}(0x{_id:x})"

        return result

    @staticmethod
    @abstractmethod
    def _prefix()->str:
        pass

    @staticmethod
    @abstractmethod
    def _get_info_func()->CL.Func:
        pass

    @staticmethod
    @abstractmethod
    def _info_types_map()->Dict[IntEnum, type]:
        pass

    @staticmethod
    @abstractmethod
    def _info_enum()->type:
        pass

    @staticmethod
    @abstractmethod
    def _release_func()->CL.Func:
        pass

    @classmethod
    def _release(cls, obj_id:c_void_p):
        release_func = cls._release_func()
        if obj_id and release_func is not None:
            release_func(obj_id)

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
        elif issubclass(cls, Structure):
            return cls.from_buffer_copy(buffer)
        elif issubclass(cls, cl_bool):
            return bool(cl_uint.from_buffer_copy(buffer).value)
        elif issubclass(cls, IntEnum):
            return cls(cls.dtype().from_buffer_copy(buffer).value)
        elif issubclass(cls, IntFlag):
            return cls(cls.dtype().from_buffer_copy(buffer).value)
        elif issubclass(cls, str):
            return buffer.value.decode("utf-8")
        elif issubclass(cls, bytes):
            return buffer.raw
        