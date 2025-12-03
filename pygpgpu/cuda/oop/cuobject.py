from __future__ import annotations

from ctypes import c_int, c_void_p, pointer
from typing import Dict, Any, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import weakref

from ..driver import CUDA, CUInfo

if TYPE_CHECKING:
    from .context import Context
    from .device import Device


class CUObject(ABC):

    _instances = weakref.WeakValueDictionary()

    def __init__(self, id:Union[int, c_void_p]):
        self._id:Union[int, c_void_p] = id
        self._info:Dict[str, Any] = {}
        self._finalizer = weakref.finalize(self, self._release, self._id)
        CUObject._instances[self.__hash__()] = self

        self._context = None
        if self.__class__.__name__ not in ["Context", "Device", "NVRTCProgram"]:
            from .context import Context
            self._context = Context.current()

    @property
    def id(self)->Union[int, c_void_p]:
        return self._id
    
    @property
    def context(self)->Context:
        if self.__class__.__name__ == "Context":
            return self
        else:
            return self._context
    
    @property
    def device(self)->Device:
        if self.__class__.__name__ == "Device":
            return self
        else:
            try:
                return self._context.device
            except:
                return None
    
    @staticmethod
    def instance(id:Union[int, c_void_p])->CUObject:
        if isinstance(id, c_void_p):
            id = id.value

        if id not in CUObject._instances:
            return None
        
        return CUObject._instances[id]

    def __hash__(self)->int:
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

        if key is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        value = self._fetch_info(key)
        if key not in CUInfo.no_cached_info:
            self._info[name] = value

        return value
    
    def __eq__(self, other:CUObject)->bool:
        return (self.__class__ == other.__class__ and self.id == other.id)
    
    def _fetch_info(self, key)->int:
        get_info_func = self._get_info_func()

        result = c_int(0)
        ptr_result = pointer(result)
        get_info_func(ptr_result, key, self.id)

        return result.value

    def __repr__(self)->str:
        try:
            result = f"{self.__class__.__name__}('{self.name}')"
        except:
            _id = self.id
            if isinstance(_id, int):
                result = f"{self.__class__.__name__}({_id})"
            else:
                _id = _id.value
                result = f"{self.__class__.__name__}(0x{_id:x})"

        return result

    @staticmethod
    @abstractmethod
    def _prefix()->str:
        pass

    @staticmethod
    @abstractmethod
    def _get_info_func()->CUDA.Func:
        pass

    @staticmethod
    @abstractmethod
    def _info_enum()->type:
        pass

    @staticmethod
    @abstractmethod
    def _release_func()->CUDA.Func:
        pass

    @classmethod
    def _release(cls, obj_id:Union[c_void_p, int]):
        release_func = cls._release_func()
        if obj_id and release_func is not None:
            release_func(obj_id)
        