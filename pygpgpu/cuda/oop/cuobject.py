from __future__ import annotations

from ctypes import c_int, c_size_t, c_void_p, cast, pointer, POINTER
from typing import Dict, Any, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import weakref

from ..driver import CUInfo

if TYPE_CHECKING:
    from .context import Context
    from .device import Device


class CUObject(ABC):

    _instances = weakref.WeakValueDictionary()

    # Registry that lets a native callback find the object it was registered for
    # without the object referencing itself.
    #
    # Callbacks receive a void* the driver carries back verbatim. Storing a
    # ctypes.py_object(self) there would work, but py_object holds a strong
    # reference, so the object formed a reference cycle with itself and could
    # only be reclaimed by the cyclic collector -- a stream stayed alive long
    # after its last use, which is what made streams pile up. Storing id(self)
    # instead and resolving it through this weak registry keeps the address
    # stable for the driver while leaving the reference weak, so the object dies
    # as soon as nothing else holds it.
    _callback_targets:weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def _callback_data(self) -> c_void_p:
        """The void* to hand a driver callback; resolves back via _from_callback."""
        ident = id(self)
        CUObject._callback_targets[ident] = self
        # The driver only carries the pointer, so its value is what matters; a
        # one-element buffer gives it a real address to point at.
        self._callback_ident = c_size_t(ident)
        return cast(pointer(self._callback_ident), c_void_p)

    @classmethod
    def _from_callback(cls, user_data:c_void_p):
        """Resolve the object a callback was registered for, or None if it is gone."""
        try:
            ident = cast(user_data, POINTER(c_size_t)).contents.value
        except Exception:
            return None
        return cls._callback_targets.get(ident)

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
        elif hasattr(self._id, "value"):
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
    def _get_info_func()->Any:
        pass

    @staticmethod
    @abstractmethod
    def _info_enum()->type|None:
        pass

    @staticmethod
    @abstractmethod
    def _release_func()->Any:
        pass

    @classmethod
    def _release(cls, obj_id:Union[c_void_p, int]):
        release_func = cls._release_func()
        if obj_id and release_func is not None:
            # Destruction runs from weakref.finalize as well, where raising is
            # pointless: the interpreter only prints the error and carries on. A
            # handle can legitimately be dead by then -- a stream cached for a
            # thread whose CUDA context has since been destroyed, for instance --
            # so a failed destroy is swallowed rather than reported.
            try:
                release_func(obj_id)
            except BaseException:
                pass
        