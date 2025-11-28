from __future__ import annotations
import functools
from typing import Dict, Tuple, TYPE_CHECKING, List, Optional, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..opencl.oop import CommandQueue, Event, MemObject
    from ..opencl.driver import cl_mem_flags, imagend_t


def _not_const(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self:ndarray = args[0]
        self.to_host()
        self._set_host_dirty()
        result = func(*args, **kwargs)
        return self._change_result(result)
    
    return wrapper


def _const(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self:ndarray = args[0]
        self.to_host()
        result = func(*args, **kwargs)
        return self._change_result(result)
    
    return wrapper


class Descriptor:

    def __init__(self, old_descriptor):
        self.old_descriptor = old_descriptor
        self.getter = _const(self.old_descriptor.__get__)
        self.setter = _not_const(self.old_descriptor.__set__)

    def __get__(self, obj, objtype=None):
        return self.getter(obj, objtype)

    def __set__(self, obj, value):
        return self.setter(obj, value)

    def __delete__(self, obj):
        return self.old_descriptor.__delete__(obj)


class ndarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._parent = None
        obj._mem_objs = {}
        obj._arg_name = ""
        obj._should_copy = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        self._mem_objs:Dict[Tuple[str, int, cl_mem_flags], List[MemObject, CommandQueue]] = {}
        self._arg_name:str = ""
        self._should_copy:bool = True
        if np.shares_memory(obj, self):
            self._parent:Optional[ndarray] = obj
        else:
            self._parent:Optional[ndarray] = None

    def _set_host_dirty(self):
        for mem_obj, cmd_queue in self._mem_objs.values():
            mem_obj.dirty_on_host = True

        if self._parent is not None:
            self._parent._set_host_dirty()

    __setitem__ = _not_const(np.ndarray.__setitem__)
    __iadd__ = _not_const(np.ndarray.__iadd__)
    __isub__ = _not_const(np.ndarray.__isub__)
    __imul__ = _not_const(np.ndarray.__imul__)
    __itruediv__ = _not_const(np.ndarray.__itruediv__)
    __ifloordiv__ = _not_const(np.ndarray.__ifloordiv__)
    __imod__ = _not_const(np.ndarray.__imod__)
    __ipow__ = _not_const(np.ndarray.__ipow__)
    __ilshift__ = _not_const(np.ndarray.__ilshift__)
    __irshift__ = _not_const(np.ndarray.__irshift__)
    __iand__ = _not_const(np.ndarray.__iand__)
    __ixor__ = _not_const(np.ndarray.__ixor__)
    __ior__ = _not_const(np.ndarray.__ior__)

    def _change_to_ndarray(self, value:np.ndarray):
        if value is self:
            return value

        if not isinstance(value, ndarray):
            value = value.view(ndarray)

        value._mem_objs = {}
        value._arg_name = ""
        if np.shares_memory(value, self):
            value._parent = self
        else:
            value._parent = None

        return value
    
    def _change_result(self, result):
        if isinstance(result, np.ndarray):
            result = self._change_to_ndarray(result)

        if isinstance(result, tuple):
            result = list(result)
            for i, sub_result in enumerate(result):
                if isinstance(sub_result, np.ndarray):
                    result[i] = self._change_to_ndarray(sub_result)

            result = tuple(result)

        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.to_host()
        for value in inputs:
            if isinstance(value, ndarray):
                value.to_host()

        for value in kwargs.values():
            if isinstance(value, ndarray):
                value.to_host()

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if 'out' in kwargs:
            out = kwargs["out"]
            if out is self or (isinstance(out, tuple) and self in out):
                self._set_host_dirty()

        result = self._change_result(result)

        return result

    def __array_function__(self, func, types, args, kwargs):
        self.to_host()
        for value in args:
            if isinstance(value, ndarray):
                value.to_host()

        for value in kwargs.values():
            if isinstance(value, ndarray):
                value.to_host()

        result = func._implementation(*args, **kwargs)
        result = self._change_result(result)

        return result

    def to_host(self):
        if not self._should_copy or not hasattr(self, "_parent") or not hasattr(self, "_mem_objs"):
            return
        
        self._should_copy = False

        if self._parent is not None:
            self._parent.to_host()

        event = None
        lastest_dirty_mem_obj:Optional[MemObject] = None
        used_cmd_queue:Optional[CommandQueue] = None
        for mem_obj, cmd_queue in self._mem_objs.values():
            if mem_obj.dirty_on_device and (lastest_dirty_mem_obj is None or mem_obj.device_dirty_time > lastest_dirty_mem_obj.device_dirty_time):
                lastest_dirty_mem_obj = mem_obj
                used_cmd_queue = cmd_queue
                break

        if lastest_dirty_mem_obj is not None:
            event = lastest_dirty_mem_obj.read(used_cmd_queue)
            event.wait()
            lastest_dirty_mem_obj.dirty_on_device = False

            if lastest_dirty_mem_obj.data is not self:
                self[:] = lastest_dirty_mem_obj.data

            lastest_dirty_mem_obj.dirty_on_host = False

            print(f"copy data to host for argument '{self._arg_name}'")

        for mem_obj, cmd_queue in self._mem_objs.values():
            mem_obj.dirty_on_device = False

        self._should_copy = True

    def _to_device(self, cmd_queue:CommandQueue, arg_name:str, flags:cl_mem_flags=(1 << 0), kernel_flags:cl_mem_flags=(1 << 0), image_info:imagend_t=None)->Tuple[MemObject, Event]:
        old_shoule_copy = self._should_copy
        self._should_copy = False

        self._arg_name = arg_name

        if image_info is None:
            key = (cmd_queue.context.id.value, flags)
        else:
            key = (cmd_queue.context.id.value, image_info.__class__.__name__, image_info.shape, image_info.dtype, image_info.flags)

        if key not in self._mem_objs:
            if image_info is None:
                mem_obj = cmd_queue.context.create_buffer(size=self.nbytes, flags=flags)
            else:
                mem_obj = cmd_queue.context.create_image(image_info)
                
            mem_obj.kernel_flags = kernel_flags
            mem_obj.dirty_on_host = True
            self._mem_objs[key] = [mem_obj, cmd_queue]
        else:
            self._mem_objs[key][1] = cmd_queue

        event = None
        mem_obj = self._mem_objs[key][0]
        if mem_obj.dirty_on_host:
            event = mem_obj.set_data(cmd_queue, self)
            CL_MEM_READ_ONLY = (1 << 2)
            if not (mem_obj.flags & CL_MEM_READ_ONLY):
                mem_obj.dirty_on_device = True

            mem_obj.dirty_on_host = False

        self._should_copy = old_shoule_copy
        return mem_obj, event

    __getitem__ = _const(np.ndarray.__getitem__)
    copy = _const(np.ndarray.copy)
    dump = _const(np.ndarray.dump)
    dumps = _const(np.ndarray.dumps)
    tobytes = _const(np.ndarray.tobytes)
    tofile = _const(np.ndarray.tofile)
    tolist = _const(np.ndarray.tolist)

    data = Descriptor(np.ndarray.data)
    ctypes = Descriptor(np.ndarray.ctypes)
    real = Descriptor(np.ndarray.real)
    imag = Descriptor(np.ndarray.imag)
    flat = Descriptor(np.ndarray.flat)
    T = Descriptor(np.ndarray.T)

    @property
    def base(self) -> Optional[ndarray]:
        return self._parent


