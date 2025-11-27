from __future__ import annotations
import functools
from typing import Dict, Tuple, TYPE_CHECKING, List, Optional, Any

import numpy as np

if TYPE_CHECKING:
    from ..opencl.oop import Buffer, CommandQueue
    from ..opencl.runtime import cl_mem_flags


class ndarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._parent = None
        obj._buffers = {}
        obj._arg_name = ""
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        self._buffers:Dict[Tuple[str, int, cl_mem_flags], List[Buffer, CommandQueue]] = {}
        self._arg_name:str = ""
        if np.shares_memory(obj, self):
            self._parent = obj
        else:
            self._parent = None

    def _set_host_dirty(self):
        for buffer, cmd_queue in self._buffers.values():
            buffer.dirty_on_host = True

        if self._parent is not None:
            self._parent._set_host_dirty()

    def _not_const(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self:ndarray = args[0]
            self.to_host()
            self._set_host_dirty()
            super_func = getattr(np.ndarray, func.__name__)
            return super_func(*args, **kwargs)
        
        return wrapper
    
    def _const(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self:ndarray = args[0]
            self.to_host()
            super_func = getattr(np.ndarray, func.__name__)
            return super_func(*args, **kwargs)
        
        return wrapper

    @_not_const
    def __setitem__(self, key, value):
        pass

    @_not_const
    def __iadd__(self, other):
        pass

    @_not_const
    def __isub__(self, other):
        pass

    @_not_const
    def __imul__(self, other):
        pass

    @_not_const
    def __itruediv__(self, other):
        pass

    @_not_const
    def __ifloordiv__(self, other):
        pass

    @_not_const
    def __imod__(self, other):
        pass

    @_not_const
    def __ipow__(self, other):
        pass

    @_not_const
    def __ilshift__(self, other):
        pass

    @_not_const
    def __irshift__(self, other):
        pass

    @_not_const
    def __iand__(self, other):
        pass

    @_not_const
    def __ixor__(self, other):
        pass

    @_not_const
    def __ior__(self, other):
        pass

    def _change_to_ndarray(self, value:np.ndarray):
        if not isinstance(value, ndarray):
            value = value.view(ndarray)

        value._buffers = {}
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
        if 'out' in kwargs:
            out = kwargs["out"]
            if out is self or (isinstance(out, tuple) and self in out):
                self._set_host_dirty()

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        result = self._change_result(result)

        return result

    def __array_function__(self, func, types, args, kwargs):
        result = func._implementation(*args, **kwargs)
        result = self._change_result(result)

        return result

    def to_host(self):
        event = None
        lastest_dirty_buffer:Optional[Buffer] = None
        used_cmd_queue:Optional[CommandQueue] = None
        for buffer, cmd_queue in self._buffers.values():
            if buffer.dirty_on_device and (lastest_dirty_buffer is None or buffer.device_dirty_time > lastest_dirty_buffer.device_dirty_time):
                lastest_dirty_buffer = buffer
                used_cmd_queue = cmd_queue
                break

        if lastest_dirty_buffer is not None:
            event = lastest_dirty_buffer.read(used_cmd_queue)
            event.wait()
            lastest_dirty_buffer.dirty_on_device = False

            if lastest_dirty_buffer.data is not self:
                self[:] = lastest_dirty_buffer.data

            lastest_dirty_buffer.dirty_on_host = False

            print(f"copy data to host for argument '{self._arg_name}'")

        for buffer, cmd_queue in self._buffers.values():
            buffer.dirty_on_device = False

    def _copy_to_device(self, backend:str, cmd_queue:CommandQueue, flags:cl_mem_flags, arg_name:str)->Buffer:
        self._arg_name = arg_name
        key = (backend, cmd_queue.context.id.value, flags)

        if key not in self._buffers:
            buffer = cmd_queue.context.create_buffer(size=self.nbytes, flags=flags)
            buffer.dirty_on_host = True
            self._buffers[key] = [buffer, cmd_queue]
        else:
            self._buffers[key][1] = cmd_queue

        buffer = self._buffers[key][0]
        if buffer.dirty_on_host:
            event = buffer.set_data(cmd_queue, self)
            CL_MEM_READ_ONLY = (1 << 2)
            if not (buffer.flags & CL_MEM_READ_ONLY):
                buffer.dirty_on_device = True

            buffer.dirty_on_host = False

            event.wait()
            print(f"copy data to device for argument '{arg_name}'")

        return buffer

    @_const
    def __getitem__(self, key):
        pass

    @_const
    def copy(self, order):
        pass

    @_const
    def dump(self, file) -> None:
        pass
    
    @_const
    def dumps(self) -> bytes:
        pass
    
    @_const
    def tobytes(self, order) -> bytes:
        pass
    
    @_const
    def tofile(self, fid, sep: str = ..., format: str = ...) -> None:
        pass
    
    @_const
    def tolist(self) -> Any:
        pass

    @_const
    @property
    def ctypes(self):
        pass