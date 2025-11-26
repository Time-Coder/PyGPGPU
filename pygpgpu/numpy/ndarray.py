import functools
from typing import Dict, Tuple, TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from ..opencl.oop import Buffer, CommandQueue
    from ..opencl.runtime import cl_mem_flags


class ndarray(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._dirty = True
        obj._buffers = {}
        obj._arg_name = ""
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        self._dirty = getattr(obj, '_dirty', False)
        self._buffers:Dict[Tuple[str, int, cl_mem_flags], List[Buffer, CommandQueue]] = {}
        self._arg_name:str = ""

    @property
    def dirty(self):
        return self._dirty

    def _mark_dirty(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            self._dirty = True
            func_name = func.__name__
            super_func = getattr(np.ndarray, func_name)
            return super_func(*args, **kwargs)
        
        return wrapper

    def __getitem__(self, key):
        self._copy_to_host()
        result = super().__getitem__(key)
        if isinstance(result, ndarray):
            result = np.array(result, copy=True)
        else:
            result = np.array(result, copy=True)
        result.flags.writeable = False
        return result

    @_mark_dirty
    def __setitem__(self, key, value):
        pass

    @_mark_dirty
    def __iadd__(self, other):
        pass

    @_mark_dirty
    def __isub__(self, other):
        pass

    @_mark_dirty
    def __imul__(self, other):
        pass

    @_mark_dirty
    def __itruediv__(self, other):
        pass

    @_mark_dirty
    def __ifloordiv__(self, other):
        pass

    @_mark_dirty
    def __imod__(self, other):
        pass

    @_mark_dirty
    def __ipow__(self, other):
        pass

    @_mark_dirty
    def __ilshift__(self, other):
        pass

    @_mark_dirty
    def __irshift__(self, other):
        pass

    @_mark_dirty
    def __iand__(self, other):
        pass

    @_mark_dirty
    def __ixor__(self, other):
        pass

    @_mark_dirty
    def __ior__(self, other):
        pass

    def __array_wrap__(self, out_arr, context=None):
        if out_arr is self:
            return out_arr
        
        if isinstance(out_arr, ndarray):
            out_arr._dirty = False

        return out_arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', None)
        if out is not None:
            if isinstance(out, tuple) and self in out:
                self._dirty = True
            elif out is self:
                self._dirty = True

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if result is not None and result is not self:
            if isinstance(result, ndarray):
                result = np.asarray(result)
            result.flags.writeable = False

        return result

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, ndarray) or issubclass(t, np.ndarray) for t in types):
            return NotImplemented

        result = func._implementation(*args, **kwargs)

        if isinstance(result, np.ndarray):
            if np.shares_memory(result, self):
                result = result.view(np.ndarray)
                result.flags.writeable = False
            else:
                result = np.array(result, copy=False)
                result.flags.writeable = True

        return result

    def _copy_to_host(self):
        event = None
        dirty_buffer = None
        for buffer, cmd_queue in self._buffers.values():
            if buffer.dirty:
                dirty_buffer = buffer
                event = buffer.read(cmd_queue)
                break

        if event is not None:
            event.wait()
            dirty_buffer.dirty = False
            print(f"copy data to host for argument '{self._arg_name}'")

    def _copy_to_device(self, backend:str, cmd_queue:CommandQueue, flags:cl_mem_flags, arg_name:str):
        self._arg_name = arg_name
        key = (backend, cmd_queue.context.id.value, flags)
        self_dirty = self._dirty

        if key not in self._buffers:
            buffer = cmd_queue.context.create_buffer(size=self.nbytes, flags=flags)
            self._buffers[key] = [buffer, cmd_queue]
            self_dirty = True

        self._buffers[key][1] = cmd_queue

        if self_dirty:
            buffer = self._buffers[key][0]
            event = buffer.set_data(cmd_queue, self)
            if not (buffer.flags & cl_mem_flags.CL_MEM_READ_ONLY):
                buffer.dirty = True

            event.wait()
            print(f"copy data to device for argument '{arg_name}'")

    def __deepcopy__(self, memo):
        return ndarray(np.array(self, copy=True))

    def copy(self, order='C'):
        new_arr = np.array(self, copy=True, order=order).view(ndarray)
        new_arr._dirty = self._dirty
        return new_arr

    def view(self, dtype=None, type=None):
        v = super().view(dtype=dtype, type=np.ndarray)
        v.flags.writeable = False
        return v

    def __setattr__(self, name, value):
        if name == 'flags' and hasattr(self, 'flags'):
            if hasattr(value, 'writeable') and value.writeable and self.flags.owndata is False:
                raise ValueError("Cannot make a view of ndarray writeable.")
        super().__setattr__(name, value)
