from __future__ import annotations
from ctypes import c_size_t, c_uint64, pointer, sizeof, Structure, byref, memmove, c_void_p, cast
from ctypes import Union as ctypes_union
from functools import partial
import concurrent.futures
import asyncio
import threading
from typing import Dict, List, TYPE_CHECKING, Union, Tuple, Optional
from typing_extensions import override
import numpy as np
from ._kernel_support import (
    complex_of_dtype, _handle_types, _structure_bytes,
    _async_executor, _async_executor_lock, _async_workers, _deliver, _set_future_error,
)
from .... import numpy as gnp
from ...driver import (
    CUDA,
    CUInfo,
    CUfunction,
    CUresult,
    cudaTextureObject_t,
    cudaSurfaceObject_t
)
from ..cuobject import CUObject
from ..program_info import ArgInfo
from ..stream import Stream
from ..mem_object import MemObject
from ..program_parser import ProgramParser
from ....utils import detect_work_size, join_with_and, local_size_for_global, device_limits, product
from ._kernel_args import _KernelArgsMixin
from ._kernel_pack import _KernelPackMixin
from ._kernel_launch import _KernelLaunchMixin
from ._kernel_async import _KernelAsyncMixin


if TYPE_CHECKING:
    from ..program import Program
    from ..context import Context










class Kernel(_KernelArgsMixin, _KernelPackMixin, _KernelLaunchMixin, _KernelAsyncMixin, CUObject):

    def __init__(self, program:Program, name:str)->None:
        kernel_id = CUfunction()
        ptr_kernel_id = pointer(kernel_id)
        CUDA.cuModuleGetFunction(ptr_kernel_id, program.id, name.encode("utf-8"))
        CUObject.__init__(self, kernel_id)

        self._name:str = name
        self._program:Program = program
        self._args:Dict[str, ArgInfo] = {}
        self._arg_list:List[ArgInfo] = []
        self._arg_names:List[str] = []

        self.type_checked:bool = False
        # Per-call geometry set by __getitem__ and consumed by the next launch.
        # The same Kernel object is shared by every thread that uses the kernel
        # (PyKernel caches one compiled instance per signature), so this must be
        # thread-local: a plain attribute would let one thread's bracket leak into
        # another thread's launch.
        self._call_state = threading.local()


    def _get_geometry(self):
        state = self._call_state
        return (getattr(state, "grid_dims", None), getattr(state, "block_dims", None))


    def _take_geometry(self):
        """Read and clear the pending bracket geometry for the current thread."""
        state = self._call_state
        grid = getattr(state, "grid_dims", None)
        block = getattr(state, "block_dims", None)
        state.grid_dims = None
        state.block_dims = None
        return grid, block


    def __getitem__(self, sizes_or_device:Tuple[Union[int, Tuple[int,...], str], ...])->Kernel:
        if not isinstance(sizes_or_device, tuple):
            sizes_or_device = (sizes_or_device,)

        if len(sizes_or_device) == 0:
            raise TypeError("Kernel.__getitem__ takes at least 1 argument, 0 were given")
        elif len(sizes_or_device) == 2:
            grid_dims = sizes_or_device[0]
            block_dims = sizes_or_device[1]
            if isinstance(grid_dims, int):
                grid_dims = (grid_dims,)

            if isinstance(block_dims, int):
                block_dims = (block_dims,)

            work_dim = len(grid_dims)
            if work_dim > 3:
                raise ValueError("dimension over 3 is not supported")
            
            if work_dim <= 0:
                raise ValueError("dimension should be at least 1")
            
            # block_dims being None means auto-infer (only grid was specified)
            if block_dims is not None and work_dim != len(block_dims):
                raise ValueError(f"grid_dims={grid_dims} and block_dims={block_dims} do not have same dimension")
        
            self._call_state.grid_dims = grid_dims
            self._call_state.block_dims = block_dims
        else:
            raise TypeError(f"Kernel.__getitem__ takes at most 2 argument, {len(sizes_or_device)} were given")
        
        return self


    def __call__(self, *args, **kwargs)->None:
        stream = self._call(args, kwargs, async_launch=False)
        try:
            stream.sync()
        finally:
            # Nothing can be queued on the stream after the sync, so it is idle and
            # safe to hand back to the pool.
            self._release_stream(stream, async_launch=False)


    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        return self._launch_async(args, kwargs, concurrent.futures.Future)


    def async_call(self, *args, **kwargs)->asyncio.Future:
        return self._launch_async(args, kwargs, asyncio.Future)


    def _call(self, args, kwargs, async_launch:bool=False)->Stream:
        stream, grid_dims, block_dims, value_ptrs, bound = self._before_call(args, kwargs, async_launch)
        CUDA.cuLaunchKernel(
            self.id,
            grid_dims[0], grid_dims[1], grid_dims[2],
            block_dims[0], block_dims[1], block_dims[2],
            0, stream.id, value_ptrs, None
        )
        if CUDA.print_info:
            print(f"launch CUDA {self} on {stream.device} with grid_dim={tuple(grid_dims)}, block_dim={tuple(block_dims)}", flush=True)

        self._after_call(stream, bound)

        return stream


    @property
    def _func_head(self)->str:
        return self.name + "(" + ", ".join(self._arg_names) + ")"


    @property
    def program(self)->Program:
        return self._program

    
    @property
    def context(self)->Context:
        return self._program.context


    @property
    def name(self)->str:
        return self._name

    
    
    @property
    def args(self)->Dict[str, ArgInfo]:
        return self._args

    @args.setter
    def args(self, args:Dict[str, ArgInfo]):
        self._args = args
        self._arg_list = list(self._args.values())
        self._arg_names = list(self._args.keys())



    @override    
    @staticmethod
    def _prefix()->str:
        return ""


    @override    
    @staticmethod
    def _get_info_func()->object:
        return None


    @override    
    @staticmethod
    def _info_enum()->type|None:
        return None


    @override    
    @staticmethod
    def _release_func():
        return None


__all__ = [
    "Kernel",
    "__call__",
    "__getitem__",
    "__init__",
    "_after_call",
    "_async_executor",
    "_async_executor_lock",
    "_async_workers",
    "_attach_completion",
    "_before_call",
    "_bind_image",
    "_call",
    "_check_args",
    "_copy_to_host",
    "_deliver",
    "_detect_work_size",
    "_func_head",
    "_get_geometry",
    "_get_info_func",
    "_handle_types",
    "_info_enum",
    "_is_vec3_arg",
    "_launch_async",
    "_launch_on",
    "_launch_when_free",
    "_pack_vec3",
    "_pack_vec3_value",
    "_prefix",
    "_prepare_call",
    "_process_arg",
    "_release_func",
    "_release_stream",
    "_set_future_error",
    "_structure_bytes",
    "_take_geometry",
    "args",
    "async_call",
    "complex_of_dtype",
    "context",
    "name",
    "program",
    "submit",
]
