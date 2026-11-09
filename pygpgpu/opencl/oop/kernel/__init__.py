from __future__ import annotations
from ctypes import c_size_t, pointer, sizeof, Structure, byref, memmove, c_void_p
from ctypes import Union as ctypes_union
from functools import partial
import concurrent.futures
import asyncio
import threading
from typing import Dict, List, TYPE_CHECKING, Union, Tuple, Optional, cast
from typing_extensions import override
import numpy as np
from ...driver import (
    cl_kernel,
    CL,
    IntEnum,
    CLInfo,
    cl_kernel_info,
    cl_kernel_arg_address_qualifier,
    cl_mem,
    cl_device_type,
    cl_event,
    ErrorCode,
    sampler_t,
    queue_t,
    image2d_t,
    cl_kernel_arg_type_qualifier
)
from ..clobject import CLObject
from ..program_info import ArgInfo
from ..command_queue import CommandQueue
from ..event import Event
from ..image2d import image2d
from ..mem_object import MemObject
from ..pipe import pipe
from ..program_parser import ProgramParser
from ....utils import (detect_work_size, join_with_and, local_size_for_global, device_limits,
                      product, blocks_to_global)
from .... import numpy as gnp
from ._kernel_args import _KernelArgsMixin
from ._kernel_launch import _KernelLaunchMixin
from ._kernel_async import _KernelAsyncMixin


if TYPE_CHECKING:
    from ..program import Program
    from ..context import Context
    from ..device import Device












class Kernel(_KernelArgsMixin, _KernelLaunchMixin, _KernelAsyncMixin, CLObject):

    def __init__(self, program:Program, kernel_id:cl_kernel)->None:
        CLObject.__init__(self, kernel_id)
        self._program:Program = program
        self._args:Dict[str, ArgInfo] = {}
        self._arg_list:List[ArgInfo] = []
        self._arg_names:List[str] = []

        self.type_checked:bool = False
        # Per-call bracket state (device / work size / member read-backs), set by
        # __getitem__ and consumed by the next launch. Thread-local because one
        # compiled Kernel instance is shared by every thread using the kernel; as
        # plain attributes one thread's bracket would leak into another's launch.
        self._call_state = threading.local()
        # `clSetKernelArg` binds the arguments onto the shared kernel handle, so the
        # bind-and-enqueue step is serialized (see _call).
        self._arg_lock = threading.Lock()


    def _take_call_state(self):
        """Read and clear the pending bracket state for the current thread."""
        state = self._call_state
        taken = (
            getattr(state, "device", None),
            getattr(state, "global_work_size", None),
            getattr(state, "local_work_size", None),
            getattr(state, "member_readbacks", None) or [],
        )
        state.device = None
        state.global_work_size = None
        state.local_work_size = None
        state.member_readbacks = []
        return taken


    def _pending_device(self) -> Optional[Device]:
        """The device a bracket selected, without consuming it."""
        return getattr(self._call_state, "device", None)


    def __getitem__(self, sizes_or_device:Tuple[Union[int, Tuple[int,...], str], ...])->Kernel:
        if not isinstance(sizes_or_device, tuple):
            sizes_or_device = (sizes_or_device,)

        if len(sizes_or_device) == 0:
            raise TypeError("Kernel.__getitem__ takes at least 1 argument, 0 were given")
        elif len(sizes_or_device) == 1:
            from ..platforms import Platforms
            device = Platforms.device(sizes_or_device[0])
            if device not in self._program.context.devices:
                raise ValueError(f"device {device} is not in current context")
            self._call_state.device = device
        elif len(sizes_or_device) == 2:
            blocks = sizes_or_device[0]
            threads = sizes_or_device[1]
            if isinstance(blocks, str) or isinstance(threads, str):
                raise TypeError(
                    "the two-element form takes (blocks_per_grid, threads_per_block); "
                    "a device name is given as a single-element bracket")
            block_dims:Tuple[int, ...] = ((blocks,) if isinstance(blocks, int)
                                          else tuple(blocks))
            thread_dims:Optional[Tuple[int, ...]] = (
                (threads,) if isinstance(threads, int)
                else (tuple(threads) if threads is not None else None))

            work_dim = len(block_dims)
            if work_dim > 3:
                raise ValueError("dimension over 3 is not supported")

            if work_dim <= 0:
                raise ValueError("dimension should be at least 1")

            # a None thread count means automatic inference (a block count alone)
            if thread_dims is None:
                global_work_size:Tuple[int, ...] = block_dims
                local_work_size:Optional[Tuple[int, ...]] = None
            else:
                if work_dim != len(thread_dims):
                    raise ValueError(
                        f"blocks={block_dims} and threads={thread_dims} do not have the "
                        f"same dimension")
                # The bracket is CUDA's spelling on every backend (see
                # utils.blocks_to_global): a grid of `blocks` groups, each `threads` wide.
                # OpenCL wants the total item count and one group's width, so the two are
                # computed here and nothing below has to know which spelling arrived.
                global_work_size, local_work_size = blocks_to_global(block_dims, thread_dims)

            self._call_state.global_work_size = global_work_size
            self._call_state.local_work_size = local_work_size
        else:
            raise TypeError(f"Kernel.__getitem__ takes at most 2 argument, {len(sizes_or_device)} were given")
        
        return self


    def __call__(self, *args, **kwargs)->None:
        context = self._program.context
        device = self._pending_device() or context.devices[0]
        queue = context.get_sync_queue(device)
        try:
            all_events = self._call(args, kwargs, queue)
            # `_call` answers the *read-back* events, and a pass whose arguments are all
            # deferred (`gnp` arrays, which include the arrays a multi-pass boundary keeps on
            # the device) produces none -- so waiting on that list alone returned while the
            # launch was still running. Anything a caller does next with those bytes is then
            # unordered against it, which is how a deferred flush downloaded a buffer the
            # kernel had not written yet (an all-zero answer on a correct launch). Finishing
            # the queue covers the launch and every read-back enqueued behind it, which is
            # what CUDA's `stream.sync()` gives the other backends.
            queue.wait()
            Event.wait(all_events)
        finally:
            # Every read-back event has completed, so the queue is idle.
            self._release_queue(queue, async_launch=False)


    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        return self._launch_async(args, kwargs, concurrent.futures.Future)


    def async_call(self, *args, **kwargs)->asyncio.Future:
        return self._launch_async(args, kwargs, asyncio.Future)


    def _call(self, args, kwargs, cmd_queue:Optional[CommandQueue]=None)->List[Event]:
        # `clSetKernelArg` writes the argument values *onto the kernel handle itself*, and
        # this handle is shared by every thread that launches this kernel -- the bracket
        # state is per thread (`_call_state`), but the handle is not. Two threads that set
        # their arguments at the same time therefore overwrite each other's, and the launch
        # that ends up enqueued reads the other thread's buffers: the kernel then writes into
        # a buffer nobody reads back, and the caller's own array stays at its initial value
        # (an all-zero result on a correct launch). Binding the arguments and enqueuing are
        # one critical section, so a thread's arguments are the ones its launch runs with.
        # The queue itself is already per thread, so the wait for the event stays outside.
        with self._arg_lock:
            cmd_queue, global_work_size, local_work_size, bound = self._before_call(args, kwargs, cmd_queue)
            work_dim = len(global_work_size)

            event_id = cl_event(0)
            CL.clEnqueueNDRangeKernel(cmd_queue.id, self.id, work_dim, None, global_work_size, local_work_size, 0, None, pointer(event_id))
            launch_event = Event(self.context, event_id, CL.clEnqueueNDRangeKernel)
            if CL.print_info:
                print(f"launch OpenCL {self} on {cmd_queue.device} with global_work_size={tuple(global_work_size)}, local_work_size={tuple(local_work_size)}", flush=True)

        return self._after_call(cmd_queue, launch_event, bound)


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
        return self.function_name

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
        return "CL_KERNEL"


    @override    
    @staticmethod
    def _get_info_func()->object:
        return CL.clGetKernelInfo


    @override    
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]|None:
        return CLInfo.kernel_info_types


    @override    
    @staticmethod
    def _info_enum()->type|None:
        return cl_kernel_info


    @override    
    @staticmethod
    def _release_func():
        return CL.clReleaseKernel


__all__ = [
    "Kernel",
    "__call__",
    "__getitem__",
    "__init__",
    "_after_call",
    "_async_executor",
    "_async_executor_lock",
    "_async_workers",
    "_before_call",
    "_bridge",
    "_call",
    "_check_args",
    "_copy_to_host",
    "_deliver",
    "_detect_work_size",
    "_element_dtype",
    "_func_head",
    "_get_info_func",
    "_host_dtype_for",
    "_info_enum",
    "_info_types_map",
    "_launch_async",
    "_launch_when_free",
    "_pending_device",
    "_prefix",
    "_release_func",
    "_release_queue",
    "_set_arg",
    "_set_error",
    "_structure_bytes",
    "_take_call_state",
    "args",
    "async_call",
    "complex_of_dtype",
    "context",
    "name",
    "program",
    "submit",
]
