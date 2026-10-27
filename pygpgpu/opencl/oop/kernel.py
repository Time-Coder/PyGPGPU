from __future__ import annotations
from ctypes import c_size_t, pointer, sizeof, Structure, byref, memmove, c_void_p
from functools import partial
import concurrent.futures
import asyncio
import threading
from typing import Dict, Any, List, TYPE_CHECKING, Union, Tuple, Optional
from typing_extensions import override

import numpy as np

from ..driver import (
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
from .clobject import CLObject
from .program_info import ArgInfo
from .command_queue import CommandQueue
from .event import Event
from .image2d import image2d
from .mem_object import MemObject
from .pipe import pipe
from .program_parser import ProgramParser
from ...utils import detect_work_size, join_with_and, local_size_for_global, device_limits, product
from ... import numpy as gnp

if TYPE_CHECKING:
    from .program import Program
    from .context import Context
    from .device import Device


_async_executor:Optional[concurrent.futures.ThreadPoolExecutor] = None
_async_executor_lock = threading.Lock()


def _async_workers()->concurrent.futures.ThreadPoolExecutor:
    """Lazily create the worker pool that completes asynchronous launches.

    A launch that finds every pooled queue busy is finished here rather than on
    the caller's thread, so submit()/async_call() never block.
    """
    global _async_executor
    with _async_executor_lock:
        if _async_executor is None:
            _async_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="pygpgpu-opencl")
        return _async_executor


def _set_error(future:Any, error:BaseException)->None:
    """Deliver an exception to an asyncio or concurrent future from any thread."""
    if isinstance(future, asyncio.Future):
        loop = future.get_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(future.set_exception, error)
            return
    future.set_exception(error)


def _deliver(future:Any, pending:concurrent.futures.Future)->Optional[BaseException]:
    """Move a completed stream future onto the caller's future; return any error."""
    try:
        error = pending.exception()
    except BaseException as caught:
        error = caught
    if error is None:
        if isinstance(future, asyncio.Future):
            loop = future.get_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(future.set_result, None)
            else:
                future.set_result(None)
        else:
            future.set_result(None)
    else:
        _set_error(future, error)
    return error


def _bridge(completion:Any, future:Any, on_done:Any)->None:
    """Mirror a completion future onto the caller's and then run ``on_done``.

    ``on_done`` returns the queue to the pool, so it must wait until every event
    the launch registered has completed.
    """
    def relay(pending:concurrent.futures.Future)->None:
        _deliver(future, pending)
        on_done()

    completion.add_done_callback(relay)


class Kernel(CLObject):

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
            from .platforms import Platforms
            device = Platforms.device(sizes_or_device[0])
            if device not in self._program.context.devices:
                raise ValueError(f"device {device} is not in current context")
            self._call_state.device = device
        elif len(sizes_or_device) == 2:
            global_sizes = sizes_or_device[0]
            local_sizes = sizes_or_device[1]
            if isinstance(global_sizes, str) or isinstance(local_sizes, str):
                raise TypeError(
                    "the two-element form takes (global_work_size, local_work_size); "
                    "a device name is given as a single-element bracket")
            global_work_size:Tuple[int, ...] = ((global_sizes,) if isinstance(global_sizes, int)
                                                else tuple(global_sizes))
            local_work_size:Optional[Tuple[int, ...]] = (
                (local_sizes,) if isinstance(local_sizes, int)
                else (tuple(local_sizes) if local_sizes is not None else None))

            work_dim = len(global_work_size)
            if work_dim > 3:
                raise ValueError("dimension over 3 is not supported")
            
            if work_dim <= 0:
                raise ValueError("dimension should be at least 1")
            
            # a None local_work_size means automatic inference (global specified only)
            if local_work_size is not None and work_dim != len(local_work_size):
                raise ValueError(f"global_work_size={global_work_size} and local_work_size={local_work_size} do not have same dimension")
        
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
            Event.wait(all_events)
        finally:
            # Every read-back event has completed, so the queue is idle.
            self._release_queue(queue, async_launch=False)

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        return self._launch_async(args, kwargs, concurrent.futures.Future)

    def async_call(self, *args, **kwargs)->asyncio.Future:
        return self._launch_async(args, kwargs, asyncio.Future)

    def _launch_async(self, args, kwargs, future_type:type):
        """Launch without waiting, drawing from the asynchronous queue pool.

        When every pooled queue is busy this returns a pending future at once and
        finishes the launch on a worker thread once a queue frees up, so the
        caller never blocks.
        """
        future = future_type()
        context = self._program.context
        device = self._pending_device() or context.devices[0]
        queue = context.get_async_queue(device, blocking=False)

        if queue is None:
            _async_workers().submit(self._launch_when_free, future, args, kwargs, future_type)
            return future

        try:
            events = self._call(args, kwargs, queue)
        except BaseException as error:
            self._release_queue(queue, async_launch=True)
            future.set_exception(error)
            return future

        _bridge(Event.create_future(events, future_type), future,
                lambda: self._release_queue(queue, async_launch=True))
        return future

    def _launch_when_free(self, future, args, kwargs, future_type:type)->None:
        """Worker-thread half of an asynchronous launch that had to wait."""
        context = self._program.context
        device = context.devices[0]
        try:
            queue = context.get_async_queue(device, blocking=True)
        except BaseException as error:
            _set_error(future, error)
            return

        try:
            events = self._call(args, kwargs, queue)
        except BaseException as error:
            self._release_queue(queue, async_launch=True)
            _set_error(future, error)
            return

        _bridge(Event.create_future(events, future_type), future,
                lambda: self._release_queue(queue, async_launch=True))

    def _release_queue(self, queue:CommandQueue, async_launch:bool)->None:
        context = self._program.context
        if async_launch:
            context.release_async_queue(queue)
        else:
            context.release_sync_queue(queue)

    def _call(self, args, kwargs, cmd_queue:Optional[CommandQueue]=None)->List[Event]:
        cmd_queue, global_work_size, local_work_size, bound = self.__before_call(args, kwargs, cmd_queue)
        work_dim = len(global_work_size)

        event_id = cl_event(0)
        CL.clEnqueueNDRangeKernel(cmd_queue.id, self.id, work_dim, None, global_work_size, local_work_size, 0, None, pointer(event_id))
        launch_event = Event(self.context, event_id, CL.clEnqueueNDRangeKernel)
        if CL.print_info:
            print(f"launch OpenCL {self} on {cmd_queue.device} with global_work_size={tuple(global_work_size)}, local_work_size={tuple(local_work_size)}", flush=True)

        return self.__after_call(cmd_queue, launch_event, bound)

    def __before_call(self, args, kwargs, cmd_queue:Optional[CommandQueue]=None):
        used_device, global_work_size, local_work_size, member_readbacks = self._take_call_state()
        # Kept for __detect_work_size / __after_call, which still consult the
        # per-call state below; the shared attributes are gone.
        self._call_state.member_readbacks = member_readbacks

        used_kwargs = self.__check_args(args, kwargs)

        if used_device is None:
            used_device = self._program.context.devices[0]

        if cmd_queue is None:
            cmd_queue = self.context.create_command_queue(used_device)
        bound:Dict[str, Any] = {}
        for key, value in used_kwargs.items():
            self._set_arg(cmd_queue, key, value, bound)

        if global_work_size is None or local_work_size is None:
            if global_work_size is None:
                global_work_size, local_work_size = self.__detect_work_size(used_device, bound)
            else:
                # when only global is specified, derive local from the given global (must divide evenly)
                max_group, max_items = device_limits(used_device)
                local_work_size = local_size_for_global(
                    global_work_size, max_work_group_size=max_group, max_work_item_sizes=max_items)

        work_dim = len(global_work_size)
        global_work_size = (c_size_t * work_dim)(*global_work_size)
        local_work_size = (c_size_t * work_dim)(*local_work_size)
        return cmd_queue, global_work_size, local_work_size, bound

    @staticmethod
    def __copy_to_host(arg_value:Union[np.ndarray, image2d_t, List[Any]], buffer:MemObject):
        host_data = arg_value
        if isinstance(arg_value, image2d_t):
            host_data = arg_value.data

        if host_data is not buffer.data:
            if isinstance(host_data, list):
                host_data[:] = buffer.data.tolist()
            elif isinstance(host_data, np.ndarray):
                host_data[:] = buffer.data
            elif isinstance(host_data, Structure):
                memmove(byref(host_data), buffer.data.ctypes.data_as(c_void_p), sizeof(host_data))

    def __after_call(self, cmd_queue:CommandQueue, launch_event:Event, bound:Dict[str, Any])->List[Event]:

        def on_completed(arg_value:Union[np.ndarray, List[Any]], mem_obj:MemObject, arg_info:ArgInfo, event:Event, error_code:ErrorCode):
            if error_code == ErrorCode.CL_SUCCESS:
                self.__copy_to_host(arg_value, mem_obj)
                if CL.print_info:
                    print(f"copy data to host for argument '{arg_info.name}'")

            mem_obj.unuse()

        def on_release(mem_obj:MemObject, event:Event, error_code:ErrorCode):
            mem_obj.unuse()

        events:List[Event] = []
        for arg_info in self._args.values():
            entry = bound.get(arg_info.name)
            mem_obj:Optional[MemObject] = entry[0] if entry is not None else None
            arg_value:Any = entry[1] if entry is not None else None

            if not arg_info.need_read_back:
                # read-only/write-only arguments must also return their buffer to the pool after the kernel completes, otherwise each call leaks one pooled object
                if mem_obj is not None:
                    release_call = partial(on_release, mem_obj)
                    if launch_event.finished:
                        release_call(launch_event, launch_event.status)
                    else:
                        launch_event.on_completed_callbacks.append(release_call)
                continue

            # From here on the argument's buffer must exist: lazy read-back, the immediate
            # read-back and the completion callback all dereference it.
            if mem_obj is None:
                raise RuntimeError(
                    f"argument '{arg_info.name}' has no device buffer to read back")

            if isinstance(arg_value, gnp.ndarray):
                # gnp arrays use lazy read-back: data stays on the device, to_host() pulls it on host access (before writes)
                mem_obj.dirty_on_device = True
                mem_obj.unuse()
                continue

            event:Event = mem_obj.read(cmd_queue)
            call_back = partial(on_completed, arg_value, mem_obj, arg_info)
            if event.finished:
                call_back(event, event.status)
            else:
                event.on_completed_callbacks.append(call_back)
                events.append(event)

        for entry in bound.get("__member_readbacks__", []):
            mem_obj:MemObject = entry["mem_obj"]
            if entry["need_read_back"]:
                event:Event = mem_obj.read(cmd_queue)
                call_back = partial(on_completed, entry["value"], mem_obj, entry["arg_info"])
                if event.finished:
                    call_back(event, event.status)
                else:
                    event.on_completed_callbacks.append(call_back)
                    events.append(event)
            else:
                release_call = partial(on_release, mem_obj)
                if launch_event.finished:
                    release_call(launch_event, launch_event.status)
                else:
                    launch_event.on_completed_callbacks.append(release_call)

        return events

    def __check_args(self, args, kwargs)->Dict[str, Any]:
        for key in kwargs:
            if key not in self._args:
                raise TypeError(f"{self._func_head} got an unexpected keyword argument '{key}'")

        used_kwargs = {}
        for i in range(min(len(args), len(self._args))):
            arg_name = self._arg_names[i]
            used_kwargs[arg_name] = args[i]

        for key, value in kwargs.items():
            if key in used_kwargs:
                raise TypeError(f"{self._func_head} got multiple values for argument '{key}'")
            
            used_kwargs[key] = value

        if len(args) > len(self._args):
            raise TypeError(f"{self._func_head} takes {len(self._args)} positional arguments but {len(args)} were given")
        
        if len(used_kwargs) < len(self._args):
            missed_keys = []
            for arg_name in self._args:
                if arg_name not in used_kwargs:
                    missed_keys.append(f"'{arg_name}'")

            if len(missed_keys) == 1:
                word = "argument"
            else:
                word = "arguments"

            raise TypeError(f"{self._func_head} missing {len(missed_keys)} required positional {word}: {join_with_and(missed_keys)}")

        if self.type_checked:
            for key, value in used_kwargs.items():
                arg_info = self._args[key]
                arg_info.check_type(value)

        return used_kwargs
    
    def __detect_work_size(self, device:Device, bound:Dict[str, Any]):
        max_shape = None
        for arg in self._arg_list:
            if not arg.need_read_back:
                continue
            # The shape comes from this call's binding: argument bindings are per call
            # since launches stopped sharing state between threads, and ArgInfo no
            # longer carries a mem_obj to read a shape from.
            entry = bound.get(arg.name)
            if entry is None:
                continue
            mem_obj:Optional[MemObject] = entry[0]
            if mem_obj is None:
                # Local-memory slots and image handles carry no buffer; an image pair
                # still knows its shape, everything else has nothing to contribute.
                arg_shape:Optional[Tuple[int, ...]] = getattr(entry[1], "shape", None)
                if arg_shape is None:
                    continue
                arg_shape = tuple(arg_shape)
            else:
                arg_shape = (mem_obj.data.shape if mem_obj.data is not None else mem_obj.shape)

            if ((
                    arg.base_type_str in CLInfo.vec_types and
                    len(arg_shape) >= 2 and
                    arg_shape[-1] == CLInfo.vec_types[arg.base_type_str].component_count()
                ) or (
                    len(arg_shape) == 3 and isinstance(mem_obj, image2d)
                )
            ):
                arg_shape = arg_shape[:-1]

            if not arg_shape:
                arg_shape = (1,)
                
            if max_shape is None:
                max_shape = arg_shape
            elif len(arg_shape) > len(max_shape):
                max_shape = arg_shape
            elif len(arg_shape) == len(max_shape):
                for n1, n2 in zip(arg_shape, max_shape):
                    if n1 > n2:
                        max_shape = arg_shape
                        break

        if max_shape is None:
            # no argument needs read-back so there is no usable shape; degrade to a single element
            return (1,), (1,)

        if len(max_shape) > 3:
            max_shape = (product(max_shape),)

        total_local_size:int = 256
        if device.type == cl_device_type.CL_DEVICE_TYPE_GPU:
            if "nvidia" in device.vendor.lower():
                total_local_size:int = 128
            elif "amd" in device.vendor.lower():
                total_local_size:int = 256
            elif "intel" in device.vendor.lower():
                total_local_size:int = 64
        elif device.type == cl_device_type.CL_DEVICE_TYPE_CPU:
            total_local_size:int = 64

        # Clamp to the device's real work-group limits instead of trusting the vendor guess
        max_group, max_items = device_limits(device)
        global_work_size, local_work_size = detect_work_size(
            total_local_size, max_shape, max_work_group_size=max_group, max_work_item_sizes=max_items)
        return global_work_size, local_work_size

    def _set_arg(self, cmd_queue:CommandQueue, index_or_name:Union[int, str], value:Any, bound:Dict[str, Any]):
        if isinstance(index_or_name, int):
            arg_info = self._arg_list[index_or_name]
            index = index_or_name
        else:
            arg_info = self._args[index_or_name]
            index = self._arg_names.index(index_or_name)

        arg_type_str = arg_info.type_str        
        arg_type_str = arg_info.type_str
        type_qualifiers = arg_info.type_qualifiers
        base_type_str = arg_info.base_type_str

        if type_qualifiers & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_PIPE:
            arg_type = CLInfo.basic_types[arg_type_str]
            
            if not isinstance(value, pipe):
                raise TypeError(f"type of argument '{arg_info.name}' must be pipe<{arg_type.__name__}>, got {value.__class__.__name__}.")
            
            if value.packet_type is not None and value.packet_type != arg_type:
                raise TypeError(f"type of argument '{arg_info.name}' must be pipe<{arg_type.__name__}>, got pipe<{value.packet_type.__name__}>.")

            if value.packet_type is None:
                value.packet_type = arg_type

            used_value = self.context.get_pipe(value)
            CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
        elif arg_type_str == "queue_t":
            if not isinstance(value, queue_t):
                raise TypeError(f"type of argument '{arg_info.name}' must be queue_t, got {value.__class__.__name__}.")
            
            used_value = self.context.get_queue(value)
            CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
        elif arg_type_str == "sampler_t":
            if not isinstance(value, sampler_t):
                raise TypeError(f"type of argument '{arg_info.name}' must be sampler_t, got {value.__class__.__name__}.")
            
            used_value = self.context.get_sampler(value)
            CL.clSetKernelArg(self.id, index, sizeof(used_value.id), pointer(used_value.id))
        elif base_type_str in CLInfo.basic_types or base_type_str in self._program.structs or base_type_str in CLInfo.vec_types:
            if base_type_str in CLInfo.basic_types:
                content_type = CLInfo.basic_types[base_type_str]
                is_struct = False
            elif base_type_str in CLInfo.vec_types:
                content_type = CLInfo.vec_types[base_type_str]
                is_struct = False
            else:
                content_type = self._program.structs[arg_info.base_type_str]
                is_struct = True

            if not arg_info.is_ptr:
                if isinstance(value, content_type):
                    used_value = value
                else:
                    success = True
                    try:
                        if base_type_str in CLInfo.alter_types:
                            used_value = content_type(CLInfo.alter_types[base_type_str][1](value))
                        else:
                            used_value = content_type(value)
                    except BaseException:
                        success = False

                    if not success:
                        raise TypeError(f"type of argument '{arg_info.name}' need {content_type.__name__}, got {value.__class__.__name__}, and cannot convert {value.__class__.__name__} to {content_type.__name__}")

                if is_struct:
                    bound.setdefault("__member_readbacks__", []).extend(
                        used_value.apply_pointers(cmd_queue, arg_info.member_infos or None))

                CL.clSetKernelArg(self.id, index, sizeof(used_value), pointer(used_value))
            else:
                if arg_info.address_qualifier == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_LOCAL:
                    if isinstance(value, int):
                        bytes_per_group = value
                    elif isinstance(value, np.ndarray):
                        bytes_per_group = value.size
                    elif isinstance(value, (list, tuple)):
                        bytes_per_group = np.array(value).size

                    bytes_per_group *= sizeof(content_type)                    
                    CL.clSetKernelArg(self.id, index, bytes_per_group, None)
                    bound[arg_info.name] = (None, value, None)
                else:
                    if isinstance(value, np.ndarray):
                        used_value = value
                        content_dtype = getattr(content_type, "dtype", None) if is_struct else None
                        if content_dtype is not None:
                            if value.dtype != content_dtype:
                                used_value = np.ascontiguousarray(value).view(content_dtype)
                        elif value.dtype != content_type:
                            used_value = value.astype(content_type)
                    else:
                        ProgramParser._apply_structure_pointers(value, cmd_queue)
                        used_value = np.array(value, dtype=content_type)

                    if not used_value.flags['C_CONTIGUOUS']:
                        used_value = np.ascontiguousarray(used_value)

                    buffer, event = arg_info.use_buffer(cmd_queue, used_value)
                    CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(buffer.id))
                    bound[arg_info.name] = (buffer, value, used_value)
        elif arg_type_str in CLInfo.image_types:
            image_type = CLInfo.image_types[arg_type_str]
            # A backend-neutral Image becomes the descriptor this backend binds, with
            # its channel count and element type taken from the image itself.
            from ...image import Image as _Image
            original_image:Any = value if isinstance(value, _Image) else None
            if original_image is not None:
                # Only CLAMP is portable, and the sampler is built during translation, so
                # this repeats the check to fail with the same message if that path was
                # skipped (a cached kernel, for instance).
                from ...kernel.decorator import _check_image_address
                _check_image_address(original_image, "opencl")
                value = image_type(
                    (original_image.data if original_image.data is not None
                     else np.zeros(original_image.shape, original_image.dtype)),
                    dtype=original_image.dtype)
            if not isinstance(value, (np.ndarray, image_type)):
                raise TypeError(f"type of argument '{arg_info.name}' must be {image_type.__name__}, np.ndarray or Image, got {value.__class__.__name__}.")

            used_value = value
            if isinstance(value, np.ndarray):
                used_value = image_type(value)

            image, event = arg_info.use_image(cmd_queue, used_value)
            CL.clSetKernelArg(self.id, index, sizeof(cl_mem), pointer(image.id))
            bound[arg_info.name] = (image, value, used_value)
            if original_image is not None:
                # Record where the result lives so Image.read() can fetch it, and note
                # which backend wrote it last.
                original_image._bindings["opencl"] = (image, cmd_queue)
                original_image._last_backend = "opencl"
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")

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
    def _get_info_func()->Any:
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
