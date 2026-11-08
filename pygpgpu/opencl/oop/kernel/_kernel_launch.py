# The launch pipeline: check, detect the work size, bind, run, copy back.
from __future__ import annotations
from ..device import Device
from ._kernel_support import complex_of_dtype
from ctypes import c_size_t, sizeof, Structure, byref, memmove, c_void_p
from ctypes import Union as ctypes_union
from functools import partial
from typing import Dict, List, Union, Tuple, Optional, cast
import numpy as np
from ...driver import (
    CL,
    CLInfo,
    cl_device_type,
    ErrorCode,
    image2d_t
)
from ..program_info import ArgInfo
from ..command_queue import CommandQueue
from ..event import Event
from ..image2d import image2d
from ..mem_object import MemObject
from ....utils import (detect_work_size, join_with_and, local_size_for_global, device_limits,
                      product)
from .... import numpy as gnp


class _KernelLaunchMixin:

    def _before_call(self, args, kwargs, cmd_queue:Optional[CommandQueue]=None):
        used_device, global_work_size, local_work_size, member_readbacks = self._take_call_state()
        # Kept for _detect_work_size / _after_call, which still consult the
        # per-call state below; the shared attributes are gone.
        self._call_state.member_readbacks = member_readbacks

        used_kwargs = self._check_args(args, kwargs)

        if used_device is None:
            used_device = self._program.context.devices[0]

        if cmd_queue is None:
            cmd_queue = self.context.create_command_queue(used_device)
        bound:Dict[str, object] = {}
        for key, value in used_kwargs.items():
            self._set_arg(cmd_queue, key, value, bound)

        if global_work_size is None or local_work_size is None:
            if global_work_size is None:
                global_work_size, local_work_size = self._detect_work_size(used_device, bound)
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
    def _copy_to_host(arg_value:Union[np.ndarray, image2d_t, List[object]], buffer:MemObject):
        host_data = arg_value
        if isinstance(arg_value, image2d_t):
            host_data = arg_value.data

        if host_data is not buffer.data:
            if isinstance(host_data, list):
                host_data[:] = buffer.data.tolist()
            elif isinstance(host_data, np.ndarray):
                # A complex argument's device buffer holds the record layout, which is the
                # same bytes as the complex array: the copy has to speak the complex dtype
                # or numpy refuses the assignment under its casting rule.
                device_data:object = buffer.data
                if (isinstance(device_data, np.ndarray)
                        and device_data.dtype.fields is not None
                        and host_data.dtype.kind == "c"):
                    complex_dtype:object = complex_of_dtype(device_data.dtype)
                    if complex_dtype is not None and complex_dtype == host_data.dtype:
                        device_data = device_data.view(complex_dtype)
                host_data[:] = device_data
            elif isinstance(host_data, (Structure, ctypes_union)):
                # A union instance is a `ctypes` record too: its bytes are the copy target,
                # the same way a structure's are. Testing for `Structure` alone skipped the
                # write-back, so a kernel that wrote `p.i` left the caller's union unchanged.
                memmove(byref(host_data), buffer.data.ctypes.data_as(c_void_p), sizeof(host_data))


    def _after_call(self, cmd_queue:CommandQueue, launch_event:Event, bound:Dict[str, object])->List[Event]:

        def on_completed(arg_value:Union[np.ndarray, List[object]], mem_obj:MemObject, claim_serial:int, arg_info:ArgInfo, event:Event, error_code:ErrorCode):
            if error_code == ErrorCode.CL_SUCCESS:
                self._copy_to_host(arg_value, mem_obj)
                if CL.print_info:
                    print(f"copy data to host for argument '{arg_info.name}'")

            mem_obj.unuse(claim_serial)

        def on_release(mem_obj:MemObject, claim_serial:int, event:Event, error_code:ErrorCode):
            mem_obj.unuse(claim_serial)

        events:List[Event] = []
        for arg_info in self._args.values():
            entry = bound.get(arg_info.name)
            mem_obj:Optional[MemObject] = entry[0] if entry is not None else None
            arg_value:object = entry[1] if entry is not None else None

            # A gnp array's buffer belongs to the array, not to the context's pool: the next
            # launch naming the same array reuses it, so it is never handed back and
            # `use`/`unuse` must not be run for it.
            if isinstance(arg_value, gnp.ndarray):
                if arg_info.need_read_back:
                    # Deferred read-back: the bytes stay on the device and to_host() pulls
                    # them the next time the host reads the array.
                    arg_value._mark_device_written(mem_obj)
                    continue
                if mem_obj is not None:
                    continue

            if not arg_info.need_read_back:
                # read-only/write-only arguments must also return their buffer to the pool after the kernel completes, otherwise each call leaks one pooled object.
                # The release carries the claim's serial: if the driver fires this callback
                # after the buffer has been handed out again, the serial no longer names
                # the current holder and the release must not free it (see MemObject.unuse).
                if mem_obj is not None:
                    claim_serial:int = mem_obj.claim_serial
                    release_call = partial(on_release, mem_obj, claim_serial)
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

            claim_serial = mem_obj.claim_serial
            event:Event = mem_obj.read(cmd_queue)
            call_back = partial(on_completed, arg_value, mem_obj, claim_serial, arg_info)
            if event.finished:
                call_back(event, event.status)
            else:
                event.on_completed_callbacks.append(call_back)
                events.append(event)

        # `bound` maps argument names to their values; this one key holds a list of
        # records instead, which `_process_arg` appends to (see the `setdefault`
        # below). The cast states that invariant where the list is read back.
        readbacks = cast("List[Dict[str, object]]",
                         bound.get("__member_readbacks__", []))
        for entry in readbacks:
            mem_obj:MemObject = entry["mem_obj"]
            if entry["need_read_back"]:
                claim_serial = mem_obj.claim_serial
                event:Event = mem_obj.read(cmd_queue)
                call_back = partial(on_completed, entry["value"], mem_obj, claim_serial, entry["arg_info"])
                if event.finished:
                    call_back(event, event.status)
                else:
                    event.on_completed_callbacks.append(call_back)
                    events.append(event)
            else:
                claim_serial = mem_obj.claim_serial
                release_call = partial(on_release, mem_obj, claim_serial)
                if launch_event.finished:
                    release_call(launch_event, launch_event.status)
                else:
                    launch_event.on_completed_callbacks.append(release_call)

        return events


    def _check_args(self, args, kwargs)->Dict[str, object]:
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

    
    def _detect_work_size(self, device:Device, bound:Dict[str, object]):
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
