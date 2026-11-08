# The launch pipeline: check, prepare, run, copy back.
from __future__ import annotations
from ._kernel_support import complex_of_dtype
from ctypes import c_size_t, sizeof, Structure, byref, memmove, c_void_p, cast
from ctypes import Union as ctypes_union
from functools import partial
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
from .... import numpy as gnp
from ...driver import (
    CUDA,
    CUInfo,
    CUresult
)
from ..program_info import ArgInfo
from ..stream import Stream
from ..mem_object import MemObject
from ....utils import detect_work_size, join_with_and, local_size_for_global, device_limits, product


class _KernelLaunchMixin:

    def _prepare_call(self, args, kwargs, stream:Stream):
        """Bind the arguments to ``stream`` and resolve the launch geometry.

        Split out from _before_call so an asynchronous launch can take its stream
        later (on a worker) while still binding the arguments at call time.
        """
        global_work_size, local_work_size = self._take_geometry()

        used_kwargs = self._check_args(args, kwargs)

        # Argument bindings for this launch only. They used to be stashed on the
        # shared ArgInfo objects, which two threads launching the same kernel
        # would overwrite under each other -- producing silently wrong results.
        bound:Dict[str, object] = {}
        seen_arrays:Dict[int, Tuple[MemObject, object]] = {}
        value_ptrs = (c_void_p * len(self._args))()
        i = 0
        for key, arg_info in self._args.items():
            value = used_kwargs[key]
            value_ptrs[i] = cast(self._process_arg(
                stream, arg_info, value, bound, seen_arrays), c_void_p)
            i += 1

        if global_work_size is None or local_work_size is None:
            if global_work_size is None:
                # The shape comes from this call's binding (see the OpenCL backend): the
                # per-ArgInfo mem_obj the detection used to read stopped being filled
                # when bindings became per-call, so a bare call launched (1, 1, 1) and
                # silently wrote one element.
                global_work_size, local_work_size = self._detect_work_size(bound)
            else:
                # When only grid is given, derive block from that grid (must divide evenly)
                max_group, max_items = device_limits(self._program.context.device)
                local_work_size = local_size_for_global(
                    global_work_size, max_work_group_size=max_group, max_work_item_sizes=max_items)

        # cuLaunchKernel always takes 3 dimensions
        global_work_size = tuple(global_work_size) + (1,) * (3 - len(global_work_size))
        local_work_size = tuple(local_work_size) + (1,) * (3 - len(local_work_size))
        global_work_size = (c_size_t * 3)(*global_work_size)
        local_work_size = (c_size_t * 3)(*local_work_size)
        return global_work_size, local_work_size, value_ptrs, bound


    def _before_call(self, args, kwargs, async_launch:bool=False):
        context = self._program.context
        if async_launch:
            stream = context.get_async_stream(blocking=True)
        else:
            stream = context.get_sync_stream()

        grid_dims, block_dims, value_ptrs, bound = self._prepare_call(args, kwargs, stream)
        return stream, grid_dims, block_dims, value_ptrs, bound


    @staticmethod
    def _copy_to_host(arg_value:Union[np.ndarray, List[object]], buffer:MemObject):
        host_data = arg_value
        if host_data is not buffer.data:
            if isinstance(host_data, np.ndarray) and getattr(host_data.dtype, "fields", None) is not None:
                # 3-component vector: device uses a compact layout, unpack back into structured fields
                fields:List[str] = list(host_data.dtype.fields.keys())[:3]
                packed:object = buffer.data
                flat:object = (packed.reshape(-1, 3) if packed.ndim > 1 else packed)
                count:int = int(np.prod(host_data.shape)) if host_data.ndim > 0 else 1
                for i, f in enumerate(fields):
                    host_data.reshape(-1)[f] = flat[:count, i]
                return
            if isinstance(host_data, list):
                host_data[:] = buffer.data.tolist()
            elif isinstance(host_data, np.ndarray):
                # A complex argument's device buffer holds the record layout, whose bytes are
                # the complex array's own: the copy has to speak the complex dtype, or numpy
                # refuses the assignment under its casting rule.
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


    def _after_call(self, stream:Stream, bound:Dict[str, object]):
        
        def on_completed(arg_info:ArgInfo, arg_value:object, mem_obj:MemObject, stream:Stream, error_code:CUresult):
            if error_code == CUresult.CUDA_SUCCESS:
                self._copy_to_host(arg_value, mem_obj)
                if CUDA.print_info:
                    print(f"copy data to host for argument '{arg_info.name}'")

            mem_obj.unuse()
        
        for arg_info in self._args.values():
            entry = bound.get(arg_info.name)
            mem_obj:Optional[MemObject] = entry[0] if entry is not None else None
            arg_value:object = entry[1] if entry is not None else None

            # A gnp array's buffer belongs to the array, not to the context's pool: it is
            # reused by the next launch that names the same array, so it is never handed
            # back, and `use`/`unuse` (which exist to police one pool claim per launch)
            # must not be run for it.
            if isinstance(arg_value, gnp.ndarray):
                if arg_info.need_read_back and not self._is_vec3_arg(arg_info):
                    # Deferred read-back: the bytes stay on the device and to_host() pulls
                    # them the next time the host reads the array. 3-component vectors are
                    # the exception -- the device layout is compact, so their bytes have to
                    # be unpacked immediately below.
                    arg_value._mark_device_written(mem_obj)
                    continue
                if mem_obj is not None and not arg_info.need_read_back:
                    continue

            if not arg_info.need_read_back:
                # Read-only/write-only arguments must also return their buffer to the pool once the kernel finishes, otherwise each call leaks one pooled object
                if mem_obj is not None:
                    def on_release(stream:Stream, error_code:CUresult, mem_obj:MemObject=mem_obj):
                        mem_obj.unuse()
                    stream.add_callback(on_release)
                continue

            # From here on the argument's buffer must exist: deferred read-back, the immediate
            # read-back and the completion callback all dereference it.
            if mem_obj is None:
                raise RuntimeError(
                    f"argument '{arg_info.name}' has no device buffer to read back")

            mem_obj.read(stream)
            callback = partial(on_completed, arg_info, arg_value, mem_obj)
            stream.add_callback(callback)


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

    
    def _detect_work_size(self, bound:Dict[str, object]):
        max_shape = None
        for arg in self._arg_list:
            if arg.need_read_back:
                # The binding for this launch is recorded separately (see
                # _prepare_call), so a shared ArgInfo may carry no buffer at all --
                # reading arg.mem_obj answered None for every argument and the launch
                # degraded to a single work item.
                entry = bound.get(arg.name)
                if entry is None:
                    continue
                mem_obj:object = entry[0]
                if mem_obj is None:
                    # A texture or surface handle carries no buffer; a scalar argument
                    # carries no shape. Neither contributes a launch extent.
                    arg_shape:Optional[Tuple[int, ...]] = getattr(entry[1], "shape", None)
                    if arg_shape is None:
                        continue
                    arg_shape = tuple(arg_shape)
                else:
                    arg_shape = (mem_obj.data.shape if mem_obj.data is not None else mem_obj.shape)
                if ((
                        arg.base_type_str in CUInfo.vec_types and
                        len(arg_shape) >= 2 and
                        arg_shape[-1] == CUInfo.vec_types[arg.base_type_str].component_count()
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
            # With no argument needing read-back there is no usable shape, so fall back to a single element
            return (1,), (1,)

        if len(max_shape) > 3:
            max_shape = (product(max_shape),)

        total_local_size:int = 128
        max_group, max_items = device_limits(self._program.context.device)
        global_work_size, block_dims = detect_work_size(
            total_local_size, max_shape, max_work_group_size=max_group, max_work_item_sizes=max_items)
        grid_dims = []
        for value1, value2 in zip(global_work_size, block_dims):
            grid_dims.append(value1 // value2)

        return grid_dims, block_dims
