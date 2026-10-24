from __future__ import annotations
import math
from ctypes import c_size_t, pointer, sizeof, Structure, byref, memmove, c_void_p, cast
from functools import partial
import concurrent.futures
import asyncio
from typing import Dict, Any, List, TYPE_CHECKING, Union, Tuple, Optional, override

import numpy as np

from ... import numpy as gnp
from ..driver import (
    HIP,
    HIPInfo,
    CUfunction,
    hipError
)
from .hipobject import HIPObject
from .program_info import ArgInfo
from .stream import Stream
from .mem_object import MemObject
from .program_parser import ProgramParser
from ...utils import detect_work_size, join_with_and, local_size_for_global
from ... import numpy as gnp

if TYPE_CHECKING:
    from .program import Program
    from .context import Context


class Kernel(HIPObject):

    def __init__(self, program:Program, name:str)->None:
        kernel_id = CUfunction()
        ptr_kernel_id = pointer(kernel_id)
        HIP.hipModuleGetFunction(ptr_kernel_id, program.id, name.encode("utf-8"))
        HIPObject.__init__(self, kernel_id)

        self._name:str = name
        self._program:Program = program
        self._args:Dict[str, ArgInfo] = {}
        self._arg_list:List[ArgInfo] = []
        self._arg_names:List[str] = []

        self.type_checked:bool = False
        self._grid_dims:Optional[Tuple[int]] = None
        self._block_dims:Optional[Tuple[int]] = None

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
            
            # block_dims 为 None 表示自动推断(仅指定 grid)
            if block_dims is not None and work_dim != len(block_dims):
                raise ValueError(f"grid_dims={grid_dims} and block_dims={block_dims} do not have same dimension")
        
            self._grid_dims = grid_dims
            self._block_dims = block_dims
        else:
            raise TypeError(f"Kernel.__getitem__ takes at most 2 argument, {len(sizes_or_device)} were given")
        
        return self

    def __call__(self, *args, **kwargs)->None:
        stream = self._call(args, kwargs)
        stream.sync()

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        stream = self._call(args, kwargs)
        return stream.create_future(concurrent.futures.Future)
    
    def async_call(self, *args, **kwargs)->asyncio.Future:
        stream = self._call(args, kwargs)
        return stream.create_future(asyncio.Future)

    def _call(self, args, kwargs)->Stream:
        stream, grid_dims, block_dims, value_ptrs = self.__before_call(args, kwargs)
        HIP.hipModuleLaunchKernel(
            self.id,
            grid_dims[0], grid_dims[1], grid_dims[2],
            block_dims[0], block_dims[1], block_dims[2],
            0, stream.id, value_ptrs, None
        )
        if HIP.print_info:
            print(f"launch {self} on {stream.device} with grid_dim={tuple(grid_dims)}, block_dim={tuple(block_dims)}", flush=True)

        self.__after_call(stream)

        return stream

    def __before_call(self, args, kwargs):
        global_work_size = self._grid_dims
        local_work_size = self._block_dims

        self._grid_dims = None
        self._block_dims = None

        used_kwargs = self.__check_args(args, kwargs)

        stream = Stream()

        value_ptrs = (c_void_p * len(self._args))()
        i = 0
        for key, arg_info in self._args.items():
            value = used_kwargs[key]
            value_ptrs[i] = cast(self._process_arg(stream, arg_info, value), c_void_p)
            i += 1

        if global_work_size is None or local_work_size is None:
            if global_work_size is None:
                global_work_size, local_work_size = self.__detect_work_size()
            else:
                # 仅指定 grid 时,block 按给定 grid 推导(必须整除)
                local_work_size = local_size_for_global(global_work_size)

        # cuLaunchKernel 固定按 3 维传参
        global_work_size = tuple(global_work_size) + (1,) * (3 - len(global_work_size))
        local_work_size = tuple(local_work_size) + (1,) * (3 - len(local_work_size))
        global_work_size = (c_size_t * 3)(*global_work_size)
        local_work_size = (c_size_t * 3)(*local_work_size)
        return stream, global_work_size, local_work_size, value_ptrs

    @staticmethod
    def __copy_to_host(arg_value:Union[np.ndarray, List[Any]], buffer:MemObject):
        host_data = arg_value
        if host_data is not buffer.data:
            if isinstance(host_data, np.ndarray) and getattr(host_data.dtype, "fields", None) is not None:
                # 3 分量向量: 设备端是紧凑布局,解包回结构化字段
                fields:List[str] = list(host_data.dtype.fields.keys())[:3]
                packed:Any = buffer.data
                flat:Any = (packed.reshape(-1, 3) if packed.ndim > 1 else packed)
                count:int = int(np.prod(host_data.shape)) if host_data.ndim > 0 else 1
                for i, f in enumerate(fields):
                    host_data.reshape(-1)[f] = flat[:count, i]
                return
            if isinstance(host_data, list):
                host_data[:] = buffer.data.tolist()
            elif isinstance(host_data, np.ndarray):
                host_data[:] = buffer.data
            elif isinstance(host_data, Structure):
                memmove(byref(host_data), buffer.data.ctypes.data_as(c_void_p), sizeof(host_data))

    def __after_call(self, stream:Stream):
        
        def on_completed(arg_info:ArgInfo, stream:Stream, error_code:int):
            if error_code == hipError.SUCCESS:
                self.__copy_to_host(arg_info.value, arg_info.mem_obj)
                if HIP.print_info:
                    print(f"copy data to host for argument '{arg_info.name}'")

            arg_info.mem_obj.unuse()
        
        for arg_info in self._args.values():
            if not arg_info.need_read_back:
                # 只读/只写参数也要在 kernel 完成后归还缓冲池,否则每次调用泄漏一个池对象
                if arg_info.mem_obj is not None:
                    def on_release(stream:Stream, error_code:int, mem_obj:MemObject=arg_info.mem_obj):
                        mem_obj.unuse()
                    stream.add_callback(on_release)
                continue

            if isinstance(arg_info.value, gnp.ndarray) and not self._is_vec3_arg(arg_info):
                # gnp 数组走延迟回读:数据留在设备上,宿主访问(写操作前)由 to_host() 拉取。
                # 3 分量向量例外: 设备是紧凑布局,延迟回读无法直接解包,走立即回读
                arg_info.mem_obj.dirty_on_device = True
                arg_info.mem_obj.unuse()
                continue

            arg_info.mem_obj.read(stream)
            callback = partial(on_completed, arg_info)
            stream.add_callback(callback)

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
    
    def __detect_work_size(self):
        max_shape = None
        for arg in self._arg_list:
            if arg.need_read_back:
                arg_shape = (arg.mem_obj.data.shape if arg.mem_obj.data is not None else arg.mem_obj.shape)
                if ((
                        arg.base_type_str in HIPInfo.vec_types and
                        len(arg_shape) >= 2 and
                        arg_shape[-1] == HIPInfo.vec_types[arg.base_type_str].__len__(None)
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
            # 所有参数都无需回读时没有可用形状,退化为单元素
            return (1,), (1,)

        if len(max_shape) > 3:
            max_shape = (math.prod(max_shape),)

        total_local_size:int = 128
        global_work_size, block_dims = detect_work_size(total_local_size, max_shape)
        grid_dims = []
        for value1, value2 in zip(global_work_size, block_dims):
            grid_dims.append(value1 // value2)

        return grid_dims, block_dims

    def _process_arg(self, stream:Stream, arg_info:ArgInfo, value:Any)->c_void_p:
        arg_type_str = arg_info.type_str        
        arg_type_str = arg_info.type_str
        base_type_str = arg_info.base_type_str

        if base_type_str in HIPInfo.basic_types or base_type_str in self._program.structs or base_type_str in HIPInfo.vec_types:
            if base_type_str in HIPInfo.basic_types:
                content_type = HIPInfo.basic_types[base_type_str]
                is_struct = False
            elif base_type_str in HIPInfo.vec_types:
                content_type = HIPInfo.vec_types[base_type_str]
                is_struct = False
            else:
                content_type = self._program.structs[arg_info.base_type_str]
                is_struct = True

            is_vec3:bool = (len(getattr(content_type, "_fields_", ())) == 3 and base_type_str in HIPInfo.vec_types)

            if not arg_info.is_ptr:
                if is_vec3:
                    # HIP 同 CUDA: 3 分量向量紧凑布局传参
                    packed:Any = self._pack_vec3(value, content_type)
                    return pointer(packed)

                if isinstance(value, content_type):
                    used_value = value
                else:
                    success = True
                    try:
                        if base_type_str in HIPInfo.alter_types:
                            used_value = content_type(HIPInfo.alter_types[base_type_str][1](value))
                        else:
                            used_value = content_type(value)
                    except BaseException as e:
                        success = False

                    if not success:
                        raise TypeError(f"type of argument '{arg_info.name}' need {content_type.__name__}, got {value.__class__.__name__}, and cannot convert {value.__class__.__name__} to {content_type.__name__}")

                if is_struct:
                    used_value.apply_pointers(stream)

                return pointer(used_value)
            else:
                if is_vec3:
                    # HIP 同 CUDA: float3* 等紧凑布局(元素 12B), 宿主结构化数组
                    # 16B(含 padding 字段) -> 打包/解包转换
                    used_value:Any = self._pack_vec3(value, content_type)
                elif isinstance(value, np.ndarray):
                    used_value = value
                    if value.dtype != content_type:
                        used_value = value.astype(content_type)
                else:
                    ProgramParser._apply_structure_pointers(value, stream)
                    used_value = np.array(value, dtype=content_type)

                if not used_value.flags['C_CONTIGUOUS']:
                    used_value = np.ascontiguousarray(used_value)

                buffer, event = arg_info.use_buffer(stream, used_value)
                arg_info.mem_obj = buffer
                arg_info.value = value
                return pointer(c_void_p(buffer.id.value))
        else:
            raise NotImplementedError(f"arg type {arg_type_str} is legal, but not implemented.")

    def _is_vec3_arg(self, arg_info:ArgInfo)->bool:
        base:str = arg_info.base_type_str
        cls:Any = HIPInfo.vec_types.get(base)
        return (cls is not None and len(cls._fields_) == 3)

    @staticmethod
    def _pack_vec3(value:Any, content_type:Any)->Any:
        """把 float3/int3 等 3 分量数据打包为 CUDA 紧凑布局(去掉 padding 分量)。"""
        fields:List[str] = [f for f, _ in content_type._fields_][:3]
        elem_dtype:Any = content_type.dtype.fields[fields[0]][0]

        if isinstance(value, np.ndarray) and value.dtype.fields is not None:
            count:int = int(np.prod(value.shape)) if value.ndim > 0 else 1
            flat:Any = value.reshape(-1)
            packed:Any = np.empty((count, 3), dtype=elem_dtype)
            for i, f in enumerate(fields):
                packed[:, i] = flat[f]
            return packed.reshape(value.shape + (3,)) if value.ndim > 1 else packed

        if isinstance(value, np.void):
            packed = np.empty(3, dtype=elem_dtype)
            for i, f in enumerate(fields):
                packed[i] = value[f]
            return packed

        if hasattr(value, "x"):
            packed = np.empty(3, dtype=elem_dtype)
            for i, f in enumerate(fields):
                packed[i] = getattr(value, f)
            return packed

        raise TypeError(f"cannot pack {type(value).__name__} as 3-component vector")

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
    def _get_info_func()->HIP.Func:
        return None

    @override    
    @staticmethod
    def _info_enum()->type:
        return None

    @override    
    @staticmethod
    def _release_func():
        return None