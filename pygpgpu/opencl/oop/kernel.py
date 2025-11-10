from __future__ import annotations
import copy
from  ctypes import c_void_p
from typing import Dict, Any, List, TYPE_CHECKING, Union

import numpy as np

from ..runtime import (
    cl_kernel,
    CL,
    IntEnum,
    CLInfo,
    cl_kernel_info,
    cl_kernel_arg_address_qualifier,
    cl_kernel_arg_type_qualifier,
    cl_mem_flags,
    cl_mem
)
from .clobject import CLObject
from .buffer import Buffer
from ...vectorization import sizeof, value_ptr

if TYPE_CHECKING:
    from .program import Program
    from .context import Context


class Kernel(CLObject):

    def __init__(self, kernel_id:cl_kernel, program:Program)->None:
        CLObject.__init__(self, kernel_id)
        self._program:Program = program
        self._args:Dict[str, Dict[str, Any]] = {}
        self._arg_list:List[Dict[str, Any]] = []
        self._arg_names:List[str] = []

    def __call__(self, *args, **kwargs)->None:
        pass

    def _set_arg(self, index_or_name:Union[int, str], value:Any):
        if isinstance(index_or_name, int):
            arg_info = self._arg_list[index_or_name]
            index = index_or_name
        else:
            arg_info = self._args[index_or_name]
            index = self._arg_names.index(index_or_name)

        arg_type_str = arg_info["type"]
        is_ptr = (arg_type_str[-1] == "*")
        if not is_ptr and "value" in arg_info and arg_info["value"] == value:
            return
        
        arg_type_str = arg_info["type"]
        content_type_str = arg_type_str
        if is_ptr:
            content_type_str = arg_type_str[:-1]

        if arg_type_str in CLInfo.basic_types:
            arg_type = CLInfo.basic_types[arg_type_str]
            used_value = (value if isinstance(value, arg_type) else arg_type(value))
            CL.clSetKernelArg(self.id, index, sizeof(used_value), value_ptr(used_value))
            arg_info["value"] = used_value
        elif is_ptr and content_type_str in CLInfo.basic_types:
            content_type = CLInfo.basic_types[content_type_str]
            if isinstance(value, np.ndarray):
                if value.dtype == content_type:
                    used_value = value
                else:
                    used_value = value.astype(content_type)
            else:
                used_value = np.array(value, dtype=content_type)

            if not used_value.flags['C_CONTIGUOUS']:
                used_value = np.ascontiguousarray(used_value)

            host_ptr = used_value.ctypes.data_as(c_void_p)
            data_size = used_value.nbytes

            if (
                arg_info["access_qualifier"] == cl_kernel_arg_address_qualifier.CL_KERNEL_ARG_ADDRESS_CONSTANT or
                arg_info["type_qualifiers"] & cl_kernel_arg_type_qualifier.CL_KERNEL_ARG_TYPE_CONST
            ):
                flags = cl_mem_flags.CL_MEM_READ_ONLY | cl_mem_flags.CL_MEM_COPY_HOST_PTR
            else:
                flags = cl_mem_flags.CL_MEM_READ_WRITE | cl_mem_flags.CL_MEM_COPY_HOST_PTR

            buffer = Buffer(self.context, flags, data_size, host_ptr)
            CL.clSetKernelArg(self.id, index, sizeof(cl_mem), buffer.id)
            arg_info["value"] = used_value
            arg_info["buffer"] = buffer

        
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
    def args(self)->Dict[str, Dict[str, Any]]:
        return self._args
    
    @args.setter
    def args(self, args:Dict[str, Dict[str, Any]]):
        self._args = copy.deepcopy(args)
        self._arg_list = list(self._args.values())
        self._arg_names = list(self._args.keys())

    @property
    def _prefix(self)->str:
        return "CL_KERNEL"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetKernelInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.kernel_info_types

    @property
    def _info_enum(self)->type:
        return cl_kernel_info

    @staticmethod
    def _release(kernel_id):
        if not kernel_id:
            return
        
        CL.clReleaseKernel(kernel_id)