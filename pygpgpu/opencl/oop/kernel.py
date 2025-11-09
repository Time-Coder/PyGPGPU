from __future__ import annotations
import copy
from typing import Dict, Any, TYPE_CHECKING

from ..runtime import (
    cl_kernel,
    CL,
    IntEnum,
    CLInfo,
    cl_kernel_info
)
from .clobject import CLObject

if TYPE_CHECKING:
    from .program import Program
    from .context import Context


class Kernel(CLObject):

    def __init__(self, kernel_id:cl_kernel, program:Program)->None:
        CLObject.__init__(self, kernel_id)
        self._program:Program = program
        self._args:Dict[str, Dict[str, Any]] = {}

    def _set_arg(self, index:int, value:Any):
        

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