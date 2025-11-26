from __future__ import annotations
from typing import Dict, TYPE_CHECKING, Tuple, overload

from ..runtime import cl_kernel
from .clobject import CLObject
from .program_info import ArgInfo
import concurrent.futures
import asyncio

if TYPE_CHECKING:
    from .program import Program
    from .context import Context


class Kernel(CLObject):

    def __init__(self, program:Program, kernel_id:cl_kernel)->None: ...

    @property
    def program(self)->Program: ...
    
    @property
    def context(self)->Context: ...

    @property
    def name(self)->str: ...
    
    @property
    def args(self)->Dict[str, ArgInfo]: ...
    
    @args.setter
    def args(self, args:Dict[str, ArgInfo])->None: ...

    @property
    def function_name(self)->str: ...

    @property
    def num_args(self)->int: ...

    @property
    def reference_count(self)->int: ...

    @property
    def attributes(self)->str: ...

    @property
    def type_checked(self)->bool: ...

    @type_checked.setter
    def type_checked(self, type_checked:bool)->None: ...
    
    def __call__(self, *args, **kwargs)->None: ...

    def submit(self, *args, **kwargs)->concurrent.futures.Future: ...
    
    def async_call(self, *args, **kwargs)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int,int], Tuple[int,int]])->Kernel: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int,int,int], Tuple[int,int,int]])->Kernel: ...

    @overload
    def __getitem__(self, device:str)->Kernel: ...