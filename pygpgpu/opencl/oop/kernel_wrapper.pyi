from __future__ import annotations
import concurrent.futures
import asyncio
from typing import Dict, List, TYPE_CHECKING, Tuple, overload

from .program_info import ArgInfo

if TYPE_CHECKING:
    from .program_wrapper import ProgramWrapper

from .kernel import Kernel


class KernelWrapper:

    def __init__(self, program_wrapper:ProgramWrapper, name:str, args:List[ArgInfo], type_checked:bool)->None: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel: ...

    @overload
    def __getitem__(self, device:str)->Kernel: ...

    def __call__(self, *args, **kwargs)->None: ...

    def submit(self, *args, **kwargs)->concurrent.futures.Future: ...
    
    def async_call(self, *args, **kwargs)->asyncio.Future: ...

    @property
    def program_wrapper(self)->ProgramWrapper: ...

    @property
    def name(self)->str: ...
    
    @property
    def args(self)->Dict[str, ArgInfo]: ...