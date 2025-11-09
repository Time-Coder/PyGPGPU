from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

from ..runtime import cl_uint
from .clobject import CLObject

if TYPE_CHECKING:
    from .program import Program
    from .context import Context


class Kernel(CLObject):

    @property
    def program(self)->Program: ...
    
    @property
    def context(self)->Context: ...

    @property
    def name(self)->str: ...
    
    @property
    def args(self)->Dict[str, Dict[str, Any]]: ...
    
    @args.setter
    def args(self, args:Dict[str, Dict[str, Any]])->None: ...

    @property
    def function_name(self)->str: ...

    @property
    def num_args(self)->int: ...

    @property
    def reference_count(self)->int: ...

    @property
    def attributes(self)->str: ...