from typing import Tuple, List, Optional, Dict, Any

from ..runtime import cl_context_properties
from .clobject import CLObject
from .device import Device
from .platform import Platform
from .program import Program


class Context(CLObject):

    def __init__(self, *devices): ...

    def compile(self, file_name:str, includes:Optional[List[str]] = None, defines:Optional[Dict[str, Any]] = None, options:Optional[List[str]]=None)->Program:...

    @property
    def devices(self)->Tuple[Device]: ...
    
    @property
    def platform(self)->Platform: ...

    @property
    def reference_count(self)->int: ...

    @property
    def num_devices(self)->int: ...

    @property
    def n_devices(self)->int: ...

    @property
    def device_ids(self)->List[int]: ...

    @property
    def properties(self)->List[cl_context_properties]: ...

    @property
    def d3d10_prefer_shared_resources(self)->bool: ...

    @property
    def d3d11_prefer_shared_resources(self)->bool: ...