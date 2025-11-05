from typing import Tuple, List

from ..runtime import cl_context_properties
from .clobject import CLObject
from .device import Device
from .platform import Platform


class Context(CLObject):

    def __init__(self, *devices): ...

    @property
    def devices(self)->Tuple[Device]: ...
    
    @property
    def platform(self)->Platform: ...

    @property
    def reference_count(self)->int: ...

    @property
    def num_devices(self)->int: ...

    @property
    def properties(self)->List[cl_context_properties]: ...

    @property
    def d3d10_prefer_shared_resources(self)->bool: ...

    @property
    def d3d11_prefer_shared_resources(self)->bool: ...