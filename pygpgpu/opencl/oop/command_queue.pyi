from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .context import Context
    from .device import Device

from ..runtime import (
    cl_command_queue,
    cl_command_queue_properties,
    cl_queue_properties
)
from .clobject import CLObject


class CommandQueue(CLObject):

    def __init__(self, context:Context, device:Device, properties:Optional[cl_command_queue_properties]=None): ...

    def wait(self)->None: ...

    @property
    def context(self)->Context: ...
    
    @property
    def device(self)->Device: ...

    @property
    def reference_count(self)->int: ...

    @property
    def properties(self)->cl_command_queue_properties: ...

    @property
    def properties_array(self)->List[cl_queue_properties]: ...

    @property
    def size(self)->int: ...

    @property
    def device_default(self)->cl_command_queue: ...

    def __len__(self)->int: ...