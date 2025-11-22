from __future__ import annotations

from .clobject import CLObject
import concurrent.futures
import asyncio
from typing import Optional, List, TYPE_CHECKING, Callable, Union, overload

from ..runtime import (
    cl_event,
    cl_command_execution_status,
    cl_command_type,
    ptr_cl_event,
    ErrorCode,
    CL
)

if TYPE_CHECKING:
    from .context import Context


class Event(CLObject):

    def __init__(self, context:Context, event_id:Optional[cl_event]=None, func:Optional[CL.Func]=None): ...

    def wait(self)->None: ...

    @overload
    @staticmethod
    def wait(events:List[Event])->None: ...

    @staticmethod
    def create_future(events:List[Event], future_type:type)->Union[concurrent.futures.Future, asyncio.Future]: ...

    @staticmethod
    def events_ptr(events:List[Event])->ptr_cl_event: ...

    @property
    def context(self)->Context: ...

    @property
    def status(self)->cl_command_execution_status: ...

    @property
    def finished(self)->bool: ...
    
    @status.setter
    def status(self, status:cl_command_execution_status)->None: ...

    @property
    def error_message(self)->str: ...

    @property
    def command_type(self)->cl_command_type: ...

    @property
    def reference_count(self)->int: ...

    @property
    def on_status_changed_callbacks(self)->List[Callable[[Event, Union[ErrorCode, cl_command_execution_status]], None]]: ...

    @property
    def on_completed_callbacks(self)->List[Callable[[Event, ErrorCode], None]]: ...

    @property
    def on_submitted_callbacks(self)->List[Callable[[Event], None]]: ...

    @property
    def on_started_callbacks(self)->List[Callable[[Event], None]]: ...
