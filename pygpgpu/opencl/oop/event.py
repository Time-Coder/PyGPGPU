from __future__ import annotations

from .clobject import CLObject
from ctypes import pointer, c_void_p, cast, POINTER, py_object
import concurrent.futures
import asyncio
import traceback
from typing import Dict, Optional, Callable, List, TYPE_CHECKING, Union, override, Set

from ..runtime import (
    CL,
    cl_event,
    IntEnum,
    CLInfo,
    cl_event_info,
    cl_int,
    cl_command_execution_status,
    CL_EVENT_NOTIFY_CALLBACK,
    ptr_cl_event,
    ErrorCode
)

if TYPE_CHECKING:
    from .context import Context


class Event(CLObject):

    __retained_events:Set[Event] = set()

    def __init__(self, context:Context, event_id:Optional[cl_event]=None, func:Optional[CL.Func]=None):
        error_code = cl_int(0)
        self._context:Context = context
        self._is_user_event:bool = False
        self._func:Optional[CL.Func] = func
        if not event_id:
            event_id = CL.clCreateUserEvent(context, pointer(error_code))
            self._is_user_event:bool = True

        CLObject.__init__(self, event_id)
        Event.__retained_events.add(self)

        self.on_status_changed_callbacks:List[Callable[[Event, Union[cl_command_execution_status, ErrorCode]], None]] = []
        self.on_submitted_callbacks:List[Callable[[Event], None]] = []
        self.on_completed_callbacks:List[Callable[[Event, ErrorCode], None]] = []
        self.on_started_callbacks:List[Callable[[Event], None]] = []

        user_data = cast(pointer(py_object(self)), c_void_p)
        CL.clSetEventCallback(self.id, cl_command_execution_status.CL_SUBMITTED, Event._pfn_notify, user_data)
        CL.clSetEventCallback(self.id, cl_command_execution_status.CL_RUNNING, Event._pfn_notify, user_data)
        CL.clSetEventCallback(self.id, cl_command_execution_status.CL_COMPLETE, Event._pfn_notify, user_data)
        
        status = self.status
        if status == cl_command_execution_status.CL_COMPLETE or isinstance(status, ErrorCode):
            try:
                Event.__retained_events.remove(self)
            except:
                pass

    def wait(events:Union[Event, List[Event]])->None:
        if isinstance(events, Event):
            events = [events]

        CL.clWaitForEvents(len(events), Event.events_ptr(events))
        error_messages:List[str] = []
        for event in events:
            if event.status != cl_command_execution_status.CL_COMPLETE:
                error_messages.append(event.error_message)

        if error_messages:
            raise RuntimeError("\n\n".join(error_messages))

    @staticmethod
    def events_ptr(events:List[Event])->ptr_cl_event:
        if len(events) == 0:
            return None
        else:
            return (cl_event * len(events))(*[event.id for event in events])

    @staticmethod
    def create_future(events:List[Event], future_type:type)->Union[concurrent.futures.Future, asyncio.Future]:
        future = future_type()
        error_messages:List[str] = []
        copied_events:List[Event] = events.copy()

        def on_completed(event:Event, error_code:ErrorCode):
            copied_events.remove(event)
            if error_code != ErrorCode.CL_SUCCESS:
                error_messages.append(event.error_message)

            if not copied_events:
                if not error_messages:
                    future.set_result(None)
                else:
                    future.set_exception(RuntimeError("\n\n".join(error_messages)))

        if not events:
            future.set_result(None)
        else:
            for i in range(len(events)-1, -1, -1):
                event = events[i]
                event_status = event.status
                if event_status == cl_command_execution_status.CL_COMPLETE or event_status < 0:
                    error_code = ErrorCode.CL_SUCCESS
                    if event_status < 0:
                        error_code = ErrorCode(event_status)

                    on_completed(event, error_code)
                else:
                    event.on_completed_callbacks.append(on_completed)

        return future

    @property
    def context(self)->Context:
        return self._context

    @property
    def status(self)->Union[cl_command_execution_status, ErrorCode]:
        status = self.command_execution_status
        if status > 0:
            return cl_command_execution_status(status)
        else:
            return ErrorCode(status)
    
    @status.setter
    def status(self, status:cl_command_execution_status)->None:
        if not self._is_user_event:
            raise RuntimeError("cannot set status of a non-user event")
        
        CL.clSetUserEventStatus(self.id, status)

    @property
    def func(self)->CL.Func:
        return self._func
    
    @property
    def error_message(self)->str:
        error_code = self.status
        if error_code < 0:
            if self._func is not None:
                return f"{error_code}:\n" + CLInfo.func_signatures[self._func.__name__]["errors"][error_code]
            else:
                return str(error_code)
        else:
            return ""

    @CL_EVENT_NOTIFY_CALLBACK
    def _pfn_notify(event:cl_event, event_command_status:cl_int, user_data:c_void_p):
        if event_command_status> 0:
            status = cl_command_execution_status(event_command_status)
        else:
            status = ErrorCode(event_command_status)
        
        self:Event = cast(user_data, POINTER(py_object)).contents.value
        if not isinstance(self, Event):
            raise RuntimeError("invalid event")
        
        for on_status_changed in self.on_status_changed_callbacks:
            on_status_changed(self, status)

        if status == cl_command_execution_status.CL_RUNNING:
            for on_started in self.on_started_callbacks:
                on_started(self)
        elif status == cl_command_execution_status.CL_SUBMITTED:
            for on_submitted in self.on_submitted_callbacks:
                on_submitted(self)
        else:
            if event_command_status> 0:
                error_code = ErrorCode.CL_SUCCESS
            else:
                error_code = status

            for on_completed in self.on_completed_callbacks:
                on_completed(self, error_code)

        if (status == cl_command_execution_status.CL_COMPLETE or isinstance(status, ErrorCode)) and self in Event.__retained_events:
            Event.__retained_events.remove(self)

    @override
    @staticmethod
    def _prefix()->str:
        return "CL_EVENT"

    @override
    @staticmethod
    def _get_info_func()->CL.Func:
        return CL.clGetEventInfo

    @override
    @staticmethod
    def _info_types_map()->Dict[IntEnum, type]:
        return CLInfo.event_info_types

    @override
    @staticmethod
    def _info_enum()->type:
        return cl_event_info

    @override
    @staticmethod
    def _release_func():        
        return CL.clReleaseEvent