"""CL function signatures: the event family."""
from ctypes import (
    c_void_p, c_size_t,
)
from typing import Any, Dict

from .cltypes import (
    cl_uint,
    CL_EVENT_NOTIFY_CALLBACK,
    ptr_cl_int,
    cl_context,
    cl_int,
    ptr_size_t,
    ErrorCode,
    cl_event
)
from .clinfo_errors import (
    INVALID_CONTEXT,
    OUT_OF_HOST_MEMORY,
    OUT_OF_RESOURCES,
)

SIGNATURES:Dict[str, Any] = {
        "clGetEventInfo": {
            "args": {
                "event": cl_event,
                "param_name": cl_uint,
                "param_value_size": c_size_t,
                "param_value": c_void_p,
                "param_value_size_ret": ptr_size_t
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "event is not a valid event object.",
                ErrorCode.CL_INVALID_VALUE: """one of following case happend:
    * param_name is not one of the supported values, or the size in bytes specified by param_value_size is less than size of the return type specified in the Event Object Queries table and param_value is not NULL.
    * the information to query given in param_name cannot be queried for event.""",
                ErrorCode.CL_OUT_OF_RESOURCES: OUT_OF_RESOURCES,
                ErrorCode.CL_OUT_OF_HOST_MEMORY: OUT_OF_HOST_MEMORY
            }
        },
    
        "clCreateUserEvent": {
            "args": {
                "context": cl_context,
                "errcode_ret": ptr_cl_int
            },
            "restype": cl_event,
            "errors": {
                ErrorCode.CL_INVALID_CONTEXT: INVALID_CONTEXT,
                ErrorCode.CL_OUT_OF_RESOURCES: OUT_OF_RESOURCES,
                ErrorCode.CL_OUT_OF_HOST_MEMORY: OUT_OF_HOST_MEMORY
            }
        },
    
        "clSetUserEventStatus": {
            "args": {
                "event": cl_event,
                "execution_status": cl_int
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "event is not a valid user event object.",
                ErrorCode.CL_INVALID_VALUE: "the execution_status is not CL_COMPLETE or a negative integer value.",
                ErrorCode.CL_INVALID_OPERATION: "the execution_status for event has already been changed by a previous call to clSetUserEventStatus.",
                ErrorCode.CL_OUT_OF_RESOURCES: OUT_OF_RESOURCES,
                ErrorCode.CL_OUT_OF_HOST_MEMORY: OUT_OF_HOST_MEMORY
            }
        },
    
        "clSetEventCallback": {
            "args": {
                "event": cl_event,
                "command_exec_callback_type": cl_int,
                "pfn_notify": CL_EVENT_NOTIFY_CALLBACK,
                "user_data": c_void_p
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "event is not a valid event object.",
                ErrorCode.CL_INVALID_VALUE: "pfn_event_notify is NULL or command_exec_callback_type is not CL_SUBMITTED, CL_RUNNING, or CL_COMPLETE.",
                ErrorCode.CL_OUT_OF_RESOURCES: OUT_OF_RESOURCES,
                ErrorCode.CL_OUT_OF_HOST_MEMORY: OUT_OF_HOST_MEMORY
            }
        },
    
        "clReleaseEvent": {
            "args": {
                "event": cl_event
            },
            "restype": cl_int,
            "errors": {
                ErrorCode.CL_INVALID_EVENT: "if event is not a valid event object.",
                ErrorCode.CL_OUT_OF_RESOURCES: OUT_OF_RESOURCES,
                ErrorCode.CL_OUT_OF_HOST_MEMORY: OUT_OF_HOST_MEMORY
            }
        },
}
