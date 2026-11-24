# -*- coding: utf-8 -*-
"""The error messages the CL signature tables share.

`clinfo_sig_*` describes one family of entry points each, but the specification repeats a
handful of clauses throughout them: the two resource failures close nearly every table, and a
clause about the context, the command-queue or the wait-list turns up in several families. A
message is written here once instead of being pasted into every table, so a correction reaches
every call that can report it.

Two things are deliberate. The text is the specification's own wording, quoted as-is, so a
table can still be read against Khronos' manual -- this module only removes the copies. And a
message stays beside its table when just a pair of related calls shares it (the retain/release
pairs, the buffer/image pairs): what belongs here is the clause the family split scattered
across modules, not a catalogue of every sentence.

Each constant is named for the error code whose clause it quotes, with the `CL_` prefix
dropped, so `ErrorCode.CL_INVALID_KERNEL: INVALID_KERNEL` reads as "that code, that clause".
"""

# The two failures the specification lists for almost every call, so almost every table ends
# with them.
OUT_OF_RESOURCES = (
    "there is a failure to allocate resources required by the OpenCL implementation on the "
    "device."
)

OUT_OF_HOST_MEMORY = (
    "there is a failure to allocate resources required by the OpenCL implementation on the "
    "host."
)

# The generic "this handle is not a handle" clauses, shared by the families that query or
# create the object.
INVALID_CONTEXT = "context is not a valid context."

INVALID_COMMAND_QUEUE = "command_queue is not a valid host command-queue."

INVALID_PROGRAM = "program is not a valid program object."

INVALID_KERNEL = "kernel is not a valid kernel object."

INVALID_EVENT_WAIT_LIST = (
    "event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL "
    "and num_events_in_wait_list is 0, or event objects in event_wait_list are not valid events."
)

EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = (
    "the read and write operations are blocking and the execution status of any of the events "
    "in event_wait_list is a negative integer value. This error code is missing before version "
    "1.1."
)

# The properties clauses of the memory-object and image creation calls. They are long because
# the specification enumerates every way they fail, and identical wherever they appear.
INVALID_PROPERTY = """one of following case happend:
    * a property name in properties is not a supported property name
    * the value specified for a supported property name is not valid
    * the same property name is specified more than once
    * properties does not include a supported external memory handle and CL_MEM_DEVICE_HANDLE_LIST_KHR is specified as part of properties
    * properties includes more than one external memory handle"""

INVALID_DEVICE = """one of following case happend:
    * a device identified by the property CL_MEM_DEVICE_HANDLE_LIST_KHR is not a valid device or is not associated with context
    * a device identified by property CL_MEM_DEVICE_HANDLE_LIST_KHR cannot import the requested external memory object type
    * CL_MEM_DEVICE_HANDLE_LIST_KHR is not specified as part of properties and one or more devices in context cannot import the requested external memory object type"""
