from ctypes import c_uint, c_int
from .cutypes import CUresult


class CUInfo:

    func_signatures = {
        # CUresult cuInit(unsigned int Flags)
        "cuInit": {
            "args": {
                "Flags": c_uint,
            },
            "restype": c_int,
            "errors": {
                CUresult.CUDA_ERROR_INVALID_VALUE,
                CUresult.CUDA_ERROR_INVALID_DEVICE,
                CUresult.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
                CUresult.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
            }
        }
    }

    error_codes = {
        CUresult.CUDA_SUCCESS: "no errors.",
        CUresult.CUDA_ERROR_INVALID_VALUE: "one or more of the parameters passed to the API call is not within an acceptable range of values.",
        CUresult.CUDA_ERROR_OUT_OF_MEMORY: "The API call failed because it was unable to allocate enough memory or other resources to perform the requested operation.",
        CUresult.CUDA_ERROR_NOT_INITIALIZED: "the CUDA driver has not been initialized with cuInit() or that initialization has failed.",
        CUresult.CUDA_ERROR_DEINITIALIZED: "the CUDA driver is in the process of shutting down.",
        CUresult.CUDA_ERROR_PROFILER_DISABLED: """profiler is not initialized for this run.
This can happen when the application is running with external profiling tools like visual profiler.""",
        CUresult.CUDA_ERROR_PROFILER_NOT_INITIALIZED: "attempt to enable/disable the profiling via cuProfilerStart or cuProfilerStop without initialization.",
        CUresult.CUDA_ERROR_PROFILER_ALREADY_STARTED: "call cuProfilerStart() when profiling is already enabled.",
        CUresult.CUDA_ERROR_PROFILER_ALREADY_STOPPED: "call cuProfilerStop() when profiling is already disabled.",
        CUresult.CUDA_ERROR_STUB_LIBRARY: """the CUDA driver that the application has loaded is a stub library.
Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.""",
        CUresult.CUDA_ERROR_DEVICE_UNAVAILABLE: """requested CUDA device is unavailable at the current time.
Devices are often unavailable due to use of CU_COMPUTEMODE_EXCLUSIVE_PROCESS or CU_COMPUTEMODE_PROHIBITED.""",
        CUresult.CUDA_ERROR_NO_DEVICE: "no CUDA-capable devices were detected by the installed CUDA driver.",
        CUresult.CUDA_ERROR_INVALID_DEVICE: "the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.",
        CUresult.CUDA_ERROR_DEVICE_NOT_LICENSED: "the Grid license is not applied.",
        CUresult.CUDA_ERROR_INVALID_IMAGE: """one of following case happend:
* the device kernel image is invalid.
* an invalid CUDA module.""",
        CUresult.CUDA_ERROR_INVALID_CONTEXT: """one of following case happend:
* there is no context bound to the current thread.
* the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it).
* a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
* the green context passed to an API call was not converted to a CUcontext using cuCtxFromGreenCtx API.
""",
        CUresult.CUDA_ERROR_CONTEXT_ALREADY_CURRENT: """the context being supplied as a parameter to the API call was already the active context.
attempt to push the active context via cuCtxPushCurrent().
""",
        CUresult.CUDA_ERROR_MAP_FAILED: "a map or register operation has failed.",
        CUresult.CUDA_ERROR_UNMAP_FAILED: "an unmap or unregister operation has failed.",
        CUresult.CUDA_ERROR_ARRAY_IS_MAPPED: "the specified array is currently mapped and thus cannot be destroyed.",
        CUresult.CUDA_ERROR_ALREADY_MAPPED: "the resource is already mapped.",
        CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU: """one of following case happend:
* there is no kernel image available that is suitable for the device.
* a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
""",
        CUresult.CUDA_ERROR_ALREADY_ACQUIRED: "a resource has already been acquired.",
        CUresult.CUDA_ERROR_NOT_MAPPED: "a resource is not mapped.",
        CUresult.CUDA_ERROR_NOT_MAPPED_AS_ARRAY: "a mapped resource is not available for access as an array.",
        CUresult.CUDA_ERROR_NOT_MAPPED_AS_POINTER: "a mapped resource is not available for access as a pointer.",
        CUresult.CUDA_ERROR_ECC_UNCORRECTABLE: "an uncorrectable ECC error was detected during execution.",
        CUresult.CUDA_ERROR_UNSUPPORTED_LIMIT: "the CUlimit passed to the API call is not supported by the active device.",
        CUresult.CUDA_ERROR_CONTEXT_ALREADY_IN_USE: "the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.",
        CUresult.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: "peer access is not supported across the given devices.",
        CUresult.CUDA_ERROR_INVALID_PTX: "a PTX JIT compilation failed.",
        CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: "an error with OpenGL or DirectX context.",
        CUresult.CUDA_ERROR_NVLINK_UNCORRECTABLE: "an uncorrectable NVLink error was detected during the execution.",
        CUresult.CUDA_ERROR_JIT_COMPILER_NOT_FOUND: "the PTX JIT compiler library was not found.",
        CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION: "the provided PTX was compiled with an unsupported toolchain.",
        CUresult.CUDA_ERROR_JIT_COMPILATION_DISABLED: "the PTX JIT compilation was disabled.",
        CUresult.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: "the CUexecAffinityType passed to the API call is not supported by the active device.",
        CUresult.CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC: "the code to be compiled by the PTX JIT contains unsupported call to cudaDeviceSynchronize.",
        CUresult.CUDA_ERROR_INVALID_SOURCE: """the device kernel source is invalid.
This includes compilation/linker errors encountered in device code or user error.""",
        CUresult.CUDA_ERROR_FILE_NOT_FOUND: "the file specified was not found.",
        CUresult.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: "a link to a shared object failed to resolve.",
        CUresult.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: "initialization of a shared object failed.",
        CUresult.CUDA_ERROR_OPERATING_SYSTEM: "an OS call failed.",
        CUresult.CUDA_ERROR_INVALID_HANDLE: """a resource handle passed to the API call was not valid.
Resource handles are opaque types like CUstream and CUevent.""",
        CUresult.CUDA_ERROR_ILLEGAL_STATE: "a resource required by the API call is not in a valid state to perform the requested operation.",
        CUresult.CUDA_ERROR_LOSSY_QUERY: """one of following case happend:
* an attempt was made to introspect an object in a way that would discard semantically important information.
* the object using funtionality newer than the API version used to introspect it or omission of optional return arguments.""",
        CUresult.CUDA_ERROR_NOT_FOUND: "a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.",
        CUresult.CUDA_ERROR_NOT_READY: """asynchronous operations issued previously have not completed yet.
This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion).
Calls that may return this value include cuEventQuery() and cuStreamQuery().""",
        CUresult.CUDA_ERROR_ILLEGAL_ADDRESS: """While executing a kernel, the device encountered a load or store instruction on an invalid memory address.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: """a launch did not occur because it did not have appropriate resources.
This error usually indicates that the user has attempted to pass too many arguments to the device kernel,
or the kernel launch specifies too many threads for the kernel's register count.
Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.""",
        CUresult.CUDA_ERROR_LAUNCH_TIMEOUT: """the device kernel took too long to execute.
This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: "a kernel launch that uses an incompatible texturing mode.",
        CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: "a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled.",
        CUresult.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: "cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess().",
        CUresult.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: "the primary context for the specified device has already been initialized.",
        CUresult.CUDA_ERROR_CONTEXT_IS_DESTROYED: "the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.",
        CUresult.CUDA_ERROR_ASSERT: """A device-side assert triggered during kernel execution.
The context cannot be used anymore, and must be destroyed.
All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.""",
        CUresult.CUDA_ERROR_TOO_MANY_PEERS: "the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess().",
        CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: "the memory range passed to cuMemHostRegister() has already been registered.",
        CUresult.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: "the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region.",
        CUresult.CUDA_ERROR_HARDWARE_STACK_ERROR: """While executing a kernel, the device encountered a stack error.
This can be due to stack corruption or exceeding the stack size limit.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION: """While executing a kernel, the device encountered an illegal instruction.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_MISALIGNED_ADDRESS: """While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_INVALID_ADDRESS_SPACE: """While executing a kernel,
the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local),
but was supplied a memory address not belonging to an allowed address space.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_INVALID_PC: """While executing a kernel,
the device program counter wrapped its address space.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_LAUNCH_FAILED: """An exception occurred on the device while executing a kernel.
Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory.
Less common cases can be system specific - more information about these cases can be found in the system specific user guide.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: """the number of blocks launched per grid for a kernel
that was launched via either cuLaunchCooperativeKernel or cuLaunchCooperativeKernelMultiDevice
exceeds the maximum number of blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor
or cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
as specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.""",
        CUresult.CUDA_ERROR_NOT_PERMITTED: "the attempted operation is not permitted.",
        CUresult.CUDA_ERROR_NOT_SUPPORTED: "the attempted operation is not supported on the current system or device.",
        CUresult.CUDA_ERROR_SYSTEM_NOT_READY: """the system is not yet ready to start any CUDA work.
To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running.
More information about this error can be found in the system specific user guide.""",
        CUresult.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: """there is a mismatch between the versions of the display driver and the CUDA driver.
Refer to the compatibility documentation for supported versions.""",
        CUresult.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: """the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration.
Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.""",
        CUresult.CUDA_ERROR_MPS_CONNECTION_FAILED: "the MPS client failed to connect to the MPS control daemon or the MPS server.",
        CUresult.CUDA_ERROR_MPS_RPC_FAILURE: "the remote procedural call between the MPS server and the MPS client failed.",
        CUresult.CUDA_ERROR_MPS_SERVER_NOT_READY: """the MPS server is not ready to accept new MPS client requests.
This error can be returned when the MPS server is in the process of recovering from a fatal failure.""",
        CUresult.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: "the hardware resources required to create MPS client have been exhausted.",
        CUresult.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: "the hardware resources required to support device connections have been exhausted.",
        CUresult.CUDA_ERROR_MPS_CLIENT_TERMINATED: """the MPS client has been terminated by the server.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_CDP_NOT_SUPPORTED: "the module is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.",
        CUresult.CUDA_ERROR_CDP_VERSION_MISMATCH: "a module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: "the operation is not permitted when the stream is capturing.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: "the current capture sequence on the stream has been invalidated due to a previous error.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_MERGE: "the operation would have resulted in a merge of two independent capture sequences.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: "the capture was not initiated in this stream.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_UNJOINED: "the capture sequence contains a fork that was not joined to the primary stream.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_ISOLATION: """a dependency would have been created which crosses the capture sequence boundary.
Only implicit in-stream ordering dependencies are allowed to cross the boundary.""",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: "a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.",
        CUresult.CUDA_ERROR_CAPTURED_EVENT: "the operation is not permitted on an event which was last recorded in a capturing stream.",
        CUresult.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: "A stream capture sequence not initiated with the CU_STREAM_CAPTURE_MODE_RELAXED argument to cuStreamBeginCapture was passed to cuStreamEndCapture in a different thread.",
        CUresult.CUDA_ERROR_TIMEOUT: "the timeout specified for the wait operation has lapsed.",
        CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.",
        CUresult.CUDA_ERROR_EXTERNAL_DEVICE: """an async error has occurred in a device outside of CUDA.
If CUDA was waiting for an external device's signal before consuming shared data,
the external device signaled an error indicating that the data is not valid for consumption.
This leaves the process in an inconsistent state and any further CUDA work will return the same error.
To continue using CUDA, the process must be terminated and relaunched.""",
        CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE: "a kernel launch error due to cluster misconfiguration.",
        CUresult.CUDA_ERROR_FUNCTION_NOT_LOADED: "a function handle is not loaded when calling an API that requires a loaded function.",
        CUresult.CUDA_ERROR_INVALID_RESOURCE_TYPE: "one or more resources passed in are not valid resource types for the operation.",
        CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION: "one or more resources are insufficient or non-applicable for the operation.",
        CUresult.CUDA_ERROR_UNKNOWN: "an unknown internal error has occurred.",
    }