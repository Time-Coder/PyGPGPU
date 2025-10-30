from ctypes import c_uint, c_int, c_void_p, POINTER

class CLInfo:

    func_signatures = {
        # cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
        "clGetPlatformIDs": {
            "args": {
                "num_entries": c_uint,
                "platforms": POINTER(c_void_p),
                "num_platforms": POINTER(c_uint)
            },
            "restype": c_int,
            "errors": {
                "CL_PLATFORM_NOT_FOUND_KHR": "cl_khr_icd extension is supported and zero platforms are available.",
                "CL_INVALID_VALUE": "num_entries is equal to zero and platforms is not NULL or if both num_platforms and platforms are NULL.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host."
            },
            "dll_func": None
        }
    }