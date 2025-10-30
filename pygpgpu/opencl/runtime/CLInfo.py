from ctypes import c_uint, c_int, c_void_p, c_ulong, POINTER


class CLInfo:

    func_signatures = {
        # cl_int clGetPlatformIDs(
        #   cl_uint num_entries,
        #   cl_platform_id* platforms,
        #   cl_uint* num_platforms
        # );
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
        },

        # cl_int clGetDeviceIDs(
        #   cl_platform_id platform,
        #   cl_device_type device_type,
        #   cl_uint num_entries,
        #   cl_device_id* devices,
        #   cl_uint* num_devices
        # );
        "clGetDeviceIDs": {
            "args": {
                "platform": c_void_p,
                "device_type": c_ulong,
                "num_entries": c_uint,
                "devices": POINTER(c_void_p),
                "num_devices": POINTER(c_uint)
            },
            "restype": c_int,
            "errors": {
                "CL_INVALID_PLATFORM": "platform is not a valid platform.",
                "CL_INVALID_DEVICE_TYPE": "device_type is not a valid value.",
                "CL_INVALID_VALUE": "num_entries is equal to zero and devices is not NULL or both num_devices and devices are NULL.",
                "CL_DEVICE_NOT_FOUND": "no OpenCL devices that matched device_type were found.",
                "CL_OUT_OF_RESOURCES": "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host."
            },
            "dll_func": None
        }, 

        # cl_context clCreateContext(
        #   const cl_context_properties* properties,
        #   cl_uint num_devices,
        #   const cl_device_id* devices,
        #   void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
        #   void* user_data,
        #   cl_int* errcode_ret
        # );
        "clCreateContext": {
            "args": {
                "properties": POINTER(POINTER(c_int)),
                "num_devices": c_uint,
                "devices": POINTER(c_void_p),
                "pfn_notify": c_void_p,
                "user_data": c_void_p,
                "errcode_ret": POINTER(c_int)
            },
            "restype": c_void_p,
            "errors": {
                "CL_INVALID_PLATFORM": "no platform is specified in properties and no platform could be selected, or if the platform specified in properties is not a valid platform.",
                "CL_INVALID_PROPERTY": "a context property name in properties is not a supported property name, if the value specified for a supported property name is not valid, or if the same property name is specified more than once. This error code is missing before version 1.1.",
                "CL_INVALID_VALUE": """one of following case happends:
* devices is NULL.
* num_devices is equal to zero.
* pfn_notify is NULL but user_data is not NULL.""",
                "CL_INVALID_DEVICE": "any device in devices is not a valid device.",
                "CL_DEVICE_NOT_AVAILABLE": "a device in devices is currently not available even though the device was returned by clGetDeviceIDs.",
                "CL_OUT_OF_RESOURCES": "there is a failure to allocate resources required by the OpenCL implementation on the device.",
                "CL_OUT_OF_HOST_MEMORY": "there is a failure to allocate resources required by the OpenCL implementation on the host.",
                "CL_INVALID_DX9_MEDIA_ADAPTER_KHR": "any of the values of the properties CL_CONTEXT_ADAPTER_D3D9_KHR, CL_CONTEXT_ADAPTER_D3D9EX_KHR or CL_CONTEXT_ADAPTER_DXVA_KHR is non-NULL and does not specify a valid media adapter with which the cl_device_ids against which this context is to be created may interoperate.",
                "CL_INVALID_D3D10_DEVICE_KHR": "the value of the property CL_CONTEXT_D3D10_DEVICE_KHR is non-NULL and does not specify a valid Direct3D 10 device with which the cl_device_ids against which this context is to be created may interoperate.",
                "CL_INVALID_OPERATION": "Direct3D 10 interoperability is specified by setting CL_INVALID_D3D10_DEVICE_KHR to a non-NULL value, and interoperability with another graphics API is also specified.",
                "CL_INVALID_D3D11_DEVICE_KHR": "the value of the property CL_CONTEXT_D3D11_DEVICE_KHR is non-NULL and does not specify a valid Direct3D 11 device with which the cl_device_ids against which this context is to be created may interoperate.",
                "CL_INVALID_OPERATION": "Direct3D 11 interoperability is specified by setting CL_INVALID_D3D11_DEVICE_KHR to a non-NULL value, and interoperability with another graphics API is also specified.",
                "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR": """a context was specified for an OpenGL or OpenGL ES implementation using the EGL, GLX, or WGL binding APIs, as described above; and any of the following conditions hold:
* The specified display and context properties do not identify a valid OpenGL or OpenGL ES context.
* The specified context does not support buffer and renderbuffer objects.
* The specified context is not compatible with the OpenCL context being created (for example, it exists in a physically distinct address space, such as another hardware device; or it does not support sharing data with OpenCL due to implementation restrictions).""",
                "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR": "a share group was specified for a CGL-based OpenGL implementation by setting the property CL_CGL_SHAREGROUP_KHR, and the specified share group does not identify a valid CGL share group object.",
                "CL_INVALID_OPERATION": """a context was specified as described above and any of the following conditions hold:
* A context or share group object was specified for one of CGL, EGL, GLX, or WGL and the OpenGL implementation does not support that window-system binding API.
* More than one of the properties CL_CGL_SHAREGROUP_KHR, CL_EGL_DISPLAY_KHR, CL_GLX_DISPLAY_KHR, and CL_WGL_HDC_KHR is set to a non-default value.
* Both of the properties CL_CGL_SHAREGROUP_KHR and CL_GL_CONTEXT_KHR are set to non-default values.
* Any of the devices specified in the devices argument cannot support OpenCL objects which share the data store of an OpenGL object.""",
                "CL_INVALID_PROPERTY": "both CL_CONTEXT_INTEROP_USER_SYNC, and any of the properties defined by the cl_khr_gl_sharing extension are defined in properties.",
                "CL_INVALID_PROPERTY": "the cl_khr_terminate_context extension is supported and CL_CONTEXT_TERMINATE_KHR is set to CL_TRUE in properties, but not all of the devices associated with the context support the ability to support context termination (i.e. CL_DEVICE_TERMINATE_CAPABILITY_CONTEXT_KHR is set for CL_DEVICE_TERMINATE_CAPABILITY_KHR)."
            }
        }
    }