# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 4
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                type_ = type_._type_
                if hasattr(type_, 'as_dict'):
                    value = [type_.as_dict(v) for v in value]
                else:
                    value = [i for i in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead. 
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 8:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*8

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





class struct__cl_platform_id(Structure):
    pass

cl_platform_id = ctypes.POINTER(struct__cl_platform_id)
class struct__cl_device_id(Structure):
    pass

cl_device_id = ctypes.POINTER(struct__cl_device_id)
class struct__cl_context(Structure):
    pass

cl_context = ctypes.POINTER(struct__cl_context)
class struct__cl_command_queue(Structure):
    pass

cl_command_queue = ctypes.POINTER(struct__cl_command_queue)
class struct__cl_mem(Structure):
    pass

cl_mem = ctypes.POINTER(struct__cl_mem)
class struct__cl_program(Structure):
    pass

cl_program = ctypes.POINTER(struct__cl_program)
class struct__cl_kernel(Structure):
    pass

cl_kernel = ctypes.POINTER(struct__cl_kernel)
class struct__cl_event(Structure):
    pass

cl_event = ctypes.POINTER(struct__cl_event)
class struct__cl_sampler(Structure):
    pass

cl_sampler = ctypes.POINTER(struct__cl_sampler)
cl_bool = ctypes.c_uint32
cl_bitfield = ctypes.c_uint64
cl_properties = ctypes.c_uint64
cl_device_type = ctypes.c_uint64
cl_platform_info = ctypes.c_uint32
cl_device_info = ctypes.c_uint32
cl_device_fp_config = ctypes.c_uint64
cl_device_mem_cache_type = ctypes.c_uint32
cl_device_local_mem_type = ctypes.c_uint32
cl_device_exec_capabilities = ctypes.c_uint64
cl_device_svm_capabilities = ctypes.c_uint64
cl_command_queue_properties = ctypes.c_uint64
cl_device_partition_property = ctypes.c_int64
cl_device_affinity_domain = ctypes.c_uint64
cl_context_properties = ctypes.c_int64
cl_context_info = ctypes.c_uint32
cl_queue_properties = ctypes.c_uint64
cl_command_queue_info = ctypes.c_uint32
cl_channel_order = ctypes.c_uint32
cl_channel_type = ctypes.c_uint32
cl_mem_flags = ctypes.c_uint64
cl_svm_mem_flags = ctypes.c_uint64
cl_mem_object_type = ctypes.c_uint32
cl_mem_info = ctypes.c_uint32
cl_mem_migration_flags = ctypes.c_uint64
cl_image_info = ctypes.c_uint32
cl_buffer_create_type = ctypes.c_uint32
cl_addressing_mode = ctypes.c_uint32
cl_filter_mode = ctypes.c_uint32
cl_sampler_info = ctypes.c_uint32
cl_map_flags = ctypes.c_uint64
cl_pipe_properties = ctypes.c_int64
cl_pipe_info = ctypes.c_uint32
cl_program_info = ctypes.c_uint32
cl_program_build_info = ctypes.c_uint32
cl_program_binary_type = ctypes.c_uint32
cl_build_status = ctypes.c_int32
cl_kernel_info = ctypes.c_uint32
cl_kernel_arg_info = ctypes.c_uint32
cl_kernel_arg_address_qualifier = ctypes.c_uint32
cl_kernel_arg_access_qualifier = ctypes.c_uint32
cl_kernel_arg_type_qualifier = ctypes.c_uint64
cl_kernel_work_group_info = ctypes.c_uint32
cl_kernel_sub_group_info = ctypes.c_uint32
cl_event_info = ctypes.c_uint32
cl_command_type = ctypes.c_uint32
cl_profiling_info = ctypes.c_uint32
cl_sampler_properties = ctypes.c_uint64
cl_kernel_exec_info = ctypes.c_uint32
cl_device_atomic_capabilities = ctypes.c_uint64
cl_device_device_enqueue_capabilities = ctypes.c_uint64
cl_khronos_vendor_id = ctypes.c_uint32
cl_mem_properties = ctypes.c_uint64
cl_version = ctypes.c_uint32
class struct__cl_image_format(Structure):
    pass

struct__cl_image_format._pack_ = 1 # source:False
struct__cl_image_format._fields_ = [
    ('image_channel_order', ctypes.c_uint32),
    ('image_channel_data_type', ctypes.c_uint32),
]

cl_image_format = struct__cl_image_format
class struct__cl_image_desc(Structure):
    pass

class union__cl_image_desc_0(Union):
    pass

union__cl_image_desc_0._pack_ = 1 # source:False
union__cl_image_desc_0._fields_ = [
    ('buffer', ctypes.POINTER(struct__cl_mem)),
    ('mem_object', ctypes.POINTER(struct__cl_mem)),
]

struct__cl_image_desc._pack_ = 1 # source:False
struct__cl_image_desc._anonymous_ = ('_0',)
struct__cl_image_desc._fields_ = [
    ('image_type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('image_width', ctypes.c_uint64),
    ('image_height', ctypes.c_uint64),
    ('image_depth', ctypes.c_uint64),
    ('image_array_size', ctypes.c_uint64),
    ('image_row_pitch', ctypes.c_uint64),
    ('image_slice_pitch', ctypes.c_uint64),
    ('num_mip_levels', ctypes.c_uint32),
    ('num_samples', ctypes.c_uint32),
    ('_0', union__cl_image_desc_0),
]

cl_image_desc = struct__cl_image_desc
class struct__cl_buffer_region(Structure):
    pass

struct__cl_buffer_region._pack_ = 1 # source:False
struct__cl_buffer_region._fields_ = [
    ('origin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

cl_buffer_region = struct__cl_buffer_region
class struct__cl_name_version(Structure):
    pass

struct__cl_name_version._pack_ = 1 # source:False
struct__cl_name_version._fields_ = [
    ('version', ctypes.c_uint32),
    ('name', ctypes.c_char * 64),
]

cl_name_version = struct__cl_name_version
cl_int = ctypes.c_int32
cl_uint = ctypes.c_uint32
try:
    clGetPlatformIDs = _libraries['FIXME_STUB'].clGetPlatformIDs
    clGetPlatformIDs.restype = cl_int
    clGetPlatformIDs.argtypes = [cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_platform_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    clGetPlatformInfo = _libraries['FIXME_STUB'].clGetPlatformInfo
    clGetPlatformInfo.restype = cl_int
    clGetPlatformInfo.argtypes = [cl_platform_id, cl_platform_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetDeviceIDs = _libraries['FIXME_STUB'].clGetDeviceIDs
    clGetDeviceIDs.restype = cl_int
    clGetDeviceIDs.argtypes = [cl_platform_id, cl_device_type, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clGetDeviceInfo = _libraries['FIXME_STUB'].clGetDeviceInfo
    clGetDeviceInfo.restype = cl_int
    clGetDeviceInfo.argtypes = [cl_device_id, cl_device_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateSubDevices = _libraries['FIXME_STUB'].clCreateSubDevices
    clCreateSubDevices.restype = cl_int
    clCreateSubDevices.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_int64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clRetainDevice = _libraries['FIXME_STUB'].clRetainDevice
    clRetainDevice.restype = cl_int
    clRetainDevice.argtypes = [cl_device_id]
except AttributeError:
    pass
try:
    clReleaseDevice = _libraries['FIXME_STUB'].clReleaseDevice
    clReleaseDevice.restype = cl_int
    clReleaseDevice.argtypes = [cl_device_id]
except AttributeError:
    pass
try:
    clSetDefaultDeviceCommandQueue = _libraries['FIXME_STUB'].clSetDefaultDeviceCommandQueue
    clSetDefaultDeviceCommandQueue.restype = cl_int
    clSetDefaultDeviceCommandQueue.argtypes = [cl_context, cl_device_id, cl_command_queue]
except AttributeError:
    pass
try:
    clGetDeviceAndHostTimer = _libraries['FIXME_STUB'].clGetDeviceAndHostTimer
    clGetDeviceAndHostTimer.restype = cl_int
    clGetDeviceAndHostTimer.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetHostTimer = _libraries['FIXME_STUB'].clGetHostTimer
    clGetHostTimer.restype = cl_int
    clGetHostTimer.argtypes = [cl_device_id, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateContext = _libraries['FIXME_STUB'].clCreateContext
    clCreateContext.restype = cl_context
    clCreateContext.argtypes = [ctypes.POINTER(ctypes.c_int64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateContextFromType = _libraries['FIXME_STUB'].clCreateContextFromType
    clCreateContextFromType.restype = cl_context
    clCreateContextFromType.argtypes = [ctypes.POINTER(ctypes.c_int64), cl_device_type, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainContext = _libraries['FIXME_STUB'].clRetainContext
    clRetainContext.restype = cl_int
    clRetainContext.argtypes = [cl_context]
except AttributeError:
    pass
try:
    clReleaseContext = _libraries['FIXME_STUB'].clReleaseContext
    clReleaseContext.restype = cl_int
    clReleaseContext.argtypes = [cl_context]
except AttributeError:
    pass
try:
    clGetContextInfo = _libraries['FIXME_STUB'].clGetContextInfo
    clGetContextInfo.restype = cl_int
    clGetContextInfo.argtypes = [cl_context, cl_context_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clSetContextDestructorCallback = _libraries['FIXME_STUB'].clSetContextDestructorCallback
    clSetContextDestructorCallback.restype = cl_int
    clSetContextDestructorCallback.argtypes = [cl_context, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_context), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCreateCommandQueueWithProperties = _libraries['FIXME_STUB'].clCreateCommandQueueWithProperties
    clCreateCommandQueueWithProperties.restype = cl_command_queue
    clCreateCommandQueueWithProperties.argtypes = [cl_context, cl_device_id, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainCommandQueue = _libraries['FIXME_STUB'].clRetainCommandQueue
    clRetainCommandQueue.restype = cl_int
    clRetainCommandQueue.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clReleaseCommandQueue = _libraries['FIXME_STUB'].clReleaseCommandQueue
    clReleaseCommandQueue.restype = cl_int
    clReleaseCommandQueue.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clGetCommandQueueInfo = _libraries['FIXME_STUB'].clGetCommandQueueInfo
    clGetCommandQueueInfo.restype = cl_int
    clGetCommandQueueInfo.argtypes = [cl_command_queue, cl_command_queue_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateBuffer = _libraries['FIXME_STUB'].clCreateBuffer
    clCreateBuffer.restype = cl_mem
    clCreateBuffer.argtypes = [cl_context, cl_mem_flags, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateSubBuffer = _libraries['FIXME_STUB'].clCreateSubBuffer
    clCreateSubBuffer.restype = cl_mem
    clCreateSubBuffer.argtypes = [cl_mem, cl_mem_flags, cl_buffer_create_type, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImage = _libraries['FIXME_STUB'].clCreateImage
    clCreateImage.restype = cl_mem
    clCreateImage.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(struct__cl_image_desc), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreatePipe = _libraries['FIXME_STUB'].clCreatePipe
    clCreatePipe.restype = cl_mem
    clCreatePipe.argtypes = [cl_context, cl_mem_flags, cl_uint, cl_uint, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateBufferWithProperties = _libraries['FIXME_STUB'].clCreateBufferWithProperties
    clCreateBufferWithProperties.restype = cl_mem
    clCreateBufferWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), cl_mem_flags, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImageWithProperties = _libraries['FIXME_STUB'].clCreateImageWithProperties
    clCreateImageWithProperties.restype = cl_mem
    clCreateImageWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), cl_mem_flags, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(struct__cl_image_desc), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainMemObject = _libraries['FIXME_STUB'].clRetainMemObject
    clRetainMemObject.restype = cl_int
    clRetainMemObject.argtypes = [cl_mem]
except AttributeError:
    pass
try:
    clReleaseMemObject = _libraries['FIXME_STUB'].clReleaseMemObject
    clReleaseMemObject.restype = cl_int
    clReleaseMemObject.argtypes = [cl_mem]
except AttributeError:
    pass
try:
    clGetSupportedImageFormats = _libraries['FIXME_STUB'].clGetSupportedImageFormats
    clGetSupportedImageFormats.restype = cl_int
    clGetSupportedImageFormats.argtypes = [cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, ctypes.POINTER(struct__cl_image_format), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clGetMemObjectInfo = _libraries['FIXME_STUB'].clGetMemObjectInfo
    clGetMemObjectInfo.restype = cl_int
    clGetMemObjectInfo.argtypes = [cl_mem, cl_mem_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetImageInfo = _libraries['FIXME_STUB'].clGetImageInfo
    clGetImageInfo.restype = cl_int
    clGetImageInfo.argtypes = [cl_mem, cl_image_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetPipeInfo = _libraries['FIXME_STUB'].clGetPipeInfo
    clGetPipeInfo.restype = cl_int
    clGetPipeInfo.argtypes = [cl_mem, cl_pipe_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clSetMemObjectDestructorCallback = _libraries['FIXME_STUB'].clSetMemObjectDestructorCallback
    clSetMemObjectDestructorCallback.restype = cl_int
    clSetMemObjectDestructorCallback.argtypes = [cl_mem, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_mem), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSVMAlloc = _libraries['FIXME_STUB'].clSVMAlloc
    clSVMAlloc.restype = ctypes.POINTER(None)
    clSVMAlloc.argtypes = [cl_context, cl_svm_mem_flags, size_t, cl_uint]
except AttributeError:
    pass
try:
    clSVMFree = _libraries['FIXME_STUB'].clSVMFree
    clSVMFree.restype = None
    clSVMFree.argtypes = [cl_context, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCreateSamplerWithProperties = _libraries['FIXME_STUB'].clCreateSamplerWithProperties
    clCreateSamplerWithProperties.restype = cl_sampler
    clCreateSamplerWithProperties.argtypes = [cl_context, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainSampler = _libraries['FIXME_STUB'].clRetainSampler
    clRetainSampler.restype = cl_int
    clRetainSampler.argtypes = [cl_sampler]
except AttributeError:
    pass
try:
    clReleaseSampler = _libraries['FIXME_STUB'].clReleaseSampler
    clReleaseSampler.restype = cl_int
    clReleaseSampler.argtypes = [cl_sampler]
except AttributeError:
    pass
try:
    clGetSamplerInfo = _libraries['FIXME_STUB'].clGetSamplerInfo
    clGetSamplerInfo.restype = cl_int
    clGetSamplerInfo.argtypes = [cl_sampler, cl_sampler_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateProgramWithSource = _libraries['FIXME_STUB'].clCreateProgramWithSource
    clCreateProgramWithSource.restype = cl_program
    clCreateProgramWithSource.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithBinary = _libraries['FIXME_STUB'].clCreateProgramWithBinary
    clCreateProgramWithBinary.restype = cl_program
    clCreateProgramWithBinary.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithBuiltInKernels = _libraries['FIXME_STUB'].clCreateProgramWithBuiltInKernels
    clCreateProgramWithBuiltInKernels.restype = cl_program
    clCreateProgramWithBuiltInKernels.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateProgramWithIL = _libraries['FIXME_STUB'].clCreateProgramWithIL
    clCreateProgramWithIL.restype = cl_program
    clCreateProgramWithIL.argtypes = [cl_context, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainProgram = _libraries['FIXME_STUB'].clRetainProgram
    clRetainProgram.restype = cl_int
    clRetainProgram.argtypes = [cl_program]
except AttributeError:
    pass
try:
    clReleaseProgram = _libraries['FIXME_STUB'].clReleaseProgram
    clReleaseProgram.restype = cl_int
    clReleaseProgram.argtypes = [cl_program]
except AttributeError:
    pass
try:
    clBuildProgram = _libraries['FIXME_STUB'].clBuildProgram
    clBuildProgram.restype = cl_int
    clBuildProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clCompileProgram = _libraries['FIXME_STUB'].clCompileProgram
    clCompileProgram.restype = cl_int
    clCompileProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_program)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clLinkProgram = _libraries['FIXME_STUB'].clLinkProgram
    clLinkProgram.restype = cl_program
    clLinkProgram.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_device_id)), ctypes.POINTER(ctypes.c_char), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_program)), ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clSetProgramReleaseCallback = _libraries['FIXME_STUB'].clSetProgramReleaseCallback
    clSetProgramReleaseCallback.restype = cl_int
    clSetProgramReleaseCallback.argtypes = [cl_program, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_program), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetProgramSpecializationConstant = _libraries['FIXME_STUB'].clSetProgramSpecializationConstant
    clSetProgramSpecializationConstant.restype = cl_int
    clSetProgramSpecializationConstant.argtypes = [cl_program, cl_uint, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clUnloadPlatformCompiler = _libraries['FIXME_STUB'].clUnloadPlatformCompiler
    clUnloadPlatformCompiler.restype = cl_int
    clUnloadPlatformCompiler.argtypes = [cl_platform_id]
except AttributeError:
    pass
try:
    clGetProgramInfo = _libraries['FIXME_STUB'].clGetProgramInfo
    clGetProgramInfo.restype = cl_int
    clGetProgramInfo.argtypes = [cl_program, cl_program_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetProgramBuildInfo = _libraries['FIXME_STUB'].clGetProgramBuildInfo
    clGetProgramBuildInfo.restype = cl_int
    clGetProgramBuildInfo.argtypes = [cl_program, cl_device_id, cl_program_build_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateKernel = _libraries['FIXME_STUB'].clCreateKernel
    clCreateKernel.restype = cl_kernel
    clCreateKernel.argtypes = [cl_program, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateKernelsInProgram = _libraries['FIXME_STUB'].clCreateKernelsInProgram
    clCreateKernelsInProgram.restype = cl_int
    clCreateKernelsInProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_kernel)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    clCloneKernel = _libraries['FIXME_STUB'].clCloneKernel
    clCloneKernel.restype = cl_kernel
    clCloneKernel.argtypes = [cl_kernel, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainKernel = _libraries['FIXME_STUB'].clRetainKernel
    clRetainKernel.restype = cl_int
    clRetainKernel.argtypes = [cl_kernel]
except AttributeError:
    pass
try:
    clReleaseKernel = _libraries['FIXME_STUB'].clReleaseKernel
    clReleaseKernel.restype = cl_int
    clReleaseKernel.argtypes = [cl_kernel]
except AttributeError:
    pass
try:
    clSetKernelArg = _libraries['FIXME_STUB'].clSetKernelArg
    clSetKernelArg.restype = cl_int
    clSetKernelArg.argtypes = [cl_kernel, cl_uint, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetKernelArgSVMPointer = _libraries['FIXME_STUB'].clSetKernelArgSVMPointer
    clSetKernelArgSVMPointer.restype = cl_int
    clSetKernelArgSVMPointer.argtypes = [cl_kernel, cl_uint, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clSetKernelExecInfo = _libraries['FIXME_STUB'].clSetKernelExecInfo
    clSetKernelExecInfo.restype = cl_int
    clSetKernelExecInfo.argtypes = [cl_kernel, cl_kernel_exec_info, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clGetKernelInfo = _libraries['FIXME_STUB'].clGetKernelInfo
    clGetKernelInfo.restype = cl_int
    clGetKernelInfo.argtypes = [cl_kernel, cl_kernel_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelArgInfo = _libraries['FIXME_STUB'].clGetKernelArgInfo
    clGetKernelArgInfo.restype = cl_int
    clGetKernelArgInfo.argtypes = [cl_kernel, cl_uint, cl_kernel_arg_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelWorkGroupInfo = _libraries['FIXME_STUB'].clGetKernelWorkGroupInfo
    clGetKernelWorkGroupInfo.restype = cl_int
    clGetKernelWorkGroupInfo.argtypes = [cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clGetKernelSubGroupInfo = _libraries['FIXME_STUB'].clGetKernelSubGroupInfo
    clGetKernelSubGroupInfo.restype = cl_int
    clGetKernelSubGroupInfo.argtypes = [cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t, ctypes.POINTER(None), size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clWaitForEvents = _libraries['FIXME_STUB'].clWaitForEvents
    clWaitForEvents.restype = cl_int
    clWaitForEvents.argtypes = [cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clGetEventInfo = _libraries['FIXME_STUB'].clGetEventInfo
    clGetEventInfo.restype = cl_int
    clGetEventInfo.argtypes = [cl_event, cl_event_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clCreateUserEvent = _libraries['FIXME_STUB'].clCreateUserEvent
    clCreateUserEvent.restype = cl_event
    clCreateUserEvent.argtypes = [cl_context, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clRetainEvent = _libraries['FIXME_STUB'].clRetainEvent
    clRetainEvent.restype = cl_int
    clRetainEvent.argtypes = [cl_event]
except AttributeError:
    pass
try:
    clReleaseEvent = _libraries['FIXME_STUB'].clReleaseEvent
    clReleaseEvent.restype = cl_int
    clReleaseEvent.argtypes = [cl_event]
except AttributeError:
    pass
try:
    clSetUserEventStatus = _libraries['FIXME_STUB'].clSetUserEventStatus
    clSetUserEventStatus.restype = cl_int
    clSetUserEventStatus.argtypes = [cl_event, cl_int]
except AttributeError:
    pass
try:
    clSetEventCallback = _libraries['FIXME_STUB'].clSetEventCallback
    clSetEventCallback.restype = cl_int
    clSetEventCallback.argtypes = [cl_event, cl_int, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_event), ctypes.c_int32, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    clGetEventProfilingInfo = _libraries['FIXME_STUB'].clGetEventProfilingInfo
    clGetEventProfilingInfo.restype = cl_int
    clGetEventProfilingInfo.argtypes = [cl_event, cl_profiling_info, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    clFlush = _libraries['FIXME_STUB'].clFlush
    clFlush.restype = cl_int
    clFlush.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clFinish = _libraries['FIXME_STUB'].clFinish
    clFinish.restype = cl_int
    clFinish.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clEnqueueReadBuffer = _libraries['FIXME_STUB'].clEnqueueReadBuffer
    clEnqueueReadBuffer.restype = cl_int
    clEnqueueReadBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueReadBufferRect = _libraries['FIXME_STUB'].clEnqueueReadBufferRect
    clEnqueueReadBufferRect.restype = cl_int
    clEnqueueReadBufferRect.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteBuffer = _libraries['FIXME_STUB'].clEnqueueWriteBuffer
    clEnqueueWriteBuffer.restype = cl_int
    clEnqueueWriteBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteBufferRect = _libraries['FIXME_STUB'].clEnqueueWriteBufferRect
    clEnqueueWriteBufferRect.restype = cl_int
    clEnqueueWriteBufferRect.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueFillBuffer = _libraries['FIXME_STUB'].clEnqueueFillBuffer
    clEnqueueFillBuffer.restype = cl_int
    clEnqueueFillBuffer.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBuffer = _libraries['FIXME_STUB'].clEnqueueCopyBuffer
    clEnqueueCopyBuffer.restype = cl_int
    clEnqueueCopyBuffer.argtypes = [cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBufferRect = _libraries['FIXME_STUB'].clEnqueueCopyBufferRect
    clEnqueueCopyBufferRect.restype = cl_int
    clEnqueueCopyBufferRect.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueReadImage = _libraries['FIXME_STUB'].clEnqueueReadImage
    clEnqueueReadImage.restype = cl_int
    clEnqueueReadImage.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWriteImage = _libraries['FIXME_STUB'].clEnqueueWriteImage
    clEnqueueWriteImage.restype = cl_int
    clEnqueueWriteImage.argtypes = [cl_command_queue, cl_mem, cl_bool, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueFillImage = _libraries['FIXME_STUB'].clEnqueueFillImage
    clEnqueueFillImage.restype = cl_int
    clEnqueueFillImage.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyImage = _libraries['FIXME_STUB'].clEnqueueCopyImage
    clEnqueueCopyImage.restype = cl_int
    clEnqueueCopyImage.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyImageToBuffer = _libraries['FIXME_STUB'].clEnqueueCopyImageToBuffer
    clEnqueueCopyImageToBuffer.restype = cl_int
    clEnqueueCopyImageToBuffer.argtypes = [cl_command_queue, cl_mem, cl_mem, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueCopyBufferToImage = _libraries['FIXME_STUB'].clEnqueueCopyBufferToImage
    clEnqueueCopyBufferToImage.restype = cl_int
    clEnqueueCopyBufferToImage.argtypes = [cl_command_queue, cl_mem, cl_mem, size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMapBuffer = _libraries['FIXME_STUB'].clEnqueueMapBuffer
    clEnqueueMapBuffer.restype = ctypes.POINTER(None)
    clEnqueueMapBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueMapImage = _libraries['FIXME_STUB'].clEnqueueMapImage
    clEnqueueMapImage.restype = ctypes.POINTER(None)
    clEnqueueMapImage.argtypes = [cl_command_queue, cl_mem, cl_bool, cl_map_flags, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueUnmapMemObject = _libraries['FIXME_STUB'].clEnqueueUnmapMemObject
    clEnqueueUnmapMemObject.restype = cl_int
    clEnqueueUnmapMemObject.argtypes = [cl_command_queue, cl_mem, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMigrateMemObjects = _libraries['FIXME_STUB'].clEnqueueMigrateMemObjects
    clEnqueueMigrateMemObjects.restype = cl_int
    clEnqueueMigrateMemObjects.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_mem)), cl_mem_migration_flags, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueNDRangeKernel = _libraries['FIXME_STUB'].clEnqueueNDRangeKernel
    clEnqueueNDRangeKernel.restype = cl_int
    clEnqueueNDRangeKernel.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueNativeKernel = _libraries['FIXME_STUB'].clEnqueueNativeKernel
    clEnqueueNativeKernel.restype = cl_int
    clEnqueueNativeKernel.argtypes = [cl_command_queue, ctypes.CFUNCTYPE(None, ctypes.POINTER(None)), ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_mem)), ctypes.POINTER(ctypes.POINTER(None)), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueMarkerWithWaitList = _libraries['FIXME_STUB'].clEnqueueMarkerWithWaitList
    clEnqueueMarkerWithWaitList.restype = cl_int
    clEnqueueMarkerWithWaitList.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueBarrierWithWaitList = _libraries['FIXME_STUB'].clEnqueueBarrierWithWaitList
    clEnqueueBarrierWithWaitList.restype = cl_int
    clEnqueueBarrierWithWaitList.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMFree = _libraries['FIXME_STUB'].clEnqueueSVMFree
    clEnqueueSVMFree.restype = cl_int
    clEnqueueSVMFree.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(None) * 0, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct__cl_command_queue), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(None)), ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMemcpy = _libraries['FIXME_STUB'].clEnqueueSVMMemcpy
    clEnqueueSVMMemcpy.restype = cl_int
    clEnqueueSVMMemcpy.argtypes = [cl_command_queue, cl_bool, ctypes.POINTER(None), ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMemFill = _libraries['FIXME_STUB'].clEnqueueSVMMemFill
    clEnqueueSVMMemFill.restype = cl_int
    clEnqueueSVMMemFill.argtypes = [cl_command_queue, ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMap = _libraries['FIXME_STUB'].clEnqueueSVMMap
    clEnqueueSVMMap.restype = cl_int
    clEnqueueSVMMap.argtypes = [cl_command_queue, cl_bool, cl_map_flags, ctypes.POINTER(None), size_t, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMUnmap = _libraries['FIXME_STUB'].clEnqueueSVMUnmap
    clEnqueueSVMUnmap.restype = cl_int
    clEnqueueSVMUnmap.argtypes = [cl_command_queue, ctypes.POINTER(None), cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueSVMMigrateMem = _libraries['FIXME_STUB'].clEnqueueSVMMigrateMem
    clEnqueueSVMMigrateMem.restype = cl_int
    clEnqueueSVMMigrateMem.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), cl_mem_migration_flags, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clGetExtensionFunctionAddressForPlatform = _libraries['FIXME_STUB'].clGetExtensionFunctionAddressForPlatform
    clGetExtensionFunctionAddressForPlatform.restype = ctypes.POINTER(None)
    clGetExtensionFunctionAddressForPlatform.argtypes = [cl_platform_id, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    clCreateImage2D = _libraries['FIXME_STUB'].clCreateImage2D
    clCreateImage2D.restype = cl_mem
    clCreateImage2D.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), size_t, size_t, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateImage3D = _libraries['FIXME_STUB'].clCreateImage3D
    clCreateImage3D.restype = cl_mem
    clCreateImage3D.argtypes = [cl_context, cl_mem_flags, ctypes.POINTER(struct__cl_image_format), size_t, size_t, size_t, size_t, size_t, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueMarker = _libraries['FIXME_STUB'].clEnqueueMarker
    clEnqueueMarker.restype = cl_int
    clEnqueueMarker.argtypes = [cl_command_queue, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueWaitForEvents = _libraries['FIXME_STUB'].clEnqueueWaitForEvents
    clEnqueueWaitForEvents.restype = cl_int
    clEnqueueWaitForEvents.argtypes = [cl_command_queue, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
try:
    clEnqueueBarrier = _libraries['FIXME_STUB'].clEnqueueBarrier
    clEnqueueBarrier.restype = cl_int
    clEnqueueBarrier.argtypes = [cl_command_queue]
except AttributeError:
    pass
try:
    clUnloadCompiler = _libraries['FIXME_STUB'].clUnloadCompiler
    clUnloadCompiler.restype = cl_int
    clUnloadCompiler.argtypes = []
except AttributeError:
    pass
try:
    clGetExtensionFunctionAddress = _libraries['FIXME_STUB'].clGetExtensionFunctionAddress
    clGetExtensionFunctionAddress.restype = ctypes.POINTER(None)
    clGetExtensionFunctionAddress.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    clCreateCommandQueue = _libraries['FIXME_STUB'].clCreateCommandQueue
    clCreateCommandQueue.restype = cl_command_queue
    clCreateCommandQueue.argtypes = [cl_context, cl_device_id, cl_command_queue_properties, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clCreateSampler = _libraries['FIXME_STUB'].clCreateSampler
    clCreateSampler.restype = cl_sampler
    clCreateSampler.argtypes = [cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    clEnqueueTask = _libraries['FIXME_STUB'].clEnqueueTask
    clEnqueueTask.restype = cl_int
    clEnqueueTask.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(ctypes.POINTER(struct__cl_event)), ctypes.POINTER(ctypes.POINTER(struct__cl_event))]
except AttributeError:
    pass
