# The ctypes structures: versions, name versions, image formats and descriptors.
from ctypes import c_char, c_size_t, POINTER, Structure
from ....constants import IntConstant
from typing_extensions import TypeAlias
from .core import cl_mem, cl_uint
from .device import cl_command_queue_properties



CL_VERSION_MAJOR_BITS = IntConstant("CL_VERSION_MAJOR_BITS", 10)
CL_VERSION_MINOR_BITS = IntConstant("CL_VERSION_MINOR_BITS", 10)
CL_VERSION_PATCH_BITS = IntConstant("CL_VERSION_PATCH_BITS", 12)

CL_VERSION_MAJOR_MASK = IntConstant("CL_VERSION_MAJOR_MASK", (1 << CL_VERSION_MAJOR_BITS) - 1)
CL_VERSION_MINOR_MASK = IntConstant("CL_VERSION_MINOR_MASK", (1 << CL_VERSION_MINOR_BITS) - 1)
CL_VERSION_PATCH_MASK = IntConstant("CL_VERSION_PATCH_MASK", (1 << CL_VERSION_PATCH_BITS) - 1)

class cl_version(cl_uint):

    @property
    def major(self)->int:
        return (self.value >> (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS))

    @property
    def minor(self)->int:
        return ((self.value >> CL_VERSION_PATCH_BITS) & CL_VERSION_MINOR_MASK)

    @property
    def patch(self)->int:
        return (self.value & CL_VERSION_PATCH_MASK)
    
    def __str__(self)->str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __repr__(self)->str:
        return f"{self.major}.{self.minor}.{self.patch}"
        
cl_version_khr = cl_version


CL_NAME_VERSION_MAX_NAME_SIZE = IntConstant("CL_NAME_VERSION_MAX_NAME_SIZE", 64)
CL_NAME_VERSION_MAX_NAME_SIZE_KHR = IntConstant("CL_NAME_VERSION_MAX_NAME_SIZE_KHR", 64)

class cl_name_version(Structure):
    _fields_ = [
        ("version", cl_version),
        ("name", c_char * CL_NAME_VERSION_MAX_NAME_SIZE)
    ]

    def __repr__(self)->str:
        return str((self.name.decode("utf-8"), self.version))


class cl_name_version_khr(Structure):
    _fields_ = [
        ("version", cl_version_khr),
        ("name", c_char * CL_NAME_VERSION_MAX_NAME_SIZE_KHR)
    ]

    def __repr__(self)->str:
        return str((self.name.decode("utf-8"), self.version))
    
class cl_device_integer_dot_product_acceleration_properties_khr(Structure):
    _fields_ = [
        ("signed_accelerated", cl_uint),
        ("unsigned_accelerated", cl_uint),
        ("mixed_signedness_accelerated", cl_uint),
        ("accumulating_saturating_signed_accelerated", cl_uint),
        ("accumulating_saturating_unsigned_accelerated", cl_uint),
        ("accumulating_saturating_mixed_signedness_accelerated", cl_uint)
    ]
    
class cl_device_pci_bus_info_khr(Structure):
    _fields_ = [
        ("pci_domain", cl_uint),
        ("pci_bus", cl_uint),
        ("pci_device", cl_uint),
        ("pci_function", cl_uint)
    ]

class cl_image_format(Structure):
    _fields_ = [
        ("image_channel_order", cl_uint),
        ("image_channel_data_type", cl_uint)
    ]

ptr_cl_image_format:TypeAlias = POINTER(cl_image_format)

class cl_image_desc(Structure):
    _fields_ = [
        ("image_type", cl_uint),
        ("image_width", c_size_t),
        ("image_height", c_size_t),
        ("image_depth", c_size_t),
        ("image_array_size", c_size_t),
        ("image_row_pitch", c_size_t),
        ("image_slice_pitch", c_size_t),
        ("num_mip_levels", cl_uint),
        ("num_samples", cl_uint),
        ("buffer", cl_mem)
    ]

    @property
    def mem_object(self)->cl_mem:
        return self.buffer
    
    @mem_object.setter
    def mem_object(self, value:cl_mem)->None:
        self.buffer = value

ptr_cl_image_desc:TypeAlias = POINTER(cl_image_desc)


class queue_t:

    def __init__(self, properties:cl_command_queue_properties):
        self.properties = (cl_command_queue_properties.CL_QUEUE_ON_DEVICE | properties)
