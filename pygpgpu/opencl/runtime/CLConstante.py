from __future__ import annotations
from typing import Dict, Union


class CLConstante:

    __instances:Dict[Union[str, int], CLConstante] = {}

    def __init__(self, name:str, value:int):
        self.__name:str = name
        self.__value:int = value
        CLConstante.__instances[name] = self
        CLConstante.__instances[value] = self

    @property
    def name(self)->str:
        return self.__name
    
    @property
    def value(self)->int:
        return self.__value
    
    @staticmethod
    def get(key:int)->CLConstante:
        return CLConstante.__instances[key]
    
    def __int__(self):
        return self.__value
    
    def __str__(self):
        return f"{self.__name}({self.__value})"
    
    def __eq__(self, other:int)->bool:
        return (self.__value == other)
    

def define_constantes(cls):
    # Error Codes
    cls.CL_SUCCESS = CLConstante("CL_SUCCESS", 0)
    cls.CL_DEVICE_NOT_FOUND = CLConstante("CL_DEVICE_NOT_FOUND", -1)
    cls.CL_DEVICE_NOT_AVAILABLE = CLConstante("CL_DEVICE_NOT_AVAILABLE", -2)
    cls.CL_COMPILER_NOT_AVAILABLE = CLConstante("CL_COMPILER_NOT_AVAILABLE", -3)
    cls.CL_MEM_OBJECT_ALLOCATION_FAILURE = CLConstante("CL_MEM_OBJECT_ALLOCATION_FAILURE", -4)
    cls.CL_OUT_OF_RESOURCES = CLConstante("CL_OUT_OF_RESOURCES", -5)
    cls.CL_OUT_OF_HOST_MEMORY = CLConstante("CL_OUT_OF_HOST_MEMORY", -6)
    cls.CL_PROFILING_INFO_NOT_AVAILABLE = CLConstante("CL_PROFILING_INFO_NOT_AVAILABLE", -7)
    cls.CL_MEM_COPY_OVERLAP = CLConstante("CL_MEM_COPY_OVERLAP", -8)
    cls.CL_IMAGE_FORMAT_MISMATCH = CLConstante("CL_IMAGE_FORMAT_MISMATCH", -9)
    cls.CL_IMAGE_FORMAT_NOT_SUPPORTED = CLConstante("CL_IMAGE_FORMAT_NOT_SUPPORTED", -10)
    cls.CL_BUILD_PROGRAM_FAILURE = CLConstante("CL_BUILD_PROGRAM_FAILURE", -11)
    cls.CL_MAP_FAILURE = CLConstante("CL_MAP_FAILURE", -12)
    cls.CL_MISALIGNED_SUB_BUFFER_OFFSET = CLConstante("CL_MISALIGNED_SUB_BUFFER_OFFSET", -13)
    cls.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = CLConstante("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", -14)
    cls.CL_COMPILE_PROGRAM_FAILURE = CLConstante("CL_COMPILE_PROGRAM_FAILURE", -15)
    cls.CL_LINKER_NOT_AVAILABLE = CLConstante("CL_LINKER_NOT_AVAILABLE", -16)
    cls.CL_LINK_PROGRAM_FAILURE = CLConstante("CL_LINK_PROGRAM_FAILURE", -17)
    cls.CL_DEVICE_PARTITION_FAILED = CLConstante("CL_DEVICE_PARTITION_FAILED", -18)
    cls.CL_KERNEL_ARG_INFO_NOT_AVAILABLE = CLConstante("CL_KERNEL_ARG_INFO_NOT_AVAILABLE", -19)
    cls.CL_INVALID_VALUE = CLConstante("CL_INVALID_VALUE", -30)
    cls.CL_INVALID_DEVICE_TYPE = CLConstante("CL_INVALID_DEVICE_TYPE", -31)
    cls.CL_INVALID_PLATFORM = CLConstante("CL_INVALID_PLATFORM", -32)
    cls.CL_INVALID_DEVICE = CLConstante("CL_INVALID_DEVICE", -33)
    cls.CL_INVALID_CONTEXT = CLConstante("CL_INVALID_CONTEXT", -34)
    cls.CL_INVALID_QUEUE_PROPERTIES = CLConstante("CL_INVALID_QUEUE_PROPERTIES", -35)
    cls.CL_INVALID_COMMAND_QUEUE = CLConstante("CL_INVALID_COMMAND_QUEUE", -36)
    cls.CL_INVALID_HOST_PTR = CLConstante("CL_INVALID_HOST_PTR", -37)
    cls.CL_INVALID_MEM_OBJECT = CLConstante("CL_INVALID_MEM_OBJECT", -38)
    cls.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = CLConstante("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", -39)
    cls.CL_INVALID_IMAGE_SIZE = CLConstante("CL_INVALID_IMAGE_SIZE", -40)
    cls.CL_INVALID_SAMPLER = CLConstante("CL_INVALID_SAMPLER", -41)
    cls.CL_INVALID_BINARY = CLConstante("CL_INVALID_BINARY", -42)
    cls.CL_INVALID_BUILD_OPTIONS = CLConstante("CL_INVALID_BUILD_OPTIONS", -43)
    cls.CL_INVALID_PROGRAM = CLConstante("CL_INVALID_PROGRAM", -44)
    cls.CL_INVALID_PROGRAM_EXECUTABLE = CLConstante("CL_INVALID_PROGRAM_EXECUTABLE", -45)
    cls.CL_INVALID_KERNEL_NAME = CLConstante("CL_INVALID_KERNEL_NAME", -46)
    cls.CL_INVALID_KERNEL_DEFINITION = CLConstante("CL_INVALID_KERNEL_DEFINITION", -47)
    cls.CL_INVALID_KERNEL = CLConstante("CL_INVALID_KERNEL", -48)
    cls.CL_INVALID_ARG_INDEX = CLConstante("CL_INVALID_ARG_INDEX", -49)
    cls.CL_INVALID_ARG_VALUE = CLConstante("CL_INVALID_ARG_VALUE", -50)
    cls.CL_INVALID_ARG_SIZE = CLConstante("CL_INVALID_ARG_SIZE", -51)
    cls.CL_INVALID_KERNEL_ARGS = CLConstante("CL_INVALID_KERNEL_ARGS", -52)
    cls.CL_INVALID_WORK_DIMENSION = CLConstante("CL_INVALID_WORK_DIMENSION", -53)
    cls.CL_INVALID_WORK_GROUP_SIZE = CLConstante("CL_INVALID_WORK_GROUP_SIZE", -54)
    cls.CL_INVALID_WORK_ITEM_SIZE = CLConstante("CL_INVALID_WORK_ITEM_SIZE", -55)
    cls.CL_INVALID_GLOBAL_OFFSET = CLConstante("CL_INVALID_GLOBAL_OFFSET", -56)
    cls.CL_INVALID_EVENT_WAIT_LIST = CLConstante("CL_INVALID_EVENT_WAIT_LIST", -57)
    cls.CL_INVALID_EVENT = CLConstante("CL_INVALID_EVENT", -58)
    cls.CL_INVALID_OPERATION = CLConstante("CL_INVALID_OPERATION", -59)
    cls.CL_INVALID_GL_OBJECT = CLConstante("CL_INVALID_GL_OBJECT", -60)
    cls.CL_INVALID_BUFFER_SIZE = CLConstante("CL_INVALID_BUFFER_SIZE", -61)
    cls.CL_INVALID_MIP_LEVEL = CLConstante("CL_INVALID_MIP_LEVEL", -62)
    cls.CL_INVALID_GLOBAL_WORK_SIZE = CLConstante("CL_INVALID_GLOBAL_WORK_SIZE", -63)
    cls.CL_INVALID_PROPERTY = CLConstante("CL_INVALID_PROPERTY", -64)
    cls.CL_INVALID_IMAGE_DESCRIPTOR = CLConstante("CL_INVALID_IMAGE_DESCRIPTOR", -65)
    cls.CL_INVALID_COMPILER_OPTIONS = CLConstante("CL_INVALID_COMPILER_OPTIONS", -66)
    cls.CL_INVALID_LINKER_OPTIONS = CLConstante("CL_INVALID_LINKER_OPTIONS", -67)
    cls.CL_INVALID_DEVICE_PARTITION_COUNT = CLConstante("CL_INVALID_DEVICE_PARTITION_COUNT", -68)
    cls.CL_INVALID_PIPE_SIZE = CLConstante("CL_INVALID_PIPE_SIZE", -69)
    cls.CL_INVALID_DEVICE_QUEUE = CLConstante("CL_INVALID_DEVICE_QUEUE", -70)
    cls.CL_INVALID_SPEC_ID = CLConstante("CL_INVALID_SPEC_ID", -71)
    cls.CL_MAX_SIZE_RESTRICTION_EXCEEDED = CLConstante("CL_MAX_SIZE_RESTRICTION_EXCEEDED", -72)

    return cls