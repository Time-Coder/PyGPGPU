from ctypes import c_void_p, POINTER, c_char_p, c_size_t, c_int
from typing import TypeAlias

from ...constants import IntEnum


nvrtcProgram = c_void_p

ptr_nvrtcProgram:TypeAlias = POINTER(nvrtcProgram)
ptr_ptr_char:TypeAlias = POINTER(c_char_p)
ptr_size_t:TypeAlias = POINTER(c_size_t)
ptr_int:TypeAlias = POINTER(c_int)


class nvrtcResult(IntEnum):
    NVRTC_SUCCESS = 0
    NVRTC_ERROR_OUT_OF_MEMORY = 1
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
    NVRTC_ERROR_INVALID_INPUT = 3
    NVRTC_ERROR_INVALID_PROGRAM = 4
    NVRTC_ERROR_INVALID_OPTION = 5
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
    NVRTC_ERROR_INTERNAL_ERROR = 11
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12

class GPUArch(IntEnum):
    compute_35 = 35
    compute_37 = 37
    compute_50 = 50
    compute_52 = 52
    compute_53 = 53
    compute_60 = 60
    compute_61 = 61
    compute_62 = 62
    compute_70 = 70
    compute_72 = 72
    compute_75 = 75
    compute_80 = 80
    sm_35 = 1035
    sm_37 = 1037
    sm_50 = 1050
    sm_52 = 1052
    sm_53 = 1053
    sm_60 = 1060
    sm_61 = 1061
    sm_62 = 1062
    sm_70 = 1070
    sm_72 = 1072
    sm_75 = 1075
    sm_80 = 1080

class CppStd(IntEnum):
    cpp03 = 3
    cpp11 = 11
    cpp14 = 14
    cpp17 = 17

class OptInfoKind(IntEnum):
    inline = 0