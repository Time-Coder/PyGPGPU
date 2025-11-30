from ctypes import c_char_p, POINTER
from typing import Any

from nvrtctypes import ptr_nvrtcProgram, nvrtcResult, ptr_ptr_char, nvrtcProgram, ptr_size_t, ptr_int


class NVRTC:

    check_error: bool
    print_call: bool
    print_info: bool

    class Func:

        def __call__(self, *args, **kwargs)->Any: ...

    @staticmethod
    def init(): ...

    @staticmethod
    def nvrtcCreateProgram(prog: ptr_nvrtcProgram, src: c_char_p, name: c_char_p, numHeaders: int, headers: ptr_ptr_char, includeNames: ptr_ptr_char)->nvrtcResult: ...

    @staticmethod
    def nvrtcCompileProgram(prog: nvrtcProgram, numOptions: int, options: ptr_ptr_char)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: ptr_size_t)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetProgramLog(prog: nvrtcProgram, log: c_char_p)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: ptr_size_t)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetPTX(prog: nvrtcProgram, ptx: c_char_p)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: ptr_size_t)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetCUBIN(prog: nvrtcProgram, cubin: c_char_p)->nvrtcResult: ...

    @staticmethod
    def nvrtcDestroyProgram(prog: ptr_nvrtcProgram)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetNumSupportedArchs(numArchs: ptr_int)->nvrtcResult: ...

    @staticmethod
    def nvrtcGetSupportedArchs(supportedArchs: ptr_int)->nvrtcResult: ...

    @staticmethod
    def nvrtcVersion(major: ptr_int, minor: ptr_int)->nvrtcResult: ...
