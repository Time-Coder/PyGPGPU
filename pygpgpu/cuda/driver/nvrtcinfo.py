from ctypes import c_char_p, POINTER, c_int, c_size_t

from .nvrtctypes import nvrtcProgram


class NVRTCInfo:

    func_signatures = {
        # nvrtcResult nvrtcCreateProgram(
        #   nvrtcProgram* prog, const char* src, const char* name,
        #   int numHeaders, const char** headers, const char** includeNames
        # )
        "nvrtcCreateProgram": {
            "args": {
                "prog": POINTER(nvrtcProgram),
                "src": c_char_p,
                "name": c_char_p,
                "numHeaders": c_int,
                "headers": POINTER(c_char_p),
                "includeNames": POINTER(c_char_p)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcCompileProgram(
        #   nvrtcProgram prog, int numOptions, const char** options
        # )
        "nvrtcCompileProgram": {
            "args": {
                "prog": nvrtcProgram,
                "numOptions": c_int,
                "options": POINTER(c_char_p)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetProgramLogSize(
        #   nvrtcProgram prog, size_t *logSizeRet
        # )
        "nvrtcGetProgramLogSize": {
            "args": {
                "prog": nvrtcProgram,
                "logSizeRet": POINTER(c_size_t)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log)
        "nvrtcGetProgramLog": {
            "args": {
                "prog": nvrtcProgram,
                "log": c_char_p
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet)
        "nvrtcGetPTXSize": {
            "args": {
                "prog": nvrtcProgram,
                "ptxSizeRet": POINTER(c_size_t)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx)
        "nvrtcGetPTX": {
            "args": {
                "prog": nvrtcProgram,
                "ptx": c_char_p
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet)
        "nvrtcGetCUBINSize": {
            "args": {
                "prog": nvrtcProgram,
                "cubinSizeRet": POINTER(c_size_t)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin)
        "nvrtcGetCUBIN": {
            "args": {
                "prog": nvrtcProgram,
                "cubin": c_char_p
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog)
        "nvrtcDestroyProgram": {
            "args": {
                "prog": POINTER(nvrtcProgram)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs)
        "nvrtcGetNumSupportedArchs": {
            "args": {
                "numArchs": POINTER(c_int)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs)
        "nvrtcGetSupportedArchs": {
            "args": {
                "supportedArchs": POINTER(c_int)
            },
            "restype": c_int
        },

        # nvrtcResult nvrtcVersion(int* major, int* minor)
        "nvrtcVersion": {
            "args": {
                "major": POINTER(c_int),
                "minor": POINTER(c_int)
            },
            "restype": c_int
        }
    }