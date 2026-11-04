"""The shared machinery of the host-side CUDA math libraries.

Every library bound here (cuBLAS, cuSOLVER, cuFFT, cuSPARSE, the CUDA runtime) has the
same shape, and that shape is the CUDA driver's shape (``pygpgpu/cuda/driver/cuda.py``):

* a metaclass holds the DLL, loaded lazily on first use;
* a ``Func`` wrapper binds the argument types and the return type on first use and
  turns a returned error code into a ``RuntimeError`` carrying the library's message;
* ``init()`` pre-binds the entry points, and a missing DLL is recorded rather than
  raised, so a machine without the toolkit still imports the package.

The libraries differ only in their DLL name, their signature table and their error
codes, so those three live in the class attributes each library declares, and this
module holds everything they share.
"""
from __future__ import annotations

import os
import platform
from ctypes import c_int, c_void_p, CDLL, _Pointer, Array, addressof
from typing import Any, Callable, Dict, List, Optional


class LibraryError(RuntimeError):
    """A library call returned an error code.

    The message carries the library's own name, the numeric code and the library's
    description of that code, so the failure reads the way the vendor documents it
    rather than as an opaque number.
    """


# The CUDA driver's "no context is current" code. The driver API and the runtime
# API share one error space, so the code is the same in every library.
CUDA_ERROR_INVALID_CONTEXT = 201

# What to tell the user when that code comes back. The driver's own message says
# "there is no context bound to the current thread", which is true but does not say
# what to do about it -- and the answer is specific to this package: a context has
# to be made current before a library call, because a library handle belongs to the
# context that was current when it was created.
_CONTEXT_HINT = (
    "no CUDA context is current: make one current before calling the CUDA "
    "libraries (e.g. device.default_context.make_current()), because a library "
    "handle belongs to the context that was current when it was created"
)


def add_dll_folder(folder:Optional[str])->None:
    """Make the folder holding a DLL discoverable by the loader.

    Python 3.8 added ``os.add_dll_directory`` and stopped consulting ``PATH`` for
    extension dependencies; on 3.7 the attribute does not exist at all, so that call
    raised AttributeError and the library could never load there even with a valid
    toolkit installed. 3.7 keeps using ``PATH``, which is what CPython itself
    consults before 3.8.
    """
    if not folder:
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is not None:
        # Not keeping the returned handle is deliberate: closing it would remove the
        # directory again, and the DLL stays loaded for the life of the process.
        try:
            add_dir(folder)
        except OSError:
            pass
        return
    path = os.environ.get("PATH", "")
    if folder not in path.split(os.pathsep):
        os.environ["PATH"] = folder + os.pathsep + path


def resolve_dll(dll_name:str, so_name:Optional[str]=None,
               roots:Optional[List[str]]=None)->str:
    """The path of a library's shared library, or the bare name when not found.

    The CUDA toolkit installs its libraries in a versioned folder that is on PATH only
    when the toolkit was installed with that option, so the well-known install
    location is searched as well. The newest version wins -- an older toolkit on PATH
    should not shadow a newer one the user installed.

    The ``roots`` are the folders to search, each holding the versioned subfolders. A
    caller that wants a different toolkit (ROCm's, say) passes its own; the default is
    the CUDA one.
    """
    system = platform.system()
    if system != "Windows":
        # On the other platforms the libraries are the unversioned SONAMEs, and the
        # loader finds them the same way it finds the driver's own.
        return so_name if so_name else dll_name

    candidates:List[str] = [dll_name]
    search_roots:List[str] = roots if roots is not None else [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_HOME"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        "/usr/local/cuda",
    ]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            versions = sorted(os.listdir(root), reverse=True)
        except OSError:
            continue
        for version in versions:
            candidate = os.path.join(root, version, "bin", dll_name)
            if os.path.isfile(candidate):
                candidates.append(candidate)

    for candidate in candidates:
        if os.path.isfile(candidate):
            add_dll_folder(os.path.dirname(os.path.abspath(candidate)))
            return candidate

    # Nothing found on disk: hand the bare name to the loader, which searches PATH.
    return dll_name


class MetaLibrary(type):
    """Holds the DLL of one library and the function objects bound from it.

    The DLL is loaded on first use, never at import time: importing this package must
    not cost a CUDA runtime load, and a machine without the toolkit must still be able
    to import it (the failure then surfaces at the call, with the loader's own
    OSError).
    """

    def __getattr__(cls, name:str)->Callable:
        # The metaclass attribute lookup only reaches here for a name the class does
        # not have. A library's entry points are reached as `CUBLAS.cublasSgemm_v2`,
        # so the lookup is by the DLL's own symbol name.
        if name.startswith("_"):
            raise AttributeError(name)

        func_map = cls._func_map
        if name not in func_map:
            func_map[name] = cls.Func(name, cls)

        return func_map[name]


class Library(metaclass=MetaLibrary):
    """The base of every bound library.

    A library declares ``_library_name`` (the vendor's name for it), ``_dll_name``
    (the file to load), ``_signatures`` (the argument/return table) and
    ``_error_codes`` (the numeric code to message map). Everything else is shared.
    """

    _library_name:str = ""
    _dll_name:str = ""
    # The name the same library carries on the other platforms (the unversioned
    # SONAME). None means the Windows name is used everywhere.
    _so_name:Optional[str] = None
    _signatures:Dict[str, Dict[str, Any]] = {}
    _error_codes:Dict[int, str] = {}
    # The functions whose return value is not an error code.
    _no_error_check:List[str] = []

    # Per-library state. The DLL is loaded lazily; the function map holds one object
    # per symbol name, so a name is bound once and reused.
    _dll_instance:Optional[CDLL] = None
    _func_map:Dict[str, Callable] = {}
    # The exception init() hit when the machine has no runtime (usually the OSError
    # for a missing DLL). None once a call has actually succeeded. Diagnostic only:
    # every real call re-raises whatever the library itself raises, so nothing here
    # can make a broken runtime look usable.
    last_init_error:Optional[BaseException] = None

    # The switches the CUDA driver class carries, so the two read the same way.
    print_call:bool = False
    print_info:bool = True
    check_error:bool = True

    class Func:

        def __init__(self, name:str, library:type):
            self.name = name
            self._library = library

        def __call__(self, *args, **kwargs):
            library = self._library

            if library.print_call:
                call_str = self.name + "(" + ", ".join(
                    [self.__str_arg(arg) for arg in args]
                    + [f"{key}={self.__str_arg(value)}" for key, value in kwargs.items()]) + ")"
                print(call_str, end="", flush=True)

            func_info = library._signatures[self.name]
            if "dll_func" not in func_info:
                func = getattr(library.dll(), self.name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
            else:
                func = func_info["dll_func"]

            current_n_args:int = len(args)
            target_n_args:int = len(func_info["args"])
            args_names = None
            if current_n_args < target_n_args:
                args = list(args)
                args_names = list(func_info["args"].keys())
                for i in range(current_n_args, target_n_args):
                    args.append(kwargs[args_names[i]])

            return_value = func(*args)
            error_code:int = 0
            if func_info["restype"] == c_int:
                try:
                    error_code = int(return_value)
                except Exception:
                    error_code = 0
            elif func_info["restype"] == c_void_p:
                return_value = c_void_p(return_value)

            if library.print_call:
                print(f"->{return_value}", flush=True)

            if library.check_error and self.name not in library._no_error_check:
                if error_code != 0:
                    message = library._error_codes.get(error_code)
                    if message is None:
                        message = "unknown error."

                    # The driver's own "no context" message does not say what to do
                    # about it, and this is the one code whose answer is specific to
                    # this package, so the hint is added here rather than in each
                    # library's table.
                    message = library._context_hint_for(error_code, message)

                    raise LibraryError(
                        f"{library._library_name}: {error_code}: {message} ({self.name})")

            return return_value

        @staticmethod
        def __get_ctypes_type_name(tp)->str:
            if isinstance(tp, type):
                name = tp.__name__
            else:
                name = type(tp).__name__

            name_map = {
                'c_char': 'char',
                'c_wchar': 'wchar',
                'c_byte': 'char',
                'c_ubyte': 'uchar',
                'c_short': 'short',
                'c_ushort': 'ushort',
                'c_int': 'int',
                'c_uint': 'uint',
                'c_long': 'long',
                'c_ulong': 'ulong',
                'c_longlong': 'int64_t',
                'c_ulonglong': 'uint64_t',
                'c_float': 'float',
                'c_double': 'double',
                'c_bool': 'bool',
                'c_void_p': 'void*',
                'c_char_p': 'char*',
                'LP_c_ubyte': 'uchar*'
            }
            if name in name_map:
                return name_map[name]
            else:
                return "void*"

        @staticmethod
        def __str_arg(arg)->str:
            if type(arg).__name__ == 'CArgObject':
                obj = arg._obj
                type_name = Library.Func.__get_ctypes_type_name(obj)
                addr = addressof(obj)
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, _Pointer):
                pointed_type = arg._type_
                type_name = Library.Func.__get_ctypes_type_name(pointed_type)
                addr = addressof(arg.contents) if arg else 0
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, Array):
                elem_type = arg._type_
                elem_type_name = Library.Func.__get_ctypes_type_name(elem_type)
                length = len(arg)
                addr = addressof(arg)
                return f"({elem_type_name}[{length}])(0x{addr:x})"

            if hasattr(arg, 'value'):
                return str(arg.value)

            if arg is None:
                return "NULL"

            return str(arg)

    @classmethod
    def dll(cls)->CDLL:
        """The library's DLL, loaded on first use."""
        if cls._dll_instance is None:
            cls._dll_instance = CDLL(resolve_dll(cls._dll_name, cls._so_name))
        return cls._dll_instance

    @classmethod
    def _context_hint_for(cls, error_code:int, message:str)->str:
        """The message for an error code, with the context hint where it applies.

        The driver's own message for "no context" says "there is no context bound
        to the current thread", which is true but does not say what to do about it.
        The answer is specific to this package -- a context has to be made current
        before a library call, because a library handle belongs to the context that
        was current when it was created -- so the hint is added here, once, for
        every library that shares the driver's error space.
        """
        if error_code == CUDA_ERROR_INVALID_CONTEXT:
            return f"{message}; {_CONTEXT_HINT}"

        return message

    @classmethod
    def available(cls)->bool:
        """Whether the library's DLL can be loaded on this machine.

        A convenience for a caller that wants to ask before it uses the library; the
        answer is a fact about the machine, not a promise that a call will succeed.
        """
        try:
            cls.dll()
            return True
        except OSError:
            return False

    @classmethod
    def init(cls):
        """Pre-bind the library's entry points.

        A runtime the machine does not have must not fail the import: the entries are
        bound lazily anyway (``Func.__call__`` loads them on first use), so a missing
        DLL here only means "nothing to pre-bind". The error is recorded for the
        diagnostic and the next real call fails with the same OSError it always
        raised.
        """
        try:
            for func_name, func_info in cls._signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(cls.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
        except BaseException as error:
            cls.last_init_error = error
            if cls.print_info:
                print(f"warning: could not pre-bind {cls._library_name} "
                      f"({type(error).__name__}: {error}); "
                      f"calls will fail when one is made.")

        # preheat -- through the raw dll, NOT through Func.__call__: the binding
        # itself may run on a background thread, and Func waits for the binding, so
        # going through Func here would wait for itself.
        try:
            preheat = cls._preheat()
            if preheat is not None:
                preheat()
        except BaseException as error:
            cls.last_init_error = error

    @classmethod
    def _preheat(cls)->Optional[Callable[[], Any]]:
        """The call that forces the DLL to finish loading, if the library needs one.

        Most of these libraries have no runtime to initialize -- the CUDA driver owns
        that -- so there is nothing to preheat and the load itself is the preheat.
        """
        return None
