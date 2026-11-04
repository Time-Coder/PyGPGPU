"""A minimal Objective-C runtime bridge, bound with ctypes from libobjc.

The Metal backend drives the Metal.framework device API, which is an
Objective-C API: objects are reached by sending messages (`objc_msgSend`)
rather than by calling C functions. This module holds the ctypes plumbing for
that -- loading libobjc, registering selectors, and sending messages -- so the
rest of the backend reads like the CUDA/HIP/OpenCL drivers read.

The bridge loads lazily: nothing here touches libobjc until `load()` is called,
and `load()` is only called from the backend's background init thread (see
metal.py). On a machine without libobjc the load fails with the CDLL OSError,
which the init contract records instead of raising at import.
"""
from __future__ import annotations

from typing import Any
from threading import Lock
from ctypes import CDLL, Structure, c_char_p, c_double, c_size_t, c_void_p

# The path is the macOS default; a nonstandard install is reached through
# ctypes.util.find_library, tried once when the default does not exist.
OBJC_PATH:str = "/usr/lib/libobjc.dylib"


class ObjC:
    """The libobjc primitives, loaded once and reused by every message send.

    All attributes are None until `load()` succeeds.

    **A message send is serialized, and it has to be.** `objc_msgSend` is variadic in
    C, so ctypes needs the called method's own signature set on the shared function
    object before every call (`msg.argtypes`, `msg.restype`). Those are properties of
    ONE object used by every send, so two threads readying different signatures at the
    same time can hand the wrong one to `objc_msgSend` -- which is undefined behaviour
    at the ABI level, not a python exception. The launch path has no lock of its own
    (a Metal command queue is thread-safe at the framework level), so the serialization
    belongs here, where the shared state is.
    """

    lib:CDLL|None = None

    # Serializes the signature set-up and the send that follows it.
    _send_lock:Lock = Lock()

    @staticmethod
    def load()->CDLL:
        """Load libobjc once and bind the primitives the bridge needs."""
        if ObjC.lib is not None:
            return ObjC.lib
        global OBJC_PATH
        import os
        if not os.path.exists(OBJC_PATH):
            import ctypes.util
            found = ctypes.util.find_library("objc")
            if found:
                OBJC_PATH = found
        lib = CDLL(OBJC_PATH)

        lib.objc_getClass.argtypes = [c_char_p]
        lib.objc_getClass.restype = c_void_p
        lib.sel_registerName.argtypes = [c_char_p]
        lib.sel_registerName.restype = c_void_p
        # objc_msgSend is called with per-call argtypes/restype set by `send`,
        # because its real signature is the callee's -- that is what a message
        # send means. The platform ABI places it correctly for ctypes calls.
        lib.objc_msgSend.restype = c_void_p
        lib.objc_retain.argtypes = [c_void_p]
        lib.objc_retain.restype = c_void_p
        lib.objc_release.argtypes = [c_void_p]
        lib.objc_release.restype = None
        lib.object_getClassName.argtypes = [c_void_p]
        lib.object_getClassName.restype = c_char_p

        ObjC.lib = lib
        return lib

    @staticmethod
    def class_(name:str)->int:
        """Look an Objective-C class up by name (0 when absent)."""
        return ObjC.lib.objc_getClass(name.encode("ascii"))

    @staticmethod
    def sel(name:str)->int:
        """Register (or look up) a selector; SEL values are interned strings."""
        return ObjC.lib.sel_registerName(name.encode("ascii"))

    @staticmethod
    def send(obj:int, sel_name:str, argtypes:tuple = (), restype = c_void_p, *args)->Any:
        """Send one message: `[obj sel_name: ...args]`.

        `argtypes`/`restype` describe the CALLED method's signature, set per
        call -- objc_msgSend is variadic in C but every message has one real
        signature, and ctypes needs it spelled to pass the value correctly.
        """
        lib = ObjC.lib
        msg = lib.objc_msgSend
        # The signature belongs to the shared function object, so it is set and used
        # under the lock: a concurrent send with a different signature would otherwise
        # call `objc_msgSend` with the other call's argument layout.
        with ObjC._send_lock:
            msg.restype = restype
            msg.argtypes = [c_void_p, c_void_p, *argtypes]
            return msg(obj, ObjC.sel(sel_name), *args)

    @staticmethod
    def retain(obj:int)->int:
        """Take ownership of a +0 (autoreleased) reference at the python boundary."""
        return ObjC.lib.objc_retain(obj)

    @staticmethod
    def release(obj:int)->None:
        ObjC.lib.objc_release(obj)

    @staticmethod
    def class_name(obj:int)->str:
        return ObjC.lib.object_getClassName(obj).decode("ascii")

    @staticmethod
    def send_int(obj:int, sel_name:str)->int:
        """Send a message whose result is an integer (NSUInteger / NSInteger).

        The result type is `c_size_t`, not `c_ulong`: `NSUInteger` is pointer-sized,
        which `c_ulong` is on macOS and is not on Windows. `registryID` is a 64-bit
        value, so the wrong width would truncate it.
        """
        return ObjC.send(obj, sel_name, (), c_size_t)

    @staticmethod
    def send_float(obj:int, sel_name:str)->float:
        """Send a message whose result is a double."""
        return ObjC.send(obj, sel_name, (), c_double)

    @staticmethod
    def send_ptr(obj:int, sel_name:str, argtypes:tuple = (), *args)->int:
        """Send a message that returns an object pointer."""
        return ObjC.send(obj, sel_name, argtypes, c_void_p, *args)

    @staticmethod
    def describe(obj:int)->str:
        """The UTF-8 text of an NSString, or "" when the object is null.

        `-UTF8String` returns a `const char*` owned by the string; the bytes are
        decoded immediately, so the reference never outlives the message.
        """
        if not obj:
            return ""
        utf8 = ObjC.send(obj, "UTF8String", (), c_char_p)
        return utf8.decode("utf-8") if utf8 else ""

    @staticmethod
    def struct_result(obj:int, sel_name:str, fields:int)->tuple:
        """Send a message returning an N-field NSUInteger struct (MTLSize, MTLOrigin).

        objc_msgSend's result type is set to a ctypes Structure built on the fly,
        which is what makes the call read the returned bytes correctly: a small
        struct travels in registers on arm64 and through the stret convention on
        x86_64, and ctypes picks the platform's convention from the declared type.
        """
        field_names = [f"f{index}" for index in range(fields)]
        # `NSUInteger` fields are pointer-sized: `c_size_t`, never `c_ulong`, whose
        # width is 8 bytes on macOS and 4 elsewhere.
        struct_type = type("MTL_UInt%d" % fields,
                           (Structure,), {"_fields_": [(n, c_size_t) for n in field_names]})
        lib = ObjC.lib
        msg = lib.objc_msgSend
        # The signature is shared state, exactly as in `send`: set and used under the
        # same lock, or a concurrent send could call `objc_msgSend` with the other
        # call's return convention.
        with ObjC._send_lock:
            msg.restype = struct_type
            msg.argtypes = [c_void_p, c_void_p]
            result = msg(obj, ObjC.sel(sel_name))
        return tuple(getattr(result, name) for name in field_names)

