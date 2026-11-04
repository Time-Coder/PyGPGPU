"""The `MTLCompileOptions` a generated library is compiled with.

`newLibraryWithSource:options:error:` takes an options object, and passing nil (which
is what this backend did) means "the defaults" with no way to say otherwise. The
properties a build can configure are few, and each maps to one Objective-C selector:

* **`languageVersion`** -- the MSL dialect. The generated sources target MSL 2.x, which
  is what every Mac with a Metal device supports; a device needing a newer dialect is
  a later concern, not something to guess at here.
* **`fastMathEnabled`** -- MSL's default is YES, and it matters: fast math lets the
  compiler use `fast::` approximations and reassociate floating point. A kernel that
  needs IEEE-exact results must turn it off, and there was no way to ask for that.
* **`mathMode`** -- the newer spelling of the same choice (`MTLMathModeFast` /
  `/Relaxed` / `/Precise`). Metal prefers it over `fastMathEnabled` where both exist.
* **`preprocessorMacros`** -- the `-D` of the SDK. The generated source names nothing
  that needs a macro today, but a handwritten `.metal` source may.

**The values are Objective-C objects, so they are created through `objc_msgSend`**, not
stored as plain python values: a dictionary is an `NSDictionary`, a string an
`NSString`, and the numbers are `NSNumber`s. That is why this module has a small class
instead of a plain dataclass.

This is the one part of the back end that cannot be checked without Metal.framework:
whether `MTLCompileOptions` responds to these selectors is what the Mac run proves.
What *is* checked here is that the option object is built, that the selectors are the
ones MSL documents, and that nil stays nil when nothing was configured -- passing a
half-filled object to the compiler would be worse than passing nil.
"""
from __future__ import annotations

import ctypes
from typing import Any, Dict, List, Optional

from ..driver.objc import ObjC


# The `MTLMathMode` values (`metal-cpp` / the Metal headers name them MTLMathModeFast
# etc.). Only `fast` and `precise` are exposed: `relaxed` exists but is neither of the
# two a caller would ask for by name.
MATH_MODES: Dict[str, int] = {"fast": 0, "relaxed": 1, "precise": 2}


class CompileOptions:
    """The settings `newLibraryWithSource:options:error:` receives.

    Construct with no arguments to get Metal's own defaults; every field left `None`
    is simply not set on the object, so the compiler's default applies.
    """

    # The selectors this sets, in the order they are applied. Declared as a tuple so a
    # test can check them against the SDK's spelling without reading the code.
    SELECTORS = ("setLanguageVersion:", "setFastMathEnabled:",
                 "setMathMode:", "setPreprocessorMacros:")
    __slots__ = ("language_version", "fast_math", "math_mode", "macros")

    def __init__(self, language_version: Optional[int] = None,
                 fast_math: Optional[bool] = None,
                 math_mode: Optional[str] = None,
                 macros: Optional[Dict[str, Any]] = None)->None:
        self.language_version = language_version
        self.fast_math = fast_math
        self.math_mode = math_mode
        self.macros = macros

    @property
    def configured(self)->bool:
        """Whether anything was set; a fully-default object is passed as nil."""
        return any((self.language_version is not None, self.fast_math is not None,
                    self.math_mode is not None, self.macros is not None))

    def __repr__(self)->str:
        if not self.configured:
            return "CompileOptions(defaults)"
        parts: List[str] = []
        if self.language_version is not None:
            parts.append(f"languageVersion={self.language_version}")
        if self.fast_math is not None:
            parts.append(f"fastMath={self.fast_math}")
        if self.math_mode is not None:
            parts.append(f"mathMode={self.math_mode}")
        if self.macros is not None:
            parts.append(f"macros={sorted(self.macros)}")
        return "CompileOptions(" + ", ".join(parts) + ")"


def from_mapping(options: Any)->Optional[CompileOptions]:
    """A `CompileOptions` from what a caller passed, or None for the defaults.

    Accepts a `CompileOptions` (as it is), a mapping with the field names, or
    anything else -- including the `BuildOptions` object OpenCL's hook passes to
    every backend -- which is read as "no MSL options configured" rather than
    raising: `kernel_import` hands the same object to whichever backend owns the
    source, and an OpenCL `BuildOptions` reaching Metal is a normal case, not an
    error. That object's own `-D`/include settings mean nothing to MSL, and
    silently translating them would be a guess.
    """
    if isinstance(options, CompileOptions):
        return options if options.configured else None
    if not isinstance(options, dict):
        return None

    known = ("language_version", "fast_math", "math_mode", "macros")
    if not any(key in options for key in known):
        return None

    # Only the keys that are present are passed, so a field the caller left out stays
    # `None` (the compiler's default) rather than becoming a sentinel object.
    passed = {key: options[key] for key in known if key in options}
    return CompileOptions(**passed)


def build(options: Optional[CompileOptions])->int:
    """The Objective-C object for `options`, or 0 (nil) for the defaults.

    The answer is a +1 reference the caller owns: pass it to
    `newLibraryWithSource:options:error:` and then to `release` (which is what
    `release_compile_options` does). Nothing here keeps it alive.
    """
    if options is None or not options.configured:
        return 0

    if ObjC.lib is None:
        # Reaching here means Metal.framework never loaded, so the caller's real
        # failure is the driver's -- but this code runs *before* `objc_getClass`
        # would have dereferenced the missing library, and the `AttributeError`
        # that produces names nothing useful. The driver records its own reason
        # (`MetaMetal.last_init_error`), so report that instead of a traceback
        # about `NoneType`.
        from ..driver.metal import MetaMetal
        reason: str = getattr(MetaMetal, "last_init_error", "") or "unknown reason"
        raise RuntimeError(
            "cannot build MTLCompileOptions: the Metal runtime is not loaded "
            f"({reason})")

    objc: Any = ObjC
    alloc: int = objc.send_ptr(objc.class_("MTLCompileOptions"), "alloc", ())
    if not alloc:
        return 0
    # `alloc` answers a +1 reference and `init` may answer nil, in which case the
    # allocated object is still this call's to release. A setter that raises leaves
    # the same obligation, so the whole set-up runs under one `try`.
    target: int = objc.send_ptr(alloc, "init", ())
    if not target:
        objc.release(alloc)
        return 0

    try:
        _configure(objc, target, options)
    except BaseException:
        objc.release(target)
        raise
    return target


def _configure(objc: Any, target: int, options: CompileOptions)->None:
    """Set every field `options` names on the already-created object."""
    if options.language_version is not None:
        # `MTLLanguageVersion` and `MTLMathMode` are both `NSUInteger` (pointer-sized).
        objc.send(target, "setLanguageVersion:",
                  (ctypes.c_size_t,), ctypes.c_size_t(options.language_version))
    if options.fast_math is not None:
        objc.send(target, "setFastMathEnabled:",
                  (ctypes.c_bool,), ctypes.c_bool(options.fast_math))
    if options.math_mode is not None:
        objc.send(target, "setMathMode:",
                  (ctypes.c_size_t,), ctypes.c_size_t(MATH_MODES[options.math_mode]))
    if options.macros is not None:
        dictionary: int = _dictionary(options.macros)
        if dictionary:
            objc.send(target, "setPreprocessorMacros:",
                      (ctypes.c_void_p,), ctypes.c_void_p(dictionary))


def _dictionary(macros: Dict[str, Any])->int:
    """An `NSDictionary` of `NSString` -> `NSString` for the preprocessor macros."""
    objc: Any = ObjC
    dictionary: int = objc.send_ptr(objc.class_("NSMutableDictionary"), "dictionary", ())
    if not dictionary:
        return 0
    # `dictionary` is autoreleased (+0), and `setPreprocessorMacros:` retains it for
    # as long as the options object lives -- so it must NOT be retained here: an
    # extra `retain` with no matching `release` leaks the dictionary for the life of
    # the process. The options object the dictionary is handed to is released by the
    # caller once the compile is done (`release_compile_options`), and the dictionary
    # goes with it.
    for key, value in macros.items():
        name: int = _string(str(key))
        text: int = _string(str(value))
        if name and text:
            objc.send(dictionary, "setObject:forKey:",
                      (ctypes.c_void_p, ctypes.c_void_p),
                      ctypes.c_void_p(text), ctypes.c_void_p(name))
    return dictionary


def _string(text: str)->int:
    """An `NSString` for `text`."""
    objc: Any = ObjC
    return objc.send_ptr(objc.class_("NSString"), "stringWithUTF8String:",
                         (ctypes.c_char_p,), ctypes.c_char_p(text.encode("utf-8")))


def release_compile_options(options_id: int)->None:
    """Drop the reference `build` handed back, if it handed back one.

    `build` returns 0 for the defaults, and nil must not be released -- so the
    no-op is part of the contract rather than something the caller re-checks.
    """
    if options_id:
        ObjC.release(options_id)
