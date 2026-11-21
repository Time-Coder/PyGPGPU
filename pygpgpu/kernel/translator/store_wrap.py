"""How an integer store wraps, which is where the backends disagree.

A kernel assigning into an integer array element reduces the value modulo the element's
width, because that is what numpy does for the same assignment in the interpreter:

    out = np.zeros(1, np.uint8)
    out[0] = 255 + 1          # -> 0
    np.uint8(255) + 1         # -> 0 (numpy wraps)

The device read the element as a 32-bit ``int`` (the documented `infer_ctype` contract), so
the sum is 256, and C's own conversion of 256 into an ``unsigned char`` is *just* the wrap --
except for the negative-float case, where C's float-to-unsigned conversion is undefined and
the compiler answers 0 instead of numpy's modulo (``uint8(-1.0) == 255``).

**The two Apple backends need a different spelling, and it is not cosmetic.** Metal and
Apple's OpenCL convert float->signed by *saturating* at the signed range's edge --
``(signed char)(247.0f)`` answers 127, not 247 -- so the signed hop the other backends rely
on loses every value above 127, and a whole Mandelbrot palette came out dark on both. MSL
and OpenCL C also define float->unsigned as a *clamp* to [0, 2**n), which answers the display
case but not numpy's wrap. Those two therefore get an explicit wrap -- ``fmod`` over the
element's own modulus, then the sign folded back in -- each step a defined operation.
"""
from __future__ import annotations

from typing import Dict


# The unsigned integer element types a store must wrap, and the signed type wide enough to
# hold the same bit pattern. Converting through the signed type is what makes a *negative
# float* store defined: C's float-to-unsigned conversion is undefined for a negative value
# (the compiler answered 0 where numpy stores the modulo), while float-to-signed-integer and
# signed-to-unsigned are both defined.
UNSIGNED_STORE_TYPES:frozenset = frozenset({"uchar", "ushort", "uint", "ulong"})
# The *signed* integer element types, with the bit width of the register they wrap in.
# Apple's compilers saturate a float-to-signed conversion instead of truncating, so a
# signed element needs the same explicit wrap an unsigned one does (a `char` element
# storing 130.0 must answer -126, not the 127 a saturation gives).
SIGNED_STORE_TYPES:Dict[str, int] = {"char": 8, "short": 16, "int": 32, "long": 64}
SIGNED_FOR_UNSIGNED:Dict[str, str] = {
    "uchar": "signed char", "ushort": "short", "uint": "long long", "ulong": "long long",
}
STORE_BIT_WIDTH:Dict[str, int] = {"uchar": 8, "ushort": 16, "uint": 32, "ulong": 64}

# An integer element type whose *stored* value wraps, with the C cast that performs the
# wrap. The destination's own type does the modular reduction, which is exactly numpy's rule
# for the same assignment -- and it is a cast the compiler lowers for free, so no extra
# arithmetic is generated.
WRAPPING_STORE_TYPES:Dict[str, str] = {
    "char": "char", "short": "short", "int": "int", "long": "long",
    "uchar": "uchar", "ushort": "ushort", "uint": "uint", "ulong": "ulong",
    "bool": "bool",
}


class _StoreWrapMixin:
    """The wrap a store into an integer element performs. See the module docstring.

    The host mixin (`_AssignMixin`) supplies `_device_type`, `_next_temp`,
    `_hoist_declaration` and `_emit`, and the flags that tell the backends apart. Only
    `_wrap_for_store` is defined here, so nothing shadows the implementations those other
    mixins bring -- a stub declared here would win the MRO and fail at runtime.
    """

    def _wrap_for_store(self, element_type:object, code:str)->str:
        """`code` cast so a store into an element of `element_type` wraps like numpy.

        The cast is always emitted for an integer destination: it is idempotent when the
        value already fits, and it is the only thing that makes the two sides agree on the
        cases that do not. A float destination is untouched -- casting a `double` into a
        `float` element is ordinary narrowing and both sides agree.
        """
        if not isinstance(element_type, str) or element_type not in WRAPPING_STORE_TYPES:
            return code
        if element_type == "bool":
            return f"(({code}) != 0)"
        device_type:str = self._device_type(element_type)
        # The width the element's modular arithmetic runs at, and whether the destination
        # is signed. Both kinds need the same treatment on Apple (below).
        is_signed_element:bool = element_type in SIGNED_STORE_TYPES
        bits:int = (SIGNED_STORE_TYPES[element_type] if is_signed_element
                    else STORE_BIT_WIDTH.get(element_type, 32))
        wrapped:str = f"(({device_type})({code}))"
        if element_type in UNSIGNED_STORE_TYPES or is_signed_element:
            # Only the narrow widths need the explicit wrap. A 32/64-bit element keeps the
            # integer conversions below, which the Apple compilers perform the same way,
            # and which stay exact where a float hop would not: a float32 cannot hold
            # 4294967295, so wrapping a uint32 through `fmod` would answer 0 for it.
            narrow_bits:int = bits if bits < 32 else 0
            if narrow_bits and (self._metal_syntax or self._opencl_syntax):
                modulus:int = 1 << narrow_bits
                # `fmod` is overloaded on its argument types, so the modulus has to be a
                # `float` literal (`256.0f`, not `256.0`) or OpenCL C reports the call as
                # ambiguous between its float and double overloads. `trunc` normalises the
                # fractional part the way the C cast would, so an in-range value is
                # unchanged.
                wrapped_float:str = f"fmod(trunc((float)({code})), {modulus}.0f)"
                # The remainder is read up to three times below, so it goes through a
                # hoisted local: a compound expression repeated in every arm of a
                # conditional would be evaluated once per arm, which is both harder to
                # read in the generated source and wasteful on the device.
                remainder:str = f"_wrap_rem{self._next_temp()}"
                self._hoist_declaration(remainder, "float", zero="0.0f")
                self._emit(f"{remainder} = {wrapped_float};")
                # Fold a negative remainder up into [0, modulus): numpy's modulo is
                # non-negative, C's fmod keeps the dividend's sign.
                positive:str = (f"({remainder} < 0.0f"
                                f" ? {remainder} + {modulus}.0f"
                                f" : {remainder})")
                if is_signed_element:
                    # A signed element keeps [0, half) unchanged and moves [half, modulus)
                    # down by one modulus. Measuring on the *positive* remainder means
                    # there is no negative edge to handle -- and it is the same rule the
                    # unsigned branch's cast applies, so the two agree at every boundary.
                    half:int = modulus // 2
                    folded:str = f"_wrap_pos{self._next_temp()}"
                    self._hoist_declaration(folded, "float", zero="0.0f")
                    self._emit(f"{folded} = {positive};")
                    wrapped = (f"(({device_type})({folded} >= {half}.0f"
                               f" ? {folded} - {modulus}.0f"
                               f" : {folded}))")
                else:
                    wrapped = f"(({device_type})({positive}))"
            else:
                signed:str = SIGNED_FOR_UNSIGNED.get(element_type, "long long")
                wrapped = (f"(({device_type})((long long)({code})))"
                           if bits >= 32 else
                           f"(({device_type})(({signed})({code})))")
        return f"({wrapped})"
