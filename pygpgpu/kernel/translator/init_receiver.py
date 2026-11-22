# The receiver's parameter binding, split out of _InitMixin.__init__: the four
# shapes a device function's first parameter can take (a `_self_` field set, a
# structure pointer, a value handed on through a helper, and a cross-class
# classmethod's constant-bound cls) decide what joins the scope and the device
# signature, and together they were the longest stretch of the constructor.
from __future__ import annotations

from typing import List, Tuple



def bind_receiver_parameters(
        self, args:List, param_types:List[object]
        ) -> Tuple[List, List[object], object, str]:
    """Drop the receiver from the positional parameters, per its shape.

    A device function's first parameter is its receiver, but the four shapes carry
    it differently: a `_self_` field set and a structure pointer each take a type
    the caller's own lists supply; a value handed on through a helper is the
    helper's own first argument (see the `receiver_as_argument` flag); and a
    cross-class classmethod's cls binds to a class whose scalar attributes travel
    as compile-time constants, so it takes no type at all. Returns
    (args, param_types, receiver_type, self_param) with the receiver sliced off
    the parameter list it did not need; the caller appends the scope entry and the
    signature entry in declaration order.
    """
    receiver_type:object = None
    self_param:str = getattr(self, "self_param", "") or ""
    if self_param and args and args[0].arg == self_param:
        if self.class_constants:
            # A cross-class classmethod's cls: the reads are compile-time
            # constants (see class_constant_values), so the cls parameter carries
            # no type and does not exist in the device signature at all -- it
            # leaves the parameter list alongside its (absent) type, and only the
            # class_constants table answers the reads. The NAME stays self_param:
            # the attribute visits match the receiver spelling (`cls.offset`)
            # through it, then answer from the constants instead of a scope entry.
            self._class_constant_receiver = args[0].arg
        elif self.receiver_as_argument:
            # The receiver travels as this function's own first argument: take its
            # type off the front and name it before the remaining parameters.
            receiver_type = param_types[0] if param_types else None
            param_types = list(param_types[1:])
        args = args[1:]
    return args, param_types, receiver_type, self_param
