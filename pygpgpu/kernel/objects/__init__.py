"""Arbitrary python objects as structure arguments.

Anywhere this package accepts a structure -- a by-value argument, a nested member, an element
of a record array -- an arbitrary python object is accepted too. The structure is synthesized
at launch, and its members are the attributes the *kernel body actually reads*: a class with
fifty attributes contributes only the two its reader mentions. That set is decided by a static
scan of the kernel and the helpers it calls, because the device type has to exist before the
argument can be typed.

The scan follows the narrow shapes that can be answered:

* ``p.x`` on a parameter bound to an object;
* ``q = p; q.x`` (a local alias);
* ``helper(p)`` -- the helper's parameter joins the scan, recursively;
* ``p.method()`` -- the method's own ``self.xxx`` reads join the set through the same
  ``scan_self_attrs`` the ``self`` feature uses, and the method is copied onto the synthesized
  class so the translator's existing receiver support finds it.

Anything else is refused with a message rather than guessed at: a nested object member, an
array member, a write to the object, ``getattr``, or an object reached through a container.

Two objects of the same class are merged into one structure *when they can be*: the union of
their members, with ``int`` widened to ``float`` where the two disagree. They stay two
structures when merging would be dishonest -- one instance lacks a member the other needs, or
the same member has types that do not promote."""
# This module is the facade of a split: every name it used to define is re-exported
# from the part module that now owns it, so `objects.<name>` and
# `from pygpgpu.kernel.objects import <name>` keep working unchanged.
#
# The module-level caches live in `state.py` and are imported as objects everywhere:
# a second copy of one would re-initialise it and drop what a previous kernel
# registered in this process.
from __future__ import annotations



__all__ = [
    "looks_like_object",
    "convert_object_arguments",
]


from .state import (
    INLINE_PREFIX,
    _CONSTRUCTED,
    _CONSTRUCTED_PARAMETERS,
    _DEFAULT_FLOAT_MEMBER,
    _DEFAULT_INT_MEMBER,
    _DEVICE_SCALAR_CLASSES,
    _LIFTED_MEMBERS,
    _LIFTED_TREES,
    _LIFTED_VALUES,
    _PROMOTIONS,
    _STRUCT_CACHE,
)
from .scan import (
    Scan,
    _called_members,
    _element_path_of,
    _follow_aliases,
    _list_root,
    _member_path,
    _parse,
    _resolve_list_path,
    _scan_function,
    _tree_key,
    _writes_to,
    looks_like_object,
    self_chain,
)
from .construction import (
    _constructs_a_class,
    _init_parameters,
    _inline_constructions,
    constructed_locals,
    constructed_parameters,
    constructed_structure,
    forget_constructed,
)
from .kinds import (
    _argument_kind,
    _assigned_kind,
    _ctypes_name_of,
    _expression_kind,
    _fields_for,
    _is_number,
    _numeric_member_type,
    _prefer_class_name,
    _structure_for,
    _template_from,
    member_type,
    merge_fields,
    numeric_list_as_array,
    numeric_list_type,
    promote,
    scalar_member_type,
)
from .binding import (
    ObjectBinding,
    _complex_member_value,
    _is_complex_member_class,
    _member_home_value,
)
from .convert import (
    convert_object_arguments,
    scan_object,
)
