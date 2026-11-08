# The module-level registries, deliberately in a module with no imports of its
# own so the other parts can share them without a cycle. They are mutated in
# place and must never be re-created.
from __future__ import annotations
from typing import Dict, List
import numpy as np



# Dtypes that a `ctypes.Structure` class declared. A dtype cannot say whether it means a
# vector or a structure -- `{float x; float y;}` and `float2` compare and hash equal -- so the
# only usable evidence is the dtype *object*: numpy marks a record built from a ctypes class
# as aligned (`isalignedstruct`), which the vector dtypes this package ships are not.
# Every class that declared this layout, in registration order. Two classes can share one
# dtype, and a member call has to find the one that actually carries the method.
_DECLARED: Dict[np.dtype, type] = {}


_DECLARED: Dict[np.dtype, type] = {}
_DECLARED_ALL: Dict[np.dtype, List[type]] = {}



_REGISTRY: Dict[str, np.dtype] = {}


_REGISTRY: Dict[str, np.dtype] = {}
# dtype -> the python class name its structure is spelled after. The C name is
# `<base>_<hash>`: readable, and still distinct when two modules declare one class name.
_NAME_BASES: Dict[np.dtype, str] = {}


_NAME_BASES: Dict[np.dtype, str] = {}
# dtype -> the C name it was first given. Pinned, because a later class registration must
# not rename a layout an existing signature already spells.
_SETTLED_NAMES: Dict[np.dtype, str] = {}


_SETTLED_NAMES: Dict[np.dtype, str] = {}
# dtype -> the python file the class behind it was written in. Kept separately from
# `_DECLARED`, because a structure synthesized from a plain object has a *generated* class
# (built by this package) while its members were inferred from the caller's source: the
# declaration belongs to that caller's module, not to the file that built the class.
_DECLARED_FILES: Dict[np.dtype, str] = {}
