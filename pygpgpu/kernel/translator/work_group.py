# Declaring work-group storage: the four names that build `__shared__`/`__local` memory.
from __future__ import annotations

import ast
from typing import (List, Optional)

import numpy as np

from .notation import (
    BLOCK_ARRAY,
    PER_GROUP_BUILTINS,
)
from .type_map import (
    DTYPE_MAP,
)


class _WorkGroupMixin:

    def _handle_local_declaration(self, node:ast.Assign)->None:
        """`buf = local_array_per_group(64, np.float32)` -> work-group shared storage.

        Two spellings build one declaration, and the difference is the host model -- which
        work item's view of the storage the *name* stands for:

        * `local_array_per_group` / `shared_array_per_block` build a value that stands for
          **every group's slice**, so a body indexes it by its group first
          (`buf[blockIdx][j]`). A work item can only touch memory inside its own group, so
          the subscript is a statement about which group is meant rather than an offset.
        * `local_array` / `shared_array` build **this block's own** array, indexed directly
          (`buf[j]`). That is only meaningful in a gid-style body, where the body is one work
          item and the block it runs in is a fact about the launch; a magic-loop body runs
          once for the whole launch, so this pair is refused there (see the check below).

        opencl: one `__local` parameter per buffer, sized for a single group;
        metal: the same, spelled `threadgroup`;
        cuda: a `__shared__` array, which is already per-group.
        """
        call:ast.Call = node.value
        builtin:str = self._builtin_call_name(call) or PER_GROUP_BUILTINS[0]
        per_group:bool = builtin in PER_GROUP_BUILTINS
        if len(call.args) != 2 or call.keywords:
            self._error(f"{builtin}(shape, dtype) takes exactly 2 arguments", call)
        if not per_group:
            self._require_block_style(node, call, builtin)

        # `np.zeros(shape, dtype)`, argument for argument: the shape comes first and may be
        # an int (`256`) or a tuple (`(64, 64)`), and the dtype second.
        shape_node:ast.expr = call.args[0]
        extents:Optional[List[int]] = self._constant_shape(shape_node, f"{builtin}()")
        if extents is None:
            extents = self._baked_shape(shape_node)
        if extents is None:
            self._error(
                f"{builtin}() shape must be a constant positive integer, a tuple of "
                "them, or an int parameter of this kernel: the buffer is a fixed-size "
                "__shared__/__local array whose size is baked into the generated source, so "
                "a kernel parameter specializes the kernel per value (each value compiles "
                "its own kernel) while anything computed at launch time -- a host-region "
                "value, an expression -- cannot reach it", shape_node)
            raise AssertionError("unreachable: _error raises")  # narrows `extents` for the checker

        dtype_name:str = self._work_group_dtype(call.args[1], builtin)
        ctype:str = DTYPE_MAP[dtype_name]
        if ctype == "half":
            self._error(f"{builtin}() does not support half (fp16)", call.args[1])

        total:int = 1
        for extent in extents:
            total *= extent
        c_name:str = f"_local{len(self.local_buffers)}"
        self.local_buffers.append((c_name, ctype, total))

        target:ast.expr = node.targets[0]
        if not isinstance(target, ast.Name):
            self._error(f"{builtin}() result must be assigned to a plain variable", node)
        kind:str = "shared" if per_group else BLOCK_ARRAY
        self._define(target.id, (kind, ctype, c_name, total, tuple(extents)), node)


    def _work_group_dtype(self, node:ast.expr, builtin:str)->str:
        """The dtype name a work-group declaration carries, refusing anything else.

        A numpy attribute (`np.float32`) or its string spelling (`"float32"`) is what
        `np.zeros` accepts, so those are the two forms; the closed set is `DTYPE_MAP`'s.
        """
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if self.resolve_global(node.value.id) is np:
                name:str = node.attr
            else:
                self._error(f"{builtin}() dtype must be a numpy dtype like np.float32", node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        else:
            self._error(f"{builtin}() dtype must be a numpy dtype like np.float32", node)
        if name not in DTYPE_MAP:
            self._error(f"{builtin}() unsupported dtype: {name}", node)
        return name


    def _require_block_style(self, node:ast.Assign, call:ast.Call, builtin:str)->None:
        """Refuse a block-local array where the body has no single work item.

        Three shapes are refused, and each is a different reason:

        * a **device helper** cannot declare work-group storage at all: a helper is inlined
          into its caller, so the declaration belongs to the kernel;
        * a **magic-loop body** runs once for the whole launch, so a bare block array would
          have no work group to belong to;
        * a body with **no work item** runs once as a plain call whose only block is 0, so a
          bare array would quietly be a one-block scratch pad nothing shares.

        The last check scans the whole body rather than the flag `uses_global_id`, because
        that flag is only set when the call is *visited*, and the declaration may sit above
        the dispatch line. A `@kernel(domain=...)` declaration is known up front.
        """
        if not self.is_kernel:
            self._error(
                f"{builtin}() declares work-group storage for a kernel body: a device helper "
                f"is inlined into its caller, so the declaration has to live in the kernel. "
                f"Build the buffer in the kernel and pass it to this helper", call)
        if self.magic_loop_present:
            self._error(
                f"{builtin}() is for the gid style only: a magic-loop body runs once for the "
                f"whole launch, so a bare block array would have no work group to belong to. "
                f"Use local_array_per_group(...) and index it by its group "
                f"(`buf[blockIdx][j]`)", call)
        if self.uses_global_id or self.gid_param or self.declared_work_arrays:
            return
        dispatches:bool = any(
            isinstance(child, ast.Call) and self._builtin_call_name(child) == "get_global_id"
            for child in ast.walk(self.tree))
        if dispatches:
            return
        self._error(
            f"{builtin}() needs a work item: index the body with `get_global_id(0)` (or "
            f"declare `@kernel(domain=...)`) so this block's storage has a block to belong "
            f"to. A body with no work item runs once as a whole, and "
            f"local_array_per_group(...) indexed by `blockIdx` is the spelling for that",
            node)
