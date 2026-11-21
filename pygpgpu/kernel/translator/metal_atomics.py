"""The Metal statement-block atomics, as their own emission subject.

A Metal `atomic_cas` and every 64-bit atomic are not expressions: MSL spells the
CAS as a compare-exchange that answers a bool and takes the expected value by
pointer, and MSL has no 64-bit atomic at all. Both are emitted as a brace block
that leaves the old value in a hoisted `int` local, which the value use reads
(see `device_builtins.atomics._emit_metal_cas_loop` for the block's text).
"""
from __future__ import annotations

import ast

from ..device_builtins import emit_atomic
from .builtin_python import _BuiltinPythonMixin
from .notation import Translated


class _MetalAtomicBlockMixin(_BuiltinPythonMixin):
    """The Metal statement-block atomic emission, split from builtin_emit."""

    def _emit_metal_atomic_block(self, name:str, node:ast.Call, arr_code:str,
                                 elem:str, idx_code:str, val_code:str,
                                 compare_code:str)->Translated:
        """The Metal CAS and every 64-bit atomic, as a statement block.

        The block leaves the old value in a hoisted `int` local whose name is unique
        per call site (a GNU statement-expression is not portable MSL, and MSL has no
        64-bit atomic at all -- see `atomics._emit_atomic_metal`). A bare statement
        call answers nothing (the statement walker would add a valueless
        `<name>;` after the block); a value use reads the hoisted name.
        """
        if elem in ("long", "ulong"):
            self.has_atomic64 = True
        # Each call site gets its own result local. Two blocks in one expression (an
        # `a[i] = atomic_cas(...) + atomic_cas(...)` write, say) used to share one
        # `_cas_result`, so both reads answered whichever block ran last -- a wrong
        # number on the device, with nothing failing.
        result_name:str = f"_cas_result{self._next_temp()}"
        block:str = emit_atomic(name, arr_code, idx_code, val_code, elem,
                                self.backend, compare_code,
                                is_local=self._is_local_buffer(node.args[0]),
                                result_name=result_name)
        self._hoist_declaration(result_name, "int", zero="0")
        self._emit(block.replace(f"    int {result_name};\n", ""))
        if getattr(node, "_pygpgpu_bare_statement", False):
            return "", "void"
        return result_name, "int"
