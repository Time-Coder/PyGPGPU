from __future__ import annotations
import ast
import types
from typing import (Dict, List, Optional, Set)
from .notation import (
    DeviceType,
    OuterVarEntry,
    PTR,
)
from .predicates import (
    infer_ctype,
    is_ptr,
)
from .walk import (
    _BUILTIN_NAMES,
    _UNRESOLVED,
    _declared_scope_names,
)


class _OuterVarsMixin:

    def _bind_outer_variables(self)->None:
        """Turn every outer name the body uses into a device parameter.

        Python resolves such a name through the enclosing function's closure or the
        module globals; a device kernel has neither, so the value travels as an argument
        the host supplies at launch (see PyKernel._compile / _invoke_inner). Which names
        qualify is decided from the decorated function itself:

        * a `nonlocal`/`global` declaration makes the name writable: it becomes `T*` and
          the body reads and writes it through the pointer, so the host can copy the final
          value back into the scope;
        * any other name the function references but does not define is a free variable
          (`co_freevars`, i.e. a closure cell) or a module global (`co_names`): it becomes
          a plain `T` parameter and is read-only, because python itself would not let the
          body assign to it without a declaration.

        A cell's content can change between calls, so a read-only name is still a
        parameter -- inlining the current value would make the kernel stale, which is the
        whole reason this exists.
        """
        code:object = getattr(self.func, "__code__", None)
        if code is None:
            return
        cell_names:Set[str] = set(code.co_freevars)
        declared:Set[str] = _declared_scope_names(self.tree)
        # Names the body defines locally never reach the scope; parameters are already bound.
        bound:Set[str] = self._local_names() | set(self.self_attr_types)
        bound |= {name for name, _ in self.params}
        if self.self_param:
            bound.add(self.self_param)
        # `co_names` also holds attribute names (`x.shape` contributes "shape") and the
        # names of called builtins/helpers, so only names the body actually reads as a
        # plain variable can become parameters. A name the body *assigns* without a
        # declaration is python's local, not a scope variable, so it is not a candidate --
        # but `co_names` still lists it for the global case, hence the `referenced` filter.
        referenced:Set[str] = self._referenced_names()

        candidates:List[str] = []
        for name in sorted(declared | cell_names | referenced | set(code.co_names)):
            if name in bound or name in self.outer_vars:
                continue
            if name in _BUILTIN_NAMES:
                continue
            candidates.append(name)
        for name in candidates:
            value:object = self.resolve_global(name)
            if value is _UNRESOLVED:
                continue
            entry:Optional[OuterVarEntry] = self._outer_var_entry(name, value, code, declared)
            if entry is None:
                continue
            # Which module's namespace the name came from. A helper imported from its own
            # module may read a global of *that* module, and the host has to look there --
            # reading the kernel's globals would find nothing and pass None.
            entry["scope"] = getattr(self.resolve_global, "source_globals", None)
            self.outer_vars[name] = entry
            if entry["writable"]:
                self.params.append((name, (PTR, entry["type"])))
            else:
                self.params.append((name, entry["type"]))


    def _outer_var_entry(self, name:str, value:object, code:object,
                         declared:Set[str])->Optional[OuterVarEntry]:
        """The device parameter for one outer name, or None when it is not a value.

        A module, a class, a builtin or a helper function is not a value a parameter can
        carry: those names are resolved by the translator's own dispatch (numpy is
        `np`, `gpu`, `math`, helper calls, ...), so they are left alone. An outer name
        that *is* a value has to be a supported argument type, and a declared write has to
        be a scalar, because the scope only ever holds that variable's value.
        """
        if isinstance(value, (type, types.ModuleType)) or callable(value):
            # A module, class, builtin or helper function is not a value a parameter can
            # carry: those names are resolved by the translator's own dispatch (numpy is
            # `np`, `gpu`, `math`, helper calls, ...), so they are left alone.
            return None
        writable:bool = name in declared
        try:
            device_type:DeviceType = infer_ctype(value, self.type_map)
        except (SyntaxError, ValueError) as error:
            if not writable:
                # Not a name this feature handles; the normal "unknown name" path will
                # explain it with the right position if the body really reads it.
                return None
            raise type(error)(f"outer variable '{name}' cannot be written back: {error}") from None
        if writable and is_ptr(device_type):
            raise SyntaxError(
                f"cannot rebind the outer array '{name}': the array object itself is "
                f"shared with the caller, so a device kernel can only write its elements") from None
        return {"type": device_type, "writable": writable,
                "kind": ("closure" if name in code.co_freevars else "global")}


    def _require_outer_parameter(self, name:str, outer:Dict[str, object], node:ast.AST)->None:
        """Make sure the kernel carrying a helper's outer variable receives it too.

        The kernel's own scan only sees names its body mentions; a helper reached from it
        may reference more. The parameter is appended to the kernel's signature here, so
        the launch supplies it from the same scope (see PyKernel._compile).
        """
        if name in self.outer_vars:
            existing:Dict[str, object] = self.outer_vars[name]
            if existing["writable"] != outer["writable"] or existing["type"] != outer["type"]:
                self._error(
                    f"outer variable '{name}' is used with conflicting types by a helper: "
                    f"{existing['type']} here and {outer['type']} in the helper", node)
            return
        self.outer_vars[name] = dict(outer)
        if outer["writable"]:
            self.params.append((name, (PTR, outer["type"])))
        else:
            self.params.append((name, outer["type"]))


    def _outer_argument(self, name:str, outer:Dict[str, object])->str:
        """The call-site expression that forwards an outer variable to a helper."""
        return f"&{name}" if outer["writable"] else name
