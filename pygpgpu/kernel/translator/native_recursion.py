"""The native-recursion call sites: admission, emission, and the token finish.

A device helper is monomorphized per call site and cached by (function, argument types,
exception base); a recursive call would revisit the very cache slot that is still being
filled. The unlock (`scalar_recursion`) admits the one shape whose signature the *call
site* can spell before the body is translated, and this mixin carries the call-site half
of that: the verdict (`_native_recursion_admitted`, bar plus seed over the parsed body),
the emission (`_emit_recursive_site`, a token in the expression's place), and the finish
(`_finish_recursive_sites`, the token rewritten into the full call once the cycle's
shapes are final).

The token is the load-bearing piece. The call's own text needs the callee's trailing
parameters -- the shape extents an array parameter indexes through -- and those are
decided by the translation *of the very call the token stands in for*. So the site emits
a unique C identifier no user name can collide with (the reserved `_pg_recursive_`
prefix), records it with the translator whose lines hold it, and the finisher rewrites
every token when the body is about to render. A token composes inside a larger
expression: `1 + <token>` survives to the finish and becomes
`1 + depth(arr, i + 1, _arr_shape0)`.

A **mutual** site closes a cycle whose finisher runs before the *partner's* render, with
the partner's shapes unknown -- so the bar for a mutual cycle is stricter: both members
must be pure scalars, and the call needs no shape argument at all. A self-call's shapes
are the identity (the same function forwards its own extents), which is why the
array-parameter form stays a self-recursion shape.
"""
from __future__ import annotations

import ast
from typing import (Callable, Dict, List, Optional, Tuple)
from .notation import (
    Translated,
    device_name,
)
from .scalar_recursion import (
    RECURSIVE_BACKENDS,
    scalar_bar_met,
)
from .return_type_scan import (
    seed_return_type,
)
from .walk import (
    type_sig,
)


class _NativeRecursionMixin:
    """The call-site half of the native-recursion unlock (see scalar_recursion)."""

    def _emit_recursive_site(self, raw_func:Callable, param_types:List[object],
                             call_codes:List[str], node:ast.AST, name:str,
                             return_type:str)->Translated:
        """Emit one call inside the cycle the site closes, as a token to finish later.

        The call's own text needs the callee's trailing parameters -- the shape extents
        an array parameter indexes through -- and those are decided by the translation
        *of this very call*, which is still running. So the site emits a unique token
        in the expression's place (a C identifier no user name can collide with, thanks
        to the reserved prefix) and records it on the in-progress entry; when the
        cycle's translation completes, every token still in the owning translator's
        lines is replaced by the full call text (see `_finish_recursive_sites`). The
        token reads as an identifier, so it composes inside a larger expression
        (`1 + <token>` survives to the finish and becomes
        `1 + depth(arr, i + 1, _arr_shape0)`).

        A **mutual** site closes a cycle whose finisher runs on the *partner's* final
        signature, and the partner's shape parameters are its own spelling -- so a
        mutual site is only admitted when neither member carries a shape parameter at
        all (the pure-scalar cycle). A self-call's shapes are the identity, which is
        why the array-parameter form stays a self-recursion shape.
        """
        mutual:bool = not self._frame_is_self(raw_func, param_types)
        if mutual and any(isinstance(t, tuple) for t in param_types):
            self._error(
                f"'{name}' is a mutual recursion over a pointer parameter: the cycle's "
                f"call sites would forward each other's shape extents, which a native "
                f"call cannot spell. Make the cycle all scalars, or recurse through a "
                f"single function", node)
        c_name:str = self._device_func_name(raw_func, param_types)
        entry:Optional[Dict[str, object]] = self._recursive_frame_of(raw_func,
                                                                     param_types)
        # The launch's stack raise reads the flag off the *kernel's* translator (see
        # _compile_kernels), while this site may sit inside a nested helper's own
        # walk -- so every translator currently on the cycle's stack learns the unit
        # recurses, and the kernel's own flag is one of them.
        self.uses_native_recursion = True
        for value in self.device_cache.values():
            if isinstance(value, dict) and value.get("status") == "in_progress":
                owner:object = value.get("owner")
                if owner is not None:
                    owner.uses_native_recursion = True
        self._token_counter:int = getattr(self, "_token_counter", 0) + 1
        marker:str = f"{device_name('_pg_recursive')}_{self._token_counter}"
        pending:object = entry.get("pending_sites") if entry is not None else None
        if not isinstance(pending, list):
            pending = []
            if entry is not None:
                entry["pending_sites"] = pending
        pending.append({"token": marker, "c_name": c_name,
                        "call_codes": list(call_codes),
                        "owner": self})
        return (marker, return_type)

    def _recursive_frame_of(self, raw_func:Callable,
                            param_types:List[object])->Optional[Dict[str, object]]:
        """The in-progress entry the cycle this site closes is being filled into."""
        pair:Tuple[object, ...] = (id(raw_func), tuple(param_types))
        for value in self.device_cache.values():
            if isinstance(value, dict) and value.get("status") == "in_progress" \
                    and value.get("frame") == pair:
                return value
        return None

    def _frame_is_self(self, raw_func:Callable, param_types:List[object])->bool:
        """Whether the active frame is this site's own function (a self-call)."""
        frame:Optional[Dict[str, object]] = self._recursive_frame_of(raw_func,
                                                                     param_types)
        return frame is not None and self.func is raw_func


    def _finish_recursive_sites(self)->None:
        """Replace the tokens this translation's own lines carry with the full calls.

        Called on the translator whose body is about to render, before that render.
        For a self-call the callee is this function: the shapes its signature carries
        are the ones `shapes_used` recorded during the walk, and each site forwards
        them under this translator's own spelling -- the identity the same-function
        forwarding gives. For a mutual partner the bar the site checked guarantees a
        pure-scalar cycle, so the call needs no shape argument at all. A token inside
        a larger expression (`1 + <token>`) composes exactly as the python wrote it.
        """
        shape_codes:List[str] = []
        for array_name, dim in self.shapes_used:
            shape_codes.append(self._shape_var(array_name, dim) if dim >= 0
                               else self._tail_var(array_name, -dim - 2))
        for value in self.device_cache.values():
            if not isinstance(value, dict) \
                    or value.get("status") != "in_progress":
                continue
            pending:object = value.get("pending_sites")
            if not isinstance(pending, list):
                continue
            mine:List[Dict[str, object]] = [
                site for site in pending
                if isinstance(site, dict) and site.get("owner") is self]
            if not mine:
                continue
            for site in mine:
                token:str = str(site["token"])
                call:str = (f"{site['c_name']}"
                            f"({', '.join(list(site['call_codes']) + shape_codes)})")
                self.lines = [line.replace(token, call) for line in self.lines]
                pending.remove(site)
            if not pending:
                value.pop("pending_sites", None)


    def _native_recursion_admitted(self, raw_func:Callable,
                                   param_types:List[object])->Tuple[bool, str]:
        """Whether the cycle this site closes becomes a native recursive call.

        Every question is answerable before the callee's body is translated: the
        code-generation backend allows recursion, the body meets the scalar bar (the
        call site can spell the callee's full signature -- see scalar_recursion), and
        the frame on the in-progress entry is *this* pair, whose parameter types are
        exactly the arguments this site passes. The return type comes from the same
        static scan: every `return` names the same scalar, or the unlock stays off.
        The verdict is cached on the function object keyed by the type signature,
        because every site inside one cycle asks the same question about the same
        body -- and a helper recursing under two signatures is two verdicts.
        """
        if self.backend not in RECURSIVE_BACKENDS:
            return (False, "")
        # The frame this site closes already carries the signature the self-call
        # spells: the same function, the same parameter types. A method's body calls
        # itself through `_emit_self_method_call`, which forwards the receiver struct
        # and its `_self_<attr>` slots -- the same identity forwarding an array
        # pointer gets -- and the caller has built that argument list before the
        # verdict is asked. A *value*-method spelling (`w.walk(...)` from outside)
        # shapes the frame the same way, so the recursion inside it is admissible
        # too; the emit below spells the inner call the way the body wrote it.
        cache_key:object = ("__pg_recursion_verdict__", type_sig(param_types))
        verdict:object = getattr(raw_func, "_pg_native_recursion", None)
        verdicts:Optional[Dict[object, object]] = (
            verdict if isinstance(verdict, dict) else None)
        if verdicts is not None and cache_key in verdicts:
            answer:object = verdicts[cache_key]
            return (bool(answer[0]), str(answer[1]))
        from .source_text import _parse_kernel_source, _trimmed_source  # noqa: PLC0415

        globals_map:Optional[dict] = getattr(raw_func, "__globals__", None)
        try:
            module:ast.Module = _parse_kernel_source(_trimmed_source(raw_func))
        except (SyntaxError, OSError):
            module = None
        tree:Optional[ast.FunctionDef] = (module.body[0] if module is not None
                                          and module.body
                                          and isinstance(module.body[0],
                                                         ast.FunctionDef) else None)
        admitted:bool = False
        return_type:str = ""
        if tree is not None:
            admitted = bool(scalar_bar_met(tree, raw_func.__name__, param_types,
                                           globals_map))
            if admitted:
                seed:Optional[str] = seed_return_type(
                    tree, param_types, self.type_map.get("float", "float"),
                    globals_map)
                if seed is None:
                    # The returns disagree (or one of them is not static): the
                    # call site cannot know the type it would report, so the
                    # refusal stands.
                    admitted = False
                else:
                    return_type = seed
        if verdicts is None:
            verdicts = {}
            raw_func._pg_native_recursion = verdicts  # type: ignore[attr-defined]
        verdicts[cache_key] = (admitted, return_type)
        return (admitted, return_type)
