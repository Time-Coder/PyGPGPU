from __future__ import annotations
import ast
from typing import (Dict, List, NoReturn, Optional)
from .source_text import (
    get_dedented_source,
)
from .errors import (
    _syntax_error,
)
from .notation import (
    DeviceType,
    HostValueEntry,
    OuterVarEntry,
    PTR,
    device_name,
)
from .predicates import (
    is_float,
    is_int,
    is_ptr,
)


class _EmitMixin:

    @property
    def printf_backend(self)->str:
        """The key into the printf tables: hip compiles CUDA-style code, so it joins cuda."""
        return "cuda" if self.backend in ("cuda", "hip") else "opencl"


    def _error(self, message:str, node:Optional[ast.AST]=None)->NoReturn:
        """Raise a translation error. Declared NoReturn so callers can rely on it not
        falling through (the checker then stops treating each call site as nullable).

        ``from None`` suppresses the chain: the translator's own frames are an implementation
        detail, and what matters is the position drawn from the user's file.
        """
        raise _syntax_error(message, node, self.filename, self.source_text,
                            self.first_line) from None


    def _emit(self, line:str)->None:
        self.lines.append("    " * self._indent + line)


    def _emit_blank(self)->None:
        """A separator line, recorded so the renderer can collapse or trim it."""
        if self.lines and self.lines[-1] != "":
            self.lines.append("")


    def _render_body(self, lines:Optional[List[str]]=None)->str:
        """The emitted lines with separators placed where a reader expects them.

        The walkers record a blank line after every closed block and between statement
        runs. This pass keeps one where a *new statement at the same scope* follows a run
        or a block, and drops one where the scope simply ends (the next line dedents back
        to the parent), because there is nothing more in this scope to separate. Doubles
        collapse, and no blank may lead the body or survive at its end.

        Before the separators are placed, every local declaration whose variable a later
        assignment *widened* (`scale = 1.0` here, a double inside the loop) has its type
        rewritten -- the declaration was spelled with the type the first assignment had,
        and the C compiler rejects a store of the wider value into it.
        """
        source:List[str] = (self.lines if lines is None else lines)
        if self._declaration_sites:
            source = list(source)
            for line_index, name, initial_type in self._declaration_sites:
                final_t:DeviceType = self._lookup(name)
                final_type:str = (self._device_type(final_t)
                                  if isinstance(final_t, str) and not is_ptr(final_t)
                                  else "")
                if final_type and final_type != initial_type:
                    line:str = source[line_index]
                    marker:str = f"{initial_type} {device_name(name)} = "
                    stripped:str = line.lstrip()
                    if stripped.startswith(marker):
                        indent:str = line[:len(line) - len(stripped)]
                        source[line_index] = (f"{indent}{final_type} "
                                              f"{device_name(name)} = "
                                              f"{stripped[len(marker):]}")
        rendered:List[str] = []
        for line in source:
            if line == "" and (not rendered or rendered[-1] == ""):
                continue
            rendered.append(line)
        # A block that ends just before the brace closing its parent (`while { ... }` as
        # the last statement of a kernel) leaves two closing braces on consecutive lines.
        # The inner one still ends a block, so the separator every closing brace gets is
        # inserted between them.
        # The trailing blank the caller may have appended is trimmed first: the last real
        # line is what decides whether a separator is needed before it.
        trimmed:List[str] = list(rendered)
        while trimmed and trimmed[-1] == "":
            trimmed.pop()
        kept:List[str] = list(trimmed)
        # A blank line directly before a closing brace separated a block that ends there.
        # When the very next line closes the *parent*, that separator would sit between two
        # braces instead of between two statements, so it is dropped. Every other closing
        # brace keeps its own separator (`if`/`else` already has none by construction).
        spaced:List[str] = []
        for index, line in enumerate(kept):
            if line == "" and index + 1 < len(kept) and kept[index + 1].strip() == "}":
                continue
            spaced.append(line)
        kept = spaced
        while kept and kept[0] == "":
            kept.pop(0)
        while kept and kept[-1] == "":
            kept.pop()
        return "\n".join(kept)


    def _lookup(self, name:str)->DeviceType:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # A host-created object and an outer variable are device parameters too, so a
        # type query has to see them; the magic-loop scan asks which arrays exist before
        # any statement of the body has been visited.
        host:Optional[HostValueEntry] = self.host_values.get(name)
        if host is not None:
            return host["type"]
        host_scalar:Optional[DeviceType] = self.host_scalars.get(name)
        if host_scalar is not None:
            return host_scalar
        for scope in reversed(self.scopes):
            if name in scope:
                declared:object = scope[name]
                if isinstance(declared, tuple) and len(declared) == 2 and declared[0] == "carried":
                    return declared
                break
        outer:Optional[OuterVarEntry] = self.outer_vars.get(name)
        if outer is not None:
            # A declared outer scalar is a pointer on the device (see _visit_name).
            if outer["writable"] and not is_ptr(outer["type"]):
                return (PTR, outer["type"])
            return outer["type"]
        return None


    def _define(self, name:str, t:DeviceType, node:Optional[ast.AST])->None:
        """Record a variable's type; node is optional because some declarations have no
        source node to blame in an error message."""
        existing:DeviceType = self._lookup(name)
        if existing is None:
            self.scopes[-1][name] = t
        elif existing != t:
            if is_int(existing) and is_float(t):
                self.scopes[-1][name] = t
            elif is_float(existing) and is_float(t):
                # Widening only ever goes up (float -> double), the same rule the int ->
                # float case above follows: a python variable is untyped, so `scale = 1.0`
                # above the loop followed by `scale = x * 2.0` (a double under the
                # mapping) refused with "changes type from float to double". The scope
                # entry is what every later read and the declaration itself see, so
                # re-pointing it widens the variable in place.
                self.scopes[-1][name] = t
            else:
                self._error(f"variable '{name}' changes type from {existing} to {t}", node)


    def _is_const_struct(self, name:str)->bool:
        """Whether a name denotes a structure parameter the body never writes.

        Its signature carries `const`, so an alias of it cannot be mutable.
        """
        binding:DeviceType = self._lookup(name)
        if not (isinstance(binding, tuple) and is_ptr(binding)
                and isinstance(binding[1], str)):
            return False
        from .. import structs as _structs_module  # noqa: PLC0415

        if _structs_module.dtype_for(binding[1]) is None:
            return False
        return name not in self.member_writes


    def _trim_source(self)->str:
        return get_dedented_source(self.func)


    @property
    def nests_a_launch(self)->bool:
        """True when this kernel launches another kernel, i.e. needs linked device code."""
        return bool(self.nested_kernels)


    def _next_temp(self)->str:
        """A fresh name suffix for the temporaries a translation step introduces."""
        self._tmp_counter += 1
        return str(self._tmp_counter)


    def _eval_constant(self, node:Optional[ast.expr])->object:
        """Evaluate a compile-time expression: a shape, a dtype, a fill value.

        Only the forms a creation statement uses are supported (constants, names resolved in
        the kernel's namespace, attribute access, and calls), and an unsupported form raises
        so the caller can refuse the statement with its own message.
        """
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.resolve_global(node.id)
        if isinstance(node, ast.Attribute):
            return getattr(self._eval_constant(node.value), node.attr)
        if isinstance(node, ast.Call):
            function:object = self._eval_constant(node.func)
            arguments:List[object] = [self._eval_constant(argument) for argument in node.args]
            keywords:Dict[str, object] = {keyword.arg: self._eval_constant(keyword.value)
                                       for keyword in node.keywords if keyword.arg}
            return function(*arguments, **keywords)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_constant(element) for element in node.elts)
        raise SyntaxError("not a compile-time constant expression")
