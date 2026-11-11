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
    is_aggregate,
    is_float,
    is_int,
    is_ptr,
)
from .vectors import (
    is_vec,
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


    def _zero_for_type(self, type_text:str)->str:
        """The zero value a hoisted declaration starts at.

        A **record** (`c64`/`c128`, a structure) cannot be cast from an int -- C has no
        conversion to a structure type, and `(c128)0` is a compile error -- so it takes `{0}`,
        which zeroes every member and is valid for a scalar too. A scalar and a pointer keep
        the cast. A **vector** keeps it as well: a scalar cast to a vector is OpenCL's
        broadcast initializer, where brace-initializing one is not portable.

        One vector case cannot keep the cast: a **CUDA** compile. A vector there is a plain
        structure (CUDA's `float3` declares `float x, y, z;` and nothing else), so the cast
        asks C++ for a converting constructor the struct does not have -- "no suitable
        constructor exists to convert from int to float3" -- and it takes the record's own
        `{0}`. HIP's `float3` is a `HIP_vector_type`, which carries exactly that converting
        constructor, and OpenCL reads the cast as the broadcast: both keep the cast, which
        is what their kernels compile today.
        """
        base:str = type_text.replace("const", "").strip()
        if is_aggregate(base):
            return "{0}"
        if is_vec(base) and self.launch_backend == "cuda":
            return "{0}"
        return f"({base})0"


    def _hoist_declaration(self, name:str, type_text:str,
                           zero:Optional[str]=None)->None:
        """Record a local whose declaration belongs at the top of the function.

        An assignment inside a branch would otherwise declare it there, and python scopes
        the name to the whole function, so a read after the branch has nothing to name.
        Recorded once: later assignments to the same name are plain stores. `zero` defaults
        to what the type can express (see `_zero_for_type`); a record needs the brace form.
        """
        if name not in self._hoisted_names:
            self._hoisted_names.add(name)
            self._hoisted_declarations.append(
                (name, type_text, zero if zero is not None else self._zero_for_type(type_text)))


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
        if self._hoisted_declarations:
            # The declarations go at the top of the body, in the order the names were first
            # assigned, and every recorded site shifts by what is inserted in front of it.
            prefix:List[str] = [f"    {type_text} {device_name(name)} = {zero};"
                                for name, type_text, zero in self._hoisted_declarations]
            source = prefix + list(source)
            inserted:int = len(prefix)
            self._declaration_sites = (
                [(index, name, type_text)
                 for index, (name, type_text, _) in enumerate(self._hoisted_declarations)]
                + [(line_index + inserted, name, initial_type)
                   for line_index, name, initial_type in self._declaration_sites])
            self._hoisted_declarations = []
            self._hoisted_names = set()
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
        source node to blame in an error message.

        A **local** name is recorded on the function's own scope, not on the block being
        translated. Python scopes a name to the whole function -- `if flag: value = 1` makes
        `value` local and readable after the `if` -- while a block-scoped record disappears
        when `_push_scope_block` pops, which is what made the same source fail here with
        "unknown name". A name that is already bound in the function scope (a parameter, a
        value an earlier statement defined, a carried scalar) keeps its entry there, because
        shadowing one of those inside a block is refused rather than supported.
        """
        existing:DeviceType = self._lookup(name)
        if existing is None:
            self.scopes[0][name] = t
        elif existing != t:
            if is_int(existing) and is_float(t):
                self._retype(name, t)
            elif is_float(existing) and is_float(t):
                # Widening only ever goes up (float -> double), the same rule the int ->
                # float case above follows: a python variable is untyped, so `scale = 1.0`
                # above the loop followed by `scale = x * 2.0` (a double under the
                # mapping) refused with "changes type from float to double". The scope
                # entry is what every later read and the declaration itself see, so
                # re-pointing it widens the variable in place.
                self._retype(name, t)
            else:
                self._error(f"variable '{name}' changes type from {existing} to {t}", node)


    def _retype(self, name:str, t:DeviceType)->None:
        """Re-point an existing binding, on the scope that already holds it."""
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = t
                return
        self.scopes[0][name] = t


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
