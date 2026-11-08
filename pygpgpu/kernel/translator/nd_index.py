# The ND index machinery: launch ids folded into one work position.
from __future__ import annotations
import ast
from typing import (List, Optional, Set, Tuple)
import numpy as np
from .notation import (
    DeviceType,
    Translated,
)
from .predicates import (
    is_image,
    is_int,
    is_ptr,
)


class _NdIndexMixin:

    def _nd_magic_target(self, iter_node:ast.expr)->Optional[Tuple[str, str, str]]:
        """Parse the np.ndindex(...) / np.ndenumerate(...) magic loops.

        Returns (form, array parameter name, module name); ndenumerate's element type is
        looked up in scope by the caller.

        numpy accepts three equivalent spellings for the same iteration domain, and all of
        them have to be recognised -- otherwise the loop is not seen as a magic loop at
        all, every statement above it is translated as device code, and a host-created
        array among them is never lifted out:

            for i, j in np.ndindex(arr.shape):
            for i, j in np.ndindex(arr.shape[0], arr.shape[1]):
            for i, j in np.ndindex(2, 3):

        The number of dimensions comes from the names the loop unpacks; this method only
        records where each extent comes from (``self._nd_loop_extents``), and None means
        "the single array's own shape, one dimension per unpacked name".
        """
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Attribute)
                and isinstance(iter_node.func.value, ast.Name) and not iter_node.keywords):
            return None
        module:object = self.resolve_global(iter_node.func.value.id)
        if module is not np:
            return None
        form:str = iter_node.func.attr
        if form not in ("ndindex", "ndenumerate"):
            return None

        if form == "ndindex":
            explicit:Optional[List[Tuple[str, int]]] = self._ndindex_extents(iter_node)
            if explicit is None:
                return None
            self._nd_loop_extents = explicit or None
            # A literal-only domain (np.ndindex(2, 3)) names no array, which is valid: the
            # extents are stated outright and no shape parameter is involved.
            holder:str = (explicit[0][0] if explicit else self._nd_shape_holder)
            return form, holder, iter_node.func.value.id

        if len(iter_node.args) != 1:
            return None
        arg:ast.expr = iter_node.args[0]
        param:str = ""
        if isinstance(arg, ast.Name):
            # A local alias of an array (`src = a`, then `ndenumerate(src)`) names the same
            # buffer, so its extents are the aliased array's own.
            param = self._array_pointer_root(arg.id)
        elif (self.self_param and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == self.self_param):
            param = f"_self_{arg.attr}"
        if not param:
            return None
        holder_type:DeviceType = self._lookup(param)
        # An array contributes its extents as pointer parameters; an image carries
        # its extents in its own descriptor, which the backend knows at launch. Both
        # can define the iteration domain of a magic loop.
        if not (is_ptr(holder_type) or is_image(holder_type)):
            return None
        self._nd_loop_extents = None
        return form, param, iter_node.func.value.id


    def _ndindex_extents(self, iter_node:ast.Call)->Optional[List[Tuple[str, int]]]:
        """Where each extent of one np.ndindex(...) call comes from.

        An empty list means the call named a whole array shape (``np.ndindex(arr.shape)``
        or ``np.ndindex((a, b))``), so its dimensions are the loop's own unpacked names.
        Otherwise one entry per extent: ``(array, dimension)`` for an array dimension, or
        ``("", value)`` for a literal count.
        """
        args:List[ast.expr] = list(iter_node.args)
        if len(args) == 1:
            arg:ast.expr = args[0]
            if isinstance(arg, ast.Attribute) and arg.attr == "shape":
                holder:Optional[str] = self._shape_holder_param(arg)
                if holder is None:
                    return None
                holder_type:DeviceType = self._lookup(holder)
                if not (is_ptr(holder_type) or is_image(holder_type)):
                    return None
                self._nd_shape_holder = holder
                return []
            if isinstance(arg, ast.Tuple):
                args = list(arg.elts)
            elif isinstance(arg, ast.Subscript) and self._extent_source(arg) is not None:
                # A single axis spelled out: `np.ndindex(arr.shape[0])`. numpy reads it as
                # a one-dimensional domain of that length, exactly like the tuple and
                # multi-argument forms, so the loop is the work-item dispatch just the same.
                # This used to fall through to "not a magic loop", which pushed the creation
                # above the loop into device code and reported it as a *second* magic loop --
                # a long message about a shape the body never wrote.
                args = [arg]
            else:
                return None

        extents:List[Tuple[str, int]] = []
        for arg in args:
            pair:Optional[Tuple[str, int]] = self._extent_source(arg)
            if pair is None:
                return None
            extents.append(pair)
        return extents


    def _extent_source(self, node:ast.expr)->Optional[Tuple[str, int]]:
        """One extent of np.ndindex: ``<array>.shape[d]``, ``self.<attr>``, or a literal count."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            # np.ndindex(2, 3): the iteration domain is stated outright.
            return "", int(node.value)
        if isinstance(node, ast.Name):
            # `rows: int = out.shape[0]` above the loop, then `np.ndindex(rows, columns)`.
            # The name's *value* is the extent expression the assignment bound it to, so
            # the bound resolves through the alias table instead of the variable itself
            # (which travels as a plain C int, not as the array's extent parameter).
            binding:Optional[Tuple[str, int]] = self._shape_aliases.get(node.id)
            if binding is not None:
                holder_type:DeviceType = self._lookup(binding[0])
                if is_ptr(holder_type) or is_image(holder_type):
                    return binding
            return None
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and self.self_param and node.value.id == self.self_param):
            # A host-side integer attribute is a constant for this launch, like a literal:
            # `np.ndindex(self.image_height, self.image_width)`. The extent is read from the
            # argument carrying the attribute (see the -2 marker in PyKernel._invoke_inner).
            attribute_type:object = self.self_attr_types.get(node.attr)
            if attribute_type is not None and is_int(attribute_type):
                return f"_self_{node.attr}", -2
            if node.attr in self.self_struct_members:
                # The attribute travels inside the receiver struct: the extent reads the
                # member through the pointer (the -3 marker in _nd_bound_code, and the
                # member name in work_size_members for the host).
                member:str = self.self_struct_members[node.attr]
                from .. import structs as _structs_module  # noqa: PLC0415

                struct_dtype:Optional[np.dtype] = _structs_module.dtype_for(
                    self.self_struct_type[1])
                member_type:object = _structs_module.member_types(
                    struct_dtype, self.backend).get(member)
                if member_type is not None and is_int(member_type):
                    self.work_size_members.append(node.attr)
                    return f"(self)->{member}", -3
            return None
        if not isinstance(node, ast.Subscript):
            return None
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return None
        holder:Optional[str] = self._shape_holder_param(node.value)
        if holder is None:
            return None
        holder_type:DeviceType = self._lookup(holder)
        if not (is_ptr(holder_type) or is_image(holder_type)):
            return None
        return holder, node.slice.value


    def _nd_bound_code(self, source:Tuple[str, int])->str:
        """The C expression bounding one dimension (a shape variable, a scalar, or a literal)."""
        name, dim = source
        if not name:
            return str(dim)
        if dim == -2:
            # `np.ndindex(self.image_height, ...)`: the extent is the value of that argument
            # itself, which travels as a plain scalar parameter.
            return name
        if dim == -3:
            # The extent is a receiver-struct member: read it through the pointer.
            return name
        return self._shape_var(name, dim)


    def _launch_id_code(self, axis:int)->str:
        """The work-item id along one launch axis."""
        # Recorded so a device helper can tell whether it needs Metal's prologue
        # parameters passed down to it (see `_needs_prologue_parameters`).
        self._uses_index_builtins = True
        if self._opencl_syntax:
            # Metal shares this spelling: its prelude defines `get_global_id(dim)` as a
            # macro over the attribute-decorated kernel parameters (see metal_codegen).
            return f"get_global_id({axis})"
        component:str = (".x" if axis == 0 else ".y" if axis == 1 else ".z")
        return f"(int)(blockIdx{component} * blockDim{component} + threadIdx{component})"


    def _emit_nd_head(self, names:List[str], param:str)->None:
        """Multi-dimensional work-item head: names[0] maps to dimension 0 (slowest).

        The body compiles once per signature and the generated source is cached by it, so the
        head cannot depend on how many axes the *launch* happens to have -- a bracket is not
        part of the signature, and a source compiled for a 1-D launch would be reused by a
        2-D one. It is therefore always the same statement: fold the launch's ids into one
        flat work-item position, then decompose that position over the loop's own domain.

        Every axis folds the same way, which is what makes the launch rank irrelevant.
        Normalising each id by its grid extent (the C remainder semantics, so a one-item axis
        contributes its only value) and then by the stride to the next axis gives a position
        where axis 0 is fastest -- and the decomposition, which has the *last* loop name
        fastest, walks the domain from the last stride back. A launch of exactly the loop's
        rank and one of a single flat axis are both instances of the same fold, so neither
        needs a different source.
        """
        ndim:int = len(names)
        self.work_item_indices = list(names)
        sources:List[Tuple[str, int]] = self._nd_loop_extents or [(param, d) for d in range(ndim)]
        for d, name in enumerate(names):
            self._define(name, "int", None)
            self._nd_bound_code(sources[d])
        # The launch supplies at most three ids (get_global_id has three dimensions). The
        # name is addressed by the loop's position, not by its domain: `param` may itself be
        # an expression (`(self)->image_height`, for a receiver member), which is no C
        # identifier.
        gid:str = "_work_position"
        self._emit(f"const int {gid} = {self._flattened_launch_id(param)};")
        checks:List[str] = []
        for d, name in enumerate(names):
            # The stride to the next-faster loop axis is the size of the axes after this one,
            # so the slowest axis divides by every later extent and the fastest takes the
            # remainder of its own. That makes the decomposition the exact inverse of the
            # row-major flattening `_nd_flat_index` emits for a subscript.
            divisor:str = gid
            for inner in range(d + 1, ndim):
                divisor = f"{divisor} / {self._nd_bound_code(sources[inner])}"
            if d:
                divisor = f"{divisor} % {self._nd_bound_code(sources[d])}"
            self._emit(f"const int {name} = {divisor};")
            checks.append(f"{name} >= {self._nd_bound_code(sources[d])}")
        self._emit(f"if ({' || '.join(checks)})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()


    def _flattened_launch_id(self, param:str)->str:
        """One flat work-item position, folded from whichever launch axes are present.

        `get_global_id(d)` reports a number in `[0, size(d))`, and a grid axis of one item
        reports 0, so folding every axis by its own grid extent and the strides of the axes
        between them yields the same position for a 1-D grid and for an N-D one of the same
        total. The grid extents are launch-time values the backend already exposes, so the
        generated source does not have to know the rank.

        Axis 0 is the fastest one, which is the order `_nd_flat_index` flattens the loop's own
        names in reverse -- so the loop's last name lands on gid 0.
        """
        if self._opencl_syntax:
            sizes:List[str] = [f"get_global_size({d})" for d in range(3)]
            ids:List[str] = [f"get_global_id({d})" for d in range(3)]
        else:
            sizes = ["(int)(gridDim.x * blockDim.x)", "(int)(gridDim.y * blockDim.y)",
                     "(int)(gridDim.z * blockDim.z)"]
            ids = [self._launch_id_code(axis) for axis in range(3)]
        # ((z % nz) * ny + y % ny) * nx + x % nx: one expression, so a rank-1 grid folds to
        # its own gid. Every group is parenthesised explicitly -- `%` and `*` share a
        # precedence level in C and associate left, so `a % b * c` is `(a % b) * c`, but
        # leaving the operands bare would also silently regroup the outer sum.
        return (f"((({ids[2]} % {sizes[2]}) * {sizes[1]} + ({ids[1]} % {sizes[1]}))"
                f" * {sizes[0]} + ({ids[0]} % {sizes[0]}))")


    def _nd_flat_index(self, names:List[str], param:str)->str:
        """Row-major flat index expression: n0*s1*s2 + n1*s2 + n2."""
        terms:List[str] = []
        ndim:int = len(names)
        for d, name in enumerate(names):
            term:str = name
            for inner in range(d + 1, ndim):
                term = f"{term} * {self._shape_var(param, inner)}"
            terms.append(term)
        return " + ".join(terms)


    def _enumerate_magic_target(self, iter_node:ast.expr)->Optional[Translated]:
        """Resolve the array of `for i, x in enumerate(arr)`, returning (name, pointer type).

        The domain may also be a **literal list** (`enumerate([1.0, 2.0, 3.0])`): numpy
        accepts it, its length is a compile-time number, and the elements are constants the
        work item can read directly. That form has no array parameter, so it is reported
        with the length the list states and no pointer type (`None`), which is what the
        caller uses to emit a constant table.
        """
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "enumerate" and len(iter_node.args) == 1 and not iter_node.keywords):
            return None
        arg:ast.expr = iter_node.args[0]
        if isinstance(arg, (ast.List, ast.Tuple)) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                and not isinstance(e.value, bool) for e in arg.elts):
            self._enumerate_constants = [e.value for e in arg.elts]
            return "", None
        if isinstance(arg, ast.Name) and arg.id in self._constant_lists:
            # `weights = [1.0, 2.0]`, then `enumerate(weights)`: the list is the table.
            self._enumerate_constants = list(self._constant_lists[arg.id])
            return "", None
        self._enumerate_constants = []
        param:str = ""
        if isinstance(arg, ast.Name):
            # A local alias of an array (`src = a`, then `ndenumerate(src)`) names the same
            # buffer, so its extents are the aliased array's own.
            param = self._array_pointer_root(arg.id)
        elif (self.self_param and isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
                and arg.value.id == self.self_param):
            param = f"_self_{arg.attr}"
        else:
            return None
        t:DeviceType = self._lookup(param)
        if not is_ptr(t):
            return None
        return param, t


    def _collect_constant_lists(self, body:List[ast.stmt])->None:
        """Record `weights = [1.0, 2.0, ...]` when a loop enumerates it.

        A literal list is a compile-time table: its length is the dispatch bound and each
        element is a constant, so the loop reads them from the source itself (see
        `_emit_constant_enumerate_head`) and the binding emits nothing. Only a list an
        `enumerate` names is treated that way -- any other list stays an ordinary value.
        """
        enumerated:Set[str] = set()
        for statement in body:
            for node in ast.walk(statement):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "enumerate" and len(node.args) == 1
                        and isinstance(node.args[0], ast.Name)):
                    enumerated.add(node.args[0].id)
        for statement in body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                continue
            if statement.targets[0].id not in enumerated:
                continue
            value:ast.expr = statement.value
            if isinstance(value, (ast.List, ast.Tuple)) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                    and not isinstance(e.value, bool) for e in value.elts):
                self._constant_list_names.add(statement.targets[0].id)
                self._constant_lists[statement.targets[0].id] = [
                    e.value for e in value.elts]


    def _emit_constant_enumerate_head(self, index_name:str, value_name:str,
                                      loop:ast.For)->None:
        """The work-item head for `for i, x in enumerate([1.0, 2.0, ...])`.

        The list is a compile-time table: its length is the dispatch's bound and each
        element is a constant, so the value is a `switch` over the index rather than a
        buffer the host would have to build and pass. The bound is a literal, so the guard
        is too.
        """
        values:List[object] = self._enumerate_constants
        extent:int = len(values)
        self.work_item_indices = [index_name]
        self._define(index_name, "int", None)
        if self._opencl_syntax:
            self._emit(f"const int {index_name} = get_global_id(0);")
        else:
            self._emit(f"const int {index_name} = "
                       f"(int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({index_name} >= {extent})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()
        # The element's type is the one its constants promote to (int beside a float is a
        # `double` under the mapping, python's own arithmetic).
        element:str = ("double" if any(isinstance(v, float) for v in values) else "int")
        self._define(value_name, element, loop.target)
        self._emit(f"{self._device_type(element)} {value_name} = 0;")
        self._emit(f"switch ({index_name})")
        self._emit("{")
        for position, value in enumerate(values):
            literal:str = (self._real_literal(float(value), element)
                           if element == "double" else str(int(value)))
            self._emit(f"    case {position}: {value_name} = {literal}; break;")
        self._emit("}")
        self._emit_blank()


    def _emit_work_item_head(self, index_name:str, param:str)->None:
        """Emit the gid definition and the out-of-range guard."""
        self.work_item_indices = [index_name]
        self._define(index_name, "int", None)
        gid_bound:str = self._shape_var(param, 0)
        if self._opencl_syntax:
            self._emit(f"const int {index_name} = get_global_id(0);")
        else:
            self._emit(f"const int {index_name} = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({index_name} >= {gid_bound})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()


    def _shape_holder_param(self, attr_node:ast.expr)->Optional[str]:
        """When X in `X.shape` is an array parameter, return its parameter name.

        A local alias of an array (`src = a`) resolves to the array it names: the extent is
        the same one, and the loop over `src.shape[0]` is the same dispatch.
        """
        if not (isinstance(attr_node, ast.Attribute) and attr_node.attr == "shape"):
            return None
        base:ast.expr = attr_node.value
        if isinstance(base, ast.Name):
            return self._array_pointer_root(base.id)
        if (self.self_param and isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                and base.value.id == self.self_param):
            return f"_self_{base.attr}"
        return None


    def _shape0_range_param(self, iter_node:ast.expr)->Optional[str]:
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "range" and len(iter_node.args) == 1 and not iter_node.keywords):
            return None
        arg:ast.expr = iter_node.args[0]
        if not (isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant) and arg.slice.value == 0):
            return None
        return self._shape_holder_param(arg.value)


    def _is_shape0_range(self, iter_node:ast.expr)->bool:
        return (self._shape0_range_param(iter_node) is not None)
