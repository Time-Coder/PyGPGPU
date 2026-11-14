from __future__ import annotations
import ast
from typing import (List, Optional, Tuple)
from .nd_index import _NdIndexMixin
from .errors import (
    _unpack_count_message,
    _unpacking_error,
)
from .notation import (
    DeviceType,
    HostValueEntry,
    OuterVarEntry,
    Translated,
)
from .predicates import (
    is_image,
    is_ptr,
)
from .walk import (
    _assigned_name_of,
    _creation_target_of,
    _magic_loop_end,
    _magic_loop_index,
    _outside_loops,
)


class _MagicLoopMixin(_NdIndexMixin):

    def _translate_magic_loop(self, body:List[ast.stmt])->None:
        # The statements after the loop are python: they run once on the host, after the
        # launch, reading whatever the work items wrote (see PyKernel._run_host_region).
        # Recorded here so every dispatch shape treats the tail the same way.
        #
        # "After the loop" has to mean after the *dispatch* loop: a serial `for` above it
        # (`for j in range(lane * 4, lane * 4 + 4)`) is per-work-item code, and taking the
        # last `ast.For` of any kind as the boundary pushed everything between them -- a
        # `local_array_per_group` declaration included -- into the host tail, where a device-only
        # builtin has no work item and fails.
        self.body_tail = list(body[_magic_loop_end(body):])
        # `rows: int = out.shape[0]` above the loop: the local stands for the extent, and
        # an `np.ndindex(rows, ...)` bound resolves through it. Scanned up front, because
        # the loop *detection* below needs the table -- a bound held in a local is still
        # the work-item domain, and missing it made the loop read as a "second magic
        # loop" instead.
        for stmt in body[:_magic_loop_index(body)]:
            value:Optional[ast.expr] = _creation_target_of(stmt)
            name:Optional[str] = _assigned_name_of(stmt)
            if value is None or name is None:
                continue
            if isinstance(value, ast.Subscript) \
                    and isinstance(value.slice, ast.Constant) \
                    and isinstance(value.slice.value, int):
                holder:Optional[str] = self._shape_holder_param(value.value)
                if holder is not None:
                    holder_t:DeviceType = self._lookup(holder)
                    if is_ptr(holder_t) or is_image(holder_t):
                        self._shape_aliases[name] = (holder, int(value.slice.value))
        magic_index:Optional[int] = None
        enumerate_form:Optional[Translated] = None
        nd_form:Optional[Tuple[str, str, str]] = None
        alias_form:bool = False
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.For):
                continue
            if self._is_shape0_range(stmt.iter):
                magic_index = idx
                break
            # `n = img.shape[0]` above the loop, then `for p in range(n)`: the local is
            # the extent, recorded by the scan above, so this is the same dispatch the
            # literal spelling is. The emission path below resolves the bound through the
            # same table, which keeps the two spellings' launch sizes identical.
            if self._shape0_range_param_of_local(stmt.iter) is not None:
                magic_index = idx
                alias_form = True
                break
            enum_info:Optional[Translated] = self._enumerate_magic_target(stmt.iter)
            if enum_info is not None:
                magic_index = idx
                enumerate_form = enum_info
                break
            nd_info:Optional[Tuple[str, str, str]] = self._nd_magic_target(stmt.iter)
            if nd_info is not None:
                magic_index = idx
                nd_form = nd_info
                break

        if magic_index is None:
            # No magic loop: the user indexes via gpu.get_global_id and friends. The iteration
            # domain then has to come from somewhere: the decorator's `domain=`, or the bracket
            # at call time. Without either, the launch size cannot be derived (see
            # PyKernel._invoke).
            for stmt in body:
                self._visit_statement(stmt)
            return

        # Statements before the magic loop are translated as usual (e.g. local variable initialization).
        # The magic loop already fixes the iteration domain, so a `domain=` declaration is
        # redundant: PyKernel rejects a kernel that carries both before it reaches here.
        self.magic_loop_present = True
        self.magic_loop_line = int(getattr(body[magic_index], "lineno", 0) or 0)
        for stmt in _outside_loops(body):
            # A creation the device cannot perform is executed by the interpreter instead
            # and becomes a parameter; that is only possible here, above the loop, because
            # below it the statement would run once per work item.
            if self._host_created_assignment(stmt):
                continue
            self._visit_host_statement(stmt)

        if enumerate_form is not None:
            loop:ast.For = body[magic_index]
            param, arr_t = enumerate_form
            target = loop.target
            if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                    and all(isinstance(e, ast.Name) for e in target.elts)):
                self._error(
                    "the enumerate magic loop must unpack exactly two names: "
                    "`for i, x in enumerate(array)`", target)
            index_name:str = target.elts[0].id
            value_name:str = target.elts[1].id

            if arr_t is None:
                # A literal list: the domain is `len(list)` and the element is the constant
                # at that position. No array parameter is involved, so the dispatch reads the
                # length the list states.
                self.gid_param = ""
                self._emit_constant_enumerate_head(index_name, value_name, loop)
                self._visit_work_item_body(loop.body)
                # The domain is the list's own length: an empty array name marks a literal
                # extent (see the `name.startswith` mapping in PyKernel._compile).
                self.work_size_dims = [("", len(self._enumerate_constants))]
                return

            self.gid_param = param
            self._emit_work_item_head(index_name, param)
            # x is an element-value snapshot (matches python semantics: rebinding x does not write back to the array)
            elem_t:object = arr_t[1]
            self._define(value_name, elem_t, target.elts[1])
            self._emit(f"{elem_t} {value_name} = {param}[{index_name}];")

            self._visit_work_item_body(loop.body)
            self.work_size_dims = [(param, 0)]
            return

        if nd_form is not None:
            form, param, module_name = nd_form
            loop:ast.For = body[magic_index]
            target = loop.target

            # Target unpacking: ndindex -> (y, x); ndenumerate -> ((y, x), v)
            value_name:str = ""
            names_node:ast.expr = target
            if form == "ndenumerate":
                if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                        and isinstance(target.elts[1], ast.Name)):
                    self._error(
                        f"{module_name}.ndenumerate must unpack like "
                        f"`for (y, x), v in {module_name}.ndenumerate(array)`", target)
                names_node = target.elts[0]
                value_name = target.elts[1].id
            if not (isinstance(names_node, ast.Name)
                    or (isinstance(names_node, ast.Tuple)
                        and all(isinstance(e, ast.Name) for e in names_node.elts))):
                self._error(
                    f"{module_name}.{form} indices must be unpacked into plain names, "
                    f"e.g. `for y, x in {module_name}.ndindex(array.shape)`", target)
            # One name is the 1-D spelling (`for i in np.ndindex(arr.shape[0])`), which numpy
            # accepts; a tuple is the multi-dimensional one.
            names:List[str] = ([names_node.id] if isinstance(names_node, ast.Name)
                               else [e.id for e in names_node.elts])
            ndim:int = len(names)
            # A one-name domain (`for i in np.ndindex(arr.shape[0])`) is legal numpy and is
            # legal here: the head below handles any rank, and refusing this rank pushed a
            # plain 1-D body into the "not a magic loop" path, where the array created above
            # it became device code and the error named the wrong thing entirely.

            # One extent per unpacked name. A mismatch between the two is a *count*
            # mismatch, and python reports it as an unpacking ValueError raised by the
            # `for` statement itself -- so the device says the same thing, in the same
            # words, and the check below raises a ValueError rather than a translation
            # SyntaxError.
            # The whole-shape spelling takes its extents from the array, one per name; that
            # rank is known only at launch, because a pointer carries none.
            explicit:Optional[List[Tuple[str, int]]] = self._nd_loop_extents
            sources:List[Tuple[str, int]] = (explicit
                                             if explicit is not None
                                             else [(param, d) for d in range(ndim)])
            if explicit is not None and len(explicit) != ndim:
                # Positioned at the loop, so the message carries the file and line the
                # interpreter would blame (see _body_frame, which turns them into a frame).
                raise _unpacking_error(_unpack_count_message(ndim, len(explicit)),
                                       target, self.filename, self.source_text,
                                       self.first_line)
            if explicit is None:
                # `np.ndindex(arr.shape)` asks for the array's whole shape, but a pointer
                # carries no rank, so "whole" can only be checked at launch. Remember which
                # array it was: the launch compares its rank with `ndim` and refuses a loop
                # that names fewer dimensions than the array has.
                self.nd_whole_shape_param = param
            self._nd_loop_extents = sources

            self.gid_param = param
            self._emit_nd_head(names, param)
            if form == "ndenumerate":
                arr_t:DeviceType = self._lookup(param)
                elem_t:object = arr_t[1]
                self._define(value_name, elem_t, None)
                self._emit(f"{elem_t} {value_name} = {param}[{self._nd_flat_index(names, param)}];")
            if form == "ndenumerate" and isinstance(names_node, ast.Name):
                # `for index, v in ndenumerate(arr)` holds the index as a *tuple* in
                # python (`index == (i,)`, or `(i, j)` in 2-D), and a body may subscript
                # it (`index[0]`) before unpacking it -- numpy's own protocol. The name
                # below stands for that tuple: a constant ordinal reads the
                # decomposition the head already emitted, in python's order (slowest
                # first), which is the reverse of the head's fastest-last names.
                self._ndenumerate_index_names[names_node.id] = list(reversed(names))

            self._visit_work_item_body(loop.body)
            # The launch sizes dimensions fastest-first, which is the reverse of the order
            # the loop unpacks them: gid 0 is the last index. Getting this backwards used
            # to launch a 2x3 domain as 3x2, so the last column or row was never written.
            self.work_size_dims = list(reversed(sources))
            # The launch sizes dimensions fastest-first, which is the reverse of the loop's
            # unpack order -- the member names read by the host follow the same reversal, or
            # `image_height` would be read for the fastest axis.
            self.work_size_members = list(reversed(self.work_size_members))
            return

        loop:ast.For = body[magic_index]
        range_call:ast.Call = loop.iter
        bound:ast.expr = range_call.args[0]
        if alias_form:
            # The bound is a local the scan registered as an extent alias: resolve it to
            # the extent expression it was assigned from (`<array>.shape[0]`), so the
            # emission below runs exactly the literal spelling's path. Nothing about the
            # local is consumed: it still travels as a plain int and stays readable in the
            # body, like any local.
            resolved:Optional[ast.expr] = self._alias_extent_expr(bound)
            if resolved is None:
                self._error(
                    "a magic loop's iteration domain must be `<array>.shape[0]`, directly "
                    "or through a local it was assigned to", bound)
            bound = resolved

        # Two names, because the two questions they answer have different answers for a
        # slice. `guard_name` is what the per-item bound is read from, which is the local
        # itself (`part.shape[0]` is the slice's length); `size_name` is the parameter the
        # *launch* is sized from, which is the array the local points at, because extents
        # are launch parameters named after arrays. Launching the array's full extent and
        # skipping the excess with the guard is the same arrangement the automatic work
        # size already uses when it rounds a grid up.
        raw_name:Optional[str] = self._loop_domain_name(bound.value)
        if raw_name is None:
            self._error(
                "a magic loop's iteration domain must be `<array>.shape[0]`", bound)
        guard_name:str = (raw_name if raw_name in self._array_slice_extents
                          else self._array_pointer_root(raw_name))
        size_name:str = self._array_pointer_root(raw_name)
        outer:Optional[OuterVarEntry] = self.outer_vars.get(size_name)
        host:Optional[HostValueEntry] = self.host_values.get(size_name)
        is_outer_array:bool = ((outer is not None and is_ptr(outer["type"]))
                               or (host is not None and is_ptr(host["type"])))
        if not is_outer_array and not is_ptr(self._lookup(guard_name)):
            self._error(f"'{raw_name}' is not an array parameter of this kernel", bound)

        self.gid_param = guard_name

        gid_bound:str = self._shape_var(guard_name, 0)
        target:str = loop.target.id
        # This branch emits its own work-item head (the 1-D form), so it has to record the
        # index variable that makes a `arr[i] += v` provably race-free just as the
        # multi-dimensional head does.
        self.work_item_indices = [target]
        self._define(target, "int", loop.target)
        if self._opencl_syntax:
            self._emit(f"const int {target} = get_global_id(0);")
        else:
            self._emit(f"const int {target} = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
        self._emit(f"if ({target} >= {gid_bound})")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")
        self._emit_blank()

        # The loop body is the work-item body; nested range loops translate as real serial loops
        self._visit_work_item_body(loop.body)
        self.work_size_dims = [(size_name, 0)]


    def _loop_domain_name(self, attr_node:ast.expr)->Optional[str]:
        """The name `X` in `X.shape[0]`, exactly as the body wrote it.

        No alias or slice resolution happens here: the caller needs to know *which local*
        the bound was read from, because a slice's extent is its own length while its
        launch size is the array's.
        """
        if not (isinstance(attr_node, ast.Attribute) and attr_node.attr == "shape"):
            return None
        base:ast.expr = attr_node.value
        if isinstance(base, ast.Name):
            return base.id
        if (self.self_param and isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name) and base.value.id == self.self_param):
            return f"_self_{base.attr}"
        return None


    def _shape0_range_param_of_local(self, iter_node:ast.expr)->Optional[str]:
        """The parameter a local-alias bound (`range(rows)`) sizes the dispatch from.

        The alias table is the one source of the answer: it holds the extent a
        `name = <array>.shape[d]` assignment bound (the scan at the top of
        `_translate_magic_loop` fills it, before loop detection, so it is populated for
        every caller). A name the table does not carry is not an extent alias -- a plain
        counter, a caller's scalar -- and answers None, which keeps such a loop the serial
        loop a work item runs itself.
        """
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "range" and len(iter_node.args) == 1 and not iter_node.keywords):
            return None
        arg:ast.expr = iter_node.args[0]
        if not isinstance(arg, ast.Name):
            return None
        binding:Optional[Tuple[str, int]] = self._shape_aliases.get(arg.id)
        if binding is None or binding[1] != 0:
            return None
        holder_t:DeviceType = self._lookup(binding[0])
        if not (is_ptr(holder_t) or is_image(holder_t)):
            return None
        return binding[0]


    def _alias_extent_expr(self, node:ast.expr)->Optional[ast.expr]:
        """The `<array>.shape[0]` expression a local-alias bound stands for.

        `for p in range(rows)` with `rows = img.shape[0]` above the loop: the dispatch tail
        needs the bound as the *expression* (`_loop_domain_name` reads `X` off it, and the
        slice/guard split reads the local's own extent), so the alias is re-expanded here
        rather than carried as a parameter name. A node that is not a name in the alias
        table answers None.
        """
        if not isinstance(node, ast.Name):
            return None
        binding:Optional[Tuple[str, int]] = self._shape_aliases.get(node.id)
        if binding is None:
            return None
        holder:str = binding[0]
        dimension:int = binding[1]
        holder_t:DeviceType = self._lookup(holder)
        if not (is_ptr(holder_t) or is_image(holder_t)):
            return None
        # Rebuild the extent expression the assignment bound: `<holder>.shape[<d>]`. For a
        # receiver attribute the holder is already the `_self_<attr>` parameter the
        # assignment resolved to, and the expression spelling the tail expects for it is
        # the same `<name>.shape[d]` form with that parameter's name.
        attribute:ast.expr = ast.Attribute(value=ast.Name(id=holder, ctx=ast.Load),
                                           attr="shape", ctx=ast.Load)
        return ast.Subscript(value=attribute,
                             slice=ast.Constant(value=dimension), ctx=ast.Load)
