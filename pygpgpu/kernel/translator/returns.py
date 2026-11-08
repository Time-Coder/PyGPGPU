from __future__ import annotations
import ast
from typing import (List, Optional, Tuple)
import numpy as np
from .notation import (
    PTR,
    ReturnLayoutEntry,
    ReturnPosition,
    ReturnSiteLayout,
    ReturnSitePosition,
    Translated,
)
from .vectors import (
    is_vec,
)
from .predicates import (
    is_image,
    is_ptr,
    promote,
)


class _ReturnsMixin:

    def _return_position(self, node:ast.expr)->ReturnPosition:
        """Classify one returned expression as an array position or a value position."""
        if isinstance(node, ast.Subscript):
            # A slice/element of an array is not an array: it would be a single element,
            # which a kernel cannot hand back as a view (the device sees a flat pointer).
            if isinstance(node.slice, ast.Slice):
                self._error(
                    "a kernel cannot return an array slice: return the whole array "
                    "parameter by name, or return the element as a value", node)
        code, t = self._visit_expression(node)
        if is_image(t):
            self._error("cannot return an image: image parameters are opaque handles", node)
        if is_ptr(t):
            # An array travels as its own parameter: the host already holds the object, so
            # there is nothing to copy and nothing to allocate. Only a name identifies a
            # parameter, so an expression is refused rather than silently copied. A
            # `self.<attr>` array counts as a name: it is a parameter too (`_self_<attr>`).
            if not isinstance(node, (ast.Name, ast.Attribute)):
                self._error(
                    "a kernel can return an array parameter, but it must name it "
                    "(e.g. `return out`), not an expression", node)
            name:str = self._array_name(node)
            if name not in {param_name for param_name, _ in self.params}:
                # A host-created array is a parameter too, just not a python argument.
                if name not in self.host_values:
                    self._error(f"cannot return the array '{name}': it is not a parameter", node)
            # A carried slot is a pointer to one value, so returning it hands back that value
            # -- not an array. The parameter already holds it, so no slot is allocated.
            if name in self.carried_scalars:
                return {"kind": "value", "type": t[1], "code": f"{name}[0]",
                        "slot_parameter": name}
            if name in self.promoted_parameters:
                # Promoted for the return's sake: it is a *value*, and the parameter it now
                # travels as is the slot it lands in -- no extra argument.
                self.promoted_returns.add(name)
                return {"kind": "value", "type": t[1], "code": f"{name}[0]",
                        "slot_parameter": name}
            return {"kind": "array", "name": name, "type": t}
        if isinstance(node, ast.Name) and (
                node.id in self.carried_scalars or node.id in self.promoted_parameters):
            # Both are one location the kernel already carries: the parameter *is* the slot.
            return {"kind": "value", "type": t, "code": code,
                    "slot_parameter": node.id}
        return {"kind": "value", "type": t, "code": code}


    def _return_site(self, node:ast.Return)->List[ReturnPosition]:
        """The shape of one `return` statement, regardless of what other sites return."""
        if node.value is None:
            return []
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            # `return None` is the same as a bare `return` in python.
            return []
        if isinstance(node.value, ast.Tuple):
            for item in node.value.elts:
                if isinstance(item, ast.Tuple):
                    self._error(
                        "a returned tuple position must be a single value, not a nested tuple", item)
            return [self._return_position(item) for item in node.value.elts]
        return [self._return_position(node.value)]


    def _promote_returned_parameters(self, node:ast.Return)->None:
        """Turn a by-value parameter the body returns into a pointer parameter.

        A scalar argument travels by value, so the device only holds a copy and there is
        nothing for a return slot to point at -- the kernel would allocate a second
        parameter just to carry a number the caller already passed. Passing such an
        argument by pointer instead gives the return slot the parameter itself: one
        location, no extra argument, and the same value on both sides.
        """
        if node.value is None:
            return
        values:List[ast.expr] = (list(node.value.elts) if isinstance(node.value, ast.Tuple)
                                else [node.value])
        for value in values:
            if not isinstance(value, ast.Name):
                continue
            for index, (name, t) in enumerate(self.params):
                if name != value.id or is_ptr(t) or is_image(t):
                    continue
                # A writable carried slot is already a pointer; this only reaches an ordinary
                # scalar argument (`scale: int`).
                self.params[index] = (name, (PTR, t))
                # The scope has to follow, or a later lookup still reports the by-value type
                # and the layout allocates a slot for a value that already has one.
                for scope in self.scopes:
                    if name in scope:
                        scope[name] = (PTR, t)
                self.promoted_parameters.add(name)


    def _return_is_shared(self, node:ast.Return)->bool:
        """True when every name a post-loop return reads is a launch-wide slot.

        A carried scalar (written inside the loop, read after it) is one location for the
        whole launch, so a work item's copy of the expression is the call's result -- there
        is nothing per-item left for the claim to arbitrate. A kernel argument is shared for
        the same reason. Anything else (a local assigned below the loop) is one work item's
        value, which is what the refusal is about.
        """
        if node.value is None:
            return True
        values:List[ast.expr] = (list(node.value.elts) if isinstance(node.value, ast.Tuple)
                                else [node.value])
        for value in values:
            for name in ast.walk(value):
                if not isinstance(name, ast.Name) or not isinstance(name.ctx, ast.Load):
                    continue
                if name.id in self.carried_scalars:
                    continue
                # A kernel argument is launch-wide too: every work item reads the same value,
                # so returning one answers the same thing everywhere. A pointer argument (an
                # array) was already allowed by the array-return rule.
                if name.id in dict(self.params) or is_ptr(self._lookup(name.id)):
                    continue
                return False
        return True


    def _visit_kernel_return(self, node:ast.Return)->None:
        """Translate a `return` in a kernel body into a claim plus slot stores."""
        # Promotion comes first: it decides the parameter's type, and the layout below reads
        # that type to tell a reused parameter from a value needing a slot of its own.
        if self.is_kernel:
            self._promote_returned_parameters(node)
        positions:List[ReturnPosition] = self._return_site(node)
        if self.after_magic_loop and any(position["kind"] == "value" for position in positions):
            # The statements after the loop run once per work item, so a value accumulated
            # there is that work item's -- unless it is read from a *carried* slot, which is
            # one location for the whole launch. `total = 0; for ...: total += x[i];
            # return total` therefore answers the same thing in every work item, and the
            # claim word hands back the one value the call computed.
            if not self._return_is_shared(node):
                self._error(
                    "a value-returning return cannot appear after the magic loop: the "
                    "statements after the loop run once per work item, so a value computed "
                    "there belongs to one work item, not to the whole call. Accumulate it in "
                    "a scalar above the loop with `+=` (one shared slot), return from inside "
                    "the loop, or reduce into an array argument and return that array", node)
        site:int = len(self.return_sites)
        self.return_sites.append(positions)
        self._record_return_slots(site, positions, node)

        if not self.has_returns:
            # No return site of this kernel returns a value at all (see
            # _returns_a_value): a plain return, and the result is None.
            self._emit("return;")
            return

        # Every site claims, including a bare `return`: in python the first work item to
        # leave the function decides the result, so a bare return must be able to win
        # with None just as a value site wins with its tuple.
        claim:str = f"{self.RETURN_CLAIM}[0]"
        if self._metal_syntax:
            # MSL's compare-and-swap takes the expected value by pointer and answers a
            # bool rather than the old value. So the claim is tried with a zeroed
            # expected local and then read back: the word is 1 only for the work item
            # whose exchange actually happened, which is what `claimed == 1` tests.
            self._emit("int _claim_expected = 0;")
            self._emit(f"atomic_compare_exchange_weak_explicit("
                       f"(device atomic_int*)(&{claim}), &_claim_expected, 1, "
                       f"memory_order_relaxed, memory_order_relaxed);")
            self._emit(f"const bool _claimed = ({claim} == 1);")
            guard = "_claimed"
        elif self._opencl_syntax:
            guard = f"atomic_cmpxchg((volatile __global int*)(&{claim}), 0, 1) == 0"
        else:
            guard = f"atomicCAS(&{claim}, 0, 1) == 0"
        self._emit(f"if ({guard})")
        self._emit("{")
        # The winning site records what it returned, so the host rebuilds that site's
        # shape rather than a shape inferred from every site at once.
        self._emit(f"    {self.RETURN_SITE}[0] = {site + 1};")
        for slot, position in self._site_slots(site):
            if position["kind"] != "value":
                continue
            if slot < 0:
                # The value already lives in the parameter it travels in: there is no slot to
                # store it into, and the host reads it from the argument it passed.
                continue
            stored:str = self._complex_store(self.return_layout[slot]["type"],
                                             position["type"], position["code"])
            self._emit(f"    {self._return_slot(slot)}[0] = {stored};")
        self._emit("}")
        self._emit("return;")


    def _return_slot(self, index:int)->str:
        return f"_ret{index}"


    def _record_return_slots(self, site:int, positions:List[ReturnPosition],
                             node:ast.Return)->None:
        """Give each value position of this site a slot, extending the shared layout.

        A slot is keyed by its C element type, so two sites returning a float share one
        buffer while a site returning an int gets a buffer of its own. A returned array
        occupies no slot: the host hands back the very object it passed in.
        """
        for ordinal, position in enumerate(positions):
            if position["kind"] != "value":
                continue
            if position.get("slot_parameter"):
                # The value already has a location of its own -- the parameter it travels
                # in -- so no slot is allocated for it: there is nothing to copy and the
                # host reads it from the argument it passed.
                self._site_slot_map.setdefault(site, []).append((ordinal, -1))
                continue
            ctype:str = self._return_param_type(position["type"])
            slot:Optional[int] = self._slot_for_type(ctype)
            if slot is None:
                slot = len(self.return_layout)
                self.return_layout.append({"ctype": ctype, "type": position["type"]})
            self._site_slot_map.setdefault(site, []).append((ordinal, slot))


    def _slot_for_type(self, ctype:str)->Optional[int]:
        for index, entry in enumerate(self.return_layout):
            if entry["ctype"] == ctype:
                return index
        return None


    def _site_slots(self, site:int)->List[Tuple[int, ReturnPosition]]:
        """(slot, position) pairs this site stores into."""
        positions:List[ReturnPosition] = self.return_sites[site]
        pairs:List[Tuple[int, ReturnPosition]] = []
        for ordinal, slot in self._site_slot_map.get(site, []):
            pairs.append((slot, positions[ordinal]))
        return pairs


    def _return_param_type(self, t:object)->str:
        """The C element type of a value slot."""
        from .. import structs

        if is_vec(t):
            return t
        if isinstance(t, str) and structs.dtype_for(t) is not None:
            # A structure value travels as a one-element buffer of that structure, so the
            # slot element is the structure type itself.
            return t
        slot:Optional[Translated] = self.RETURN_SLOT_TYPES.get(t)
        if slot is None:
            self._error(f"cannot return a value of type {t}")
        return self._device_type(slot[0])


    def return_slot_dtype(self, index:int)->np.dtype:
        """The numpy dtype the host uses for one value slot.

        A vector slot uses the *host* layout (a 3-component vector has three components);
        `pygpgpu.vec_types.layout.pack` re-lays it out for the backend that needs padding,
        the same way it does for vector arguments.
        """
        from ...vec_types.layout import _host_classes
        from .. import structs

        entry:ReturnLayoutEntry = (self.return_layout or [])[index]
        if is_vec(entry["type"]):
            return _host_classes()[entry["type"]].numpy_dtype()
        structure:object = structs.dtype_for(entry["type"]) if isinstance(entry["type"], str) else None
        if structure is not None:
            return structure
        return self.RETURN_SLOT_TYPES[entry["type"]][1]


    def render_return_params(self)->List[str]:
        """The extra device parameters a kernel with returns needs.

        A kernel that returns only an array still needs the claim and site words, so this
        is gated on `has_returns` rather than on the value slots alone.

        The claim and site words are withheld when this kernel contains **no device-side
        return at all**: a body whose only `return` sits after the magic loop hands its value
        back through the host tail (see PyKernel._run_host_region), so no work item ever
        reaches the claim, the word stays zero, and the two buffers would become a pair of
        wasted uploads. A phase of a multi-loop body is considered the same way and on its
        own: every phase is a separate kernel, and whether one of them claims is a fact
        about the statements it contains.
        """
        if not self.has_returns:
            return []
        qualifier:str = self._buffer_qualifier()
        params:List[str] = [f"{qualifier}{entry['ctype']}* {self._return_slot(index)}"
                            for index, entry in enumerate(self.return_layout)]
        # A movable claim/site pair, decided by whether anything on the device claims: a
        # tail-only return never reaches the translator, so it needs neither word.
        if not self._claims_on_device:
            return params
        params.append(f"{qualifier}int* {self.RETURN_CLAIM}")
        params.append(f"{qualifier}int* {self.RETURN_SITE}")
        return params


    def _buffer_qualifier(self)->str:
        """The address-space qualifier a buffer parameter carries, with its trailing space.

        One place decides it, because the three questions it answers are different:
        Metal spells the space `device`, OpenCL spells it `__global`, and CUDA/HIP
        spell nothing at all (a plain pointer is already in the global space).
        """
        if self._metal_syntax:
            return "device "
        if self._opencl_syntax:
            return "__global "
        return ""


    def return_site_layout(self)->List[ReturnSiteLayout]:
        """What the host needs to rebuild the result of each return site.

        Per site: the positions in return order, each "value" with its slot index or
        "array" with the parameter name to hand back, plus whether the site returned a
        tuple. A site with no positions is a bare `return`, i.e. None.
        """
        layout:List[ReturnSiteLayout] = []
        for site, positions in enumerate(self.return_sites):
            entries:List[ReturnSitePosition] = [
                {"kind": "value", "ordinal": ordinal, "slot": slot,
                 # A negative slot means the value already lives in the parameter it
                 # travelled in, so the host reads it from the launch arguments.
                 **({"reused": positions[ordinal]["slot_parameter"]}
                    if slot < 0 else {})}
                for ordinal, slot in self._site_slot_map.get(site, [])]
            for ordinal, position in enumerate(positions):
                if position["kind"] == "array":
                    entries.append({"kind": "array", "ordinal": ordinal, "name": position["name"]})
            entries.sort(key=lambda entry: entry["ordinal"])
            layout.append({"positions": entries,
                           "is_tuple": len(positions) > 1})
        return layout


    def _visit_return(self, node:ast.Return)->None:
        if self.is_kernel:
            self._visit_kernel_return(node)
            return

        if node.value is None:
            self.return_type = self.return_type or "void"
            self._emit("return;")
            return

        # `return a, b` from a device helper: C has no tuple, so the elements become
        # stores into the `T*` output parameters appended after the helper's own
        # signature (see `tuple_return_types` and `_emit_device_call`, which appends
        # them and passes the call site's destinations). `@kernel` supports a tuple
        # return through its own claim machinery; this is the helper-scale form of the
        # same feature, and the unpacking at the call site (`x, y = pair(...)`) is what
        # python writes for it.
        if isinstance(node.value, ast.Tuple):
            elements:List[Translated] = [self._visit_expression(e) for e in node.value.elts]
            for _code, t in elements:
                if is_ptr(t):
                    self._error("cannot return an array from a function", node)
            self.tuple_return_types = [t for _code, t in elements]
            for ordinal, (code, _t) in enumerate(elements):
                self._emit(f"*_tuple_out{ordinal} = {code};")
            self.return_type = "void"
            self._emit("return;")
            return

        code, t = self._visit_expression(node.value)
        if is_ptr(t):
            self._error("cannot return an array from a function", node)

        if self.return_type is None:
            self.return_type = t
        elif self.return_type != t:
            unified:object = promote(self.return_type, t)
            if unified != self.return_type:
                # The cast has to be spelled in this backend's own type names: CUDA C has
                # no `uchar`/`ushort`/`uint`/`ulong`, so a bare translator name here made
                # every device helper returning one of those fail to compile under NVRTC.
                code = f"(({self._device_type(self.return_type)})({code}))"
            self.return_type = unified
        self._emit(f"return {code};")
