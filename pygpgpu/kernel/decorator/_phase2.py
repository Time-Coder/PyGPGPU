from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
import numpy as np
from .slots import (
    _kernel_scalar_argument,
)
from .shape_check import (
    _check_shape_indices,
)
from .host_regions import (
    _created_arguments,
    _host_created_values,
    _host_scalar_values,
    _positional_argument_names,
    _receiver_name,
)
from ._typing import (
    OuterState,
)


class _BuildLaunchMixin:

    def _build_launch(self, plan:object, instance:object, args:Tuple[object, ...],
                    backend_sel:Optional[str], device_sel:Optional[str],
                    global_size:Optional[Tuple[int, ...]],
                    local_size:Optional[Tuple[int, ...]],
                    launch:str = "call")->object:
        """The launch's values: the caller's arguments, the carried slots, the
        outer-scope values and the created objects, in device-parameter order.

        The order is the device signature's: the receiver first (a method's first
        parameter), then the caller's arguments, then the ndarray attributes. The
        shape extents are read from this list by device-parameter index, so the
        runtime values are part of it but are appended to the launch afterwards.
        """
        plan.active_kernels:List[object] = list(plan.launch_meta.get("kernels") or [plan.kernel_obj])
        # The extent arguments each phase kernel declares, in phase order.
        plan.kernel_shapes:List[List[Tuple[int, int]]] = list(
            plan.launch_meta.get("kernel_shapes") or [[] for _ in plan.active_kernels])
        plan.kernel_created:List[List[str]] = list(
            plan.launch_meta.get("kernel_created") or [[] for _ in plan.active_kernels])
        # Each phase's parameters in order, so the launch can align values to the signature
        # instead of assuming the shared prefix: a phase may declare a carried name the
        # first kernel does not, or skip one it does.
        plan.phase_param_names:List[List[str]] = list(
            plan.launch_meta.get("phase_param_names") or [[] for _ in plan.active_kernels])
        # Whether each phase kernel declares the return claim/site words. A pass whose only
        # return runs in its host tail never claims, so its signature omits them and its
        # argument list has to omit them too (empty for a single-phase body, whose one
        # kernel is the main translator's own).
        plan.phase_claims:List[bool] = list(plan.launch_meta.get("phase_claims") or [])

        plan.outer_writable:List[Dict[str, object]] = []
        plan.carried_slots:Dict[str, object] = {}
        # The names `full` holds, in order: the caller's arguments, then the carried slots.
        # `build_phase_args` aligns each phase's signature against these names.
        # The `_self_<attr>` parameters sit in `full` right after the caller's arguments,
        # so the prefix names must include them or the aligned loop would look up an
        # attribute value in the wrong place.
        # When the receiver travels as a structure, it is the first entry of self_values
        # and the first device parameter: name it so the aligned loop finds the holder.
        plan.receiver_name:List[str] = ([self._self_param]
                                   if plan.self_struct_members else [])
        plan.self_attr_names:List[str] = [f"_self_{name}" for name in plan.self_attrs]
        # The launch is packed in the order the device signature declares: the receiver
        # first (it is a method's first parameter), then the caller's arguments, then the
        # ndarray attributes. `arg_types` and `self_types` already lead with the receiver
        # type, so the values have to lead with the receiver too.
        plan.receiver_values:List[object] = (list(plan.self_values[:1]) if plan.self_struct_members else [])
        plan.argument_values:List[object] = (list(plan.self_values[1:]) if plan.self_struct_members
                                     else list(plan.self_values))
        plan.prefix_names:List[str] = (plan.receiver_name
                                  + list(_positional_argument_names(self.raw_func))[:len(plan.args)]
                                  + plan.self_attr_names
                                  + list(plan.launch_meta.get("carried") or []))
        plan.full:List[object] = [_kernel_scalar_argument(value, t)
                          for value, t in zip(plan.receiver_values + list(plan.args) + plan.argument_values,
                                              plan.arg_types + plan.self_types)]
        # A carried scalar is one of the kernel's own parameters (declared before the shape
        # extents), so its slot joins full here rather than the trailing created values.
        # The slot is the carried value's own width (`total = 0.0` is a float): the
        # device parameter is a `float*` there, and an int32 host buffer under it
        # truncated every contribution before the kernel even ran.
        plan.carried_types:Dict[str, object] = plan.launch_meta.get("carried_types") or {}
        for name in plan.launch_meta.get("carried") or []:
            plan.slot_t:object = plan.carried_types.get(name, "int")
            plan.dtype:np.dtype = np.dtype(np.float32 if plan.slot_t == "float" else np.int32)
            plan.slot = np.zeros(1, dtype=plan.dtype)
            plan.slot[0] = plan.dtype.type(_host_scalar_values(
                self.raw_func, [name], tuple(plan.args), instance).get(name, 0))
            plan.full.append(plan.slot)
            plan.carried_slots[name] = plan.slot
        plan.own_count:int = len(plan.full)

        # Outer-scope values are read before the launch arguments are built: a shape
        # source can name an outer array, and `full` is indexed by the device parameter
        # order (see _build_launch_args), so the outer values have to be part of it.
        plan.outer_state:OuterState = self._bind_outer_values(plan.outer_vars, plan.backend)
        # Objects the kernel body creates are built here, on every call, exactly as the
        # interpreter would at that point in the body. They travel as trailing parameters.
        # A method's creation may read `self` (`np.zeros((self.height, ...))`), so the
        # instance is bound too.
        plan.created:Dict[str, object] = _host_created_values(
            self.raw_func, tuple(plan.args), instance,
            required_created=list(plan.launch_meta.get("kernel_created_all") or []))[0]
        # A statement between two magic loops runs on the host, so its value is computed here
        # and travels to the pass that reads it. A scalar travels as a one-element slot,
        # exactly like a carried scalar; a creation the pass declares travels as the object
        # itself, exactly like any other created array.
        # One shared namespace per gap region: a later region reads what an earlier one
        # assigned, the way consecutive statements of a function body do.
        plan.gap_namespaces:List[Tuple[Dict[str, object], Dict[str, object]]] = []
        for _ in (plan.launch_meta.get("phase_gaps") or []):
            plan.region_globals:Dict[str, object] = dict(self.raw_func.__globals__)
            plan.region_locals:Dict[str, object] = {}
            for name, value in zip(_positional_argument_names(self.raw_func), plan.args):
                plan.region_locals[name] = value
            plan.receiver:str = _receiver_name(self.raw_func)
            if plan.receiver and instance is not None:
                plan.region_locals[plan.receiver] = instance
            plan.gap_namespaces.append((plan.region_globals, plan.region_locals))
        plan.gap_names:List[str] = list(plan.launch_meta.get("gap_names") or [])
        plan.created_all:List[str] = list(plan.launch_meta.get("kernel_created_all") or [])
        # A gap value the interpreter bound between the passes travels as a host scalar of
        # its own type (see _bind_host_statement), not through the carried `int` slot below:
        # `shift = len(label)` holds a number, but the statement it came from is host code
        # and its value may be anything a python name can hold.
        plan.host_scalar_names:Set[str] = set(plan.launch_meta.get("host_values") or [])
        plan.gap_slots:Dict[str, object] = {}
        # A kernel argument is not a gap slot: assigning it in a host region updates the
        # function-local parameter value the later passes receive (see _run_host_region),
        # and a slot for it would shadow that update in the argument alignment. An outer
        # name is not one either: its own writable slot is what the device reads and what
        # the region's write-back fills, so a second slot would keep the launch's value.
        plan.argument_names:List[str] = list(
            _positional_argument_names(self.raw_func))[:len(plan.args)]
        for name, value in _host_scalar_values(
                self.raw_func, [n for n in plan.gap_names
                                if n not in plan.created_all
                                and n not in plan.host_scalar_names
                                and n not in plan.argument_names
                                and n not in plan.outer_vars],
                tuple(plan.args), instance).items():
            plan.slot = np.zeros(1, dtype=np.int32)
            try:
                plan.slot[0] = int(value)
            except (TypeError, ValueError):
                # Not a number: it travels as a host scalar of its own type instead (see
                # _bind_host_statement), and this slot is not its carrier.
                continue
            plan.gap_slots[name] = plan.slot
        # The created objects the device signature declares: the aligned prefix picks the
        # real array from here rather than a scalar slot.
        plan.created_objects:Dict[str, object] = {name: plan.created[name] for name in plan.created_all
                                          if name in plan.created}

        plan.outer_writable = list(plan.outer_state["writable"])
        # The plain values host-region statements bound, in device-parameter order: the
        # translator appended them after the created objects (see _bind_host_statement), so
        # the launch supplies them in that same order.
        plan.host_value_names:List[str] = [name for name in plan.launch_meta.get("host_values") or []
                                      if name not in plan.created_all]
        # Their actual values, recomputed on *this* call: the compile that recorded the
        # signature happens once, but a host-region value can depend on the arguments
        # (`shift = len(label)`), so the launch reads the snapshot the call just made.
        plan.host_region_values:Dict[str, object] = dict(
            getattr(self, "host_region_snapshot", None)
            or plan.launch_meta.get("host_region_values") or {})
        plan.runtime_values:List[object] = list(plan.outer_state["full"]) + _created_arguments(plan.created)
        plan.return_state:Dict[str, object] = (self._alloc_return_slots(plan.returns)
                                      if plan.returns else {})
        # Whether this launch's first kernel declares the return claim/site words. A
        # single-phase body has one kernel -- the main translator's -- and this is its own
        # answer; with several phases each entry of `phase_claims` decides instead. The
        # value is recorded at compile time because the translator is not reachable here.
        plan.main_claims:bool = bool(plan.launch_meta.get("main_claims", True))
        # Launch order must match the device signature: own arguments, shape extents, then
        # the values the host supplies (outer scope, then created objects). The shape
        # extents are read from `full` by device-parameter index, so the runtime values are
        # part of `full` but are appended to the launch afterwards.
        plan.full.extend(plan.runtime_values)
        # `full` indexes like the device parameter list, which is what shape_sources were
        # recorded against. Validate the dimensions first: an index past an array's rank
        # used to raise a bare `IndexError: tuple index out of range`, which says nothing
        # about which array or loop was wrong.
        _check_shape_indices(self.__name__, plan.full, plan.shape_sources, plan.work_size_dims, plan.launch_meta)


        # The read-back of a previous launch is a stream callback, and the caller may have
        # written the host arrays since. The upload below reads those arrays' memory, so it
        # must see the callbacks' result: wait once, before anything is uploaded.
