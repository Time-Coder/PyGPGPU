from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from .. import structs
from ..translator import (
    FunctionTranslator,
)
from .host_regions import (
    _host_created_values,
    _positional_argument_names,
)


class _CompileTranslateMixin:

    def _translate_body(self, plan:object, args:List[object], arg_types:List[object],
                    self_types:List[object], self_attrs:List[str],
                    owner:Optional[type], sig:str, backend:str, device_obj:object,
                    instance:object=None,
                    self_struct_members:Optional[Dict[str, str]]=None,
                    self_struct_type:object=None)->object:
        """The translator and the device code it renders.

        The translator is built over the caller's argument types and the body's own
        scan, and the rendered code is one piece per magic loop: a body with several
        loops renders several kernels, launched in order so the arrays stay on the
        device between them.
        """
        if backend in ("cuda", "hip"):
            for plan.t in list(arg_types) + list(self_types):
                plan.base:str = (plan.t[1] if isinstance(plan.t, tuple) else plan.t)
                if plan.base == "half":
                    raise SyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead") from None

        plan.c_name:str = self._c_kernel_name(sig, backend)
        plan.device_cache:Dict[object, object] = {}
        # self_types leads with the receiver struct when the receiver travels as one, so
        # the attribute map pairs the array attributes with the types after it.
        plan.self_attr_types:Dict[str, object] = dict(zip(
            self_attrs, self_types[1:] if self_struct_members else self_types))

        # hip accepts CUDA syntax (threadIdx/atomicAdd/make_float4 etc.), so codegen goes through cuda
        plan.codegen_backend:str = ("cuda" if backend == "hip" else backend)
        # Statements above the magic loop that only build data are executed by the
        # interpreter, not translated: numpy decides the shape and dtype, and the resulting
        # object is what the kernel receives as a parameter (see
        # FunctionTranslator._host_created_assignment). The values and the line numbers were
        # computed once for this call, before the body was scanned (see _invoke_inner).
        plan.creation_values, plan._ = _host_created_values(self.raw_func, args, instance)
        plan.executed_lines:List[int] = list(getattr(self, "host_region_statements", []))
        plan.region_values:Dict[str, object] = dict(
            getattr(self, "host_region_snapshot", None) or plan.creation_values)
        # The class each STRUCTURE argument was actually built from. One dtype can stand
        # for several classes, so the dtype alone cannot say whose method `p.total` is --
        # but the launch has the instance the caller handed over, and its type is the
        # answer. Keyed by parameter name, which is what the translator's expressions carry.
        # `structure_dtype_of` excludes the vector types, which are ctypes classes too.
        plan.struct_arg_classes:Dict[str, type] = {}
        plan.parameter_names:List[str] = _positional_argument_names(self.raw_func)
        for plan.parameter_name, plan.value in zip(plan.parameter_names, args):
            # A union is a `ctypes` record as well but not a `Structure` subclass, so the
            # class is recognized through the shared predicate rather than `ctypes.Structure`.
            if not structs.is_ctypes_record(plan.value):
                continue
            # structure_dtype_of excludes the vector types, which are ctypes classes too.
            if structs.structure_dtype_of(plan.value) is None:
                continue
            plan.struct_arg_classes[plan.parameter_name] = type(plan.value)
        plan.translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), plan.codegen_backend,
            self._resolve_global, plan.device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=plan.self_attr_types,
            self_owner=owner, launch_backend=backend,
            self_struct_members=self_struct_members,
            self_struct_type=self_struct_type,
            struct_arg_classes=plan.struct_arg_classes,
            host_creation_values=plan.region_values,
            host_executed_lines=plan.executed_lines,
            type_map=self._type_map)
        # `@kernel(domain="x")` names the parameter whose shape declares the iteration
        # domain. It is applied before `translate()` so the declaration is in place while the
        # magic loop is scanned -- a body cannot carry one of its own any more.
        if self._domain:
            self._apply_declared_domain(plan.translator)
        plan.translator.translate()
        # The name has to cover what the translator actually declares. `sig` is built from
        # the *argument types* the caller passed, which do not include the extent parameters
        # derived from the body, the outer-scope values or the shared buffers -- so a body
        # whose derived signature changed kept its old name, and the generated file ended up
        # holding two declarations of it ("more than one instance of overloaded function").
        # Re-deriving the name with the rendered declaration folds that into the same hash.
        plan.c_name = self._c_kernel_name(sig, backend,
                                     declaration=";".join(plan.translator.parameter_declarations()))
        # One kernel per magic loop: a body with several renders several, launched in order
        # so the arrays stay on the device between them.
        plan.rendered:List[Tuple[str, str]] = plan.translator.render_kernels(plan.c_name)

