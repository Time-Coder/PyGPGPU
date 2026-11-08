from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from .compile_plan import CompilePlan
from ._compile_translate import _CompileTranslateMixin
from ._compile_write import _CompileWriteMixin
from ._compile_kernels import _CompileKernelsMixin


class _PykernelCompileMixin(
        _CompileTranslateMixin, _CompileWriteMixin, _CompileKernelsMixin):


    def _compile(self, args:List[object], arg_types:List[object], self_types:List[object],
                 self_attrs:List[str], owner:Optional[type], sig:str,
                 backend:str, device_obj:object, instance:object=None,
                 self_struct_members:Optional[Dict[str, str]]=None,
                 self_struct_type:object=None)->Tuple[object, List[str]]:
        """One compilation, in three parts.

        The parts share their state through `plan`, a plain namespace: the first
        translates the body, the second writes the generated source out, and the third
        compiles it and describes the launch it produces.
        """
        plan:object = CompilePlan()
        plan.backend = backend
        self._translate_body(plan, args, arg_types, self_types, self_attrs,
                             owner, sig, backend, device_obj, instance,
                             self_struct_members, self_struct_type)
        self._write_source(plan, args, arg_types, self_types, self_attrs,
                           owner, sig, backend, device_obj, instance,
                           self_struct_members, self_struct_type)
        return self._compile_kernels(plan, args, arg_types, self_types, self_attrs,
                                    owner, sig, backend, device_obj, instance,
                                    self_struct_members, self_struct_type)
