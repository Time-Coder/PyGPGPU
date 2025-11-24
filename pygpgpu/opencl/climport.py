import sys
import os
import importlib.util
from typing import Optional

from .common import compile
from .oop.build_options import BuildOptions


class CLLoader:
    def __init__(self, fullname:str, cl_path:str, type_checked:bool, build_options:BuildOptions):
        self.fullname:str = fullname
        self.cl_path:str = cl_path
        self.type_checked:bool = type_checked
        self.build_options:BuildOptions = build_options

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        program_wrapper = compile(self.cl_path, type_checked=self.type_checked, options=self.build_options, generate_pyi=True)
        for struct_info in program_wrapper._kernel_parser._struct_infos.values():
            setattr(module, struct_info.name, struct_info.struct_type)

        for kernel_name, kernel_wrapper in program_wrapper.kernels_wrappers.items():
            setattr(module, kernel_name, kernel_wrapper)


class CLFinder:

    def __init__(self):
        self.build_options:BuildOptions = BuildOptions()
        self.type_checked:bool = False

    def find_spec(self, fullname:str, path, target=None):
        cl_path = self.get_cl_path(fullname)
        if not cl_path:
            return None

        loader = CLLoader(fullname, cl_path, self.type_checked, self.build_options)
        spec = importlib.util.spec_from_file_location(fullname, cl_path, loader=loader)
        return spec

    @staticmethod        
    def get_cl_path(fullname:str):
        parts = fullname.split('.')
        for path in sys.path:
            file_path = path
            for part in parts:
                file_path += f"/{part}"

            file_path += ".cl"
            if os.path.isfile(file_path):
                return file_path
            
        return None


class climport:

    @staticmethod
    def install():
        for finder in sys.meta_path:
            if isinstance(finder, CLFinder):
                return
            
        sys.meta_path.insert(0, CLFinder())

    @staticmethod
    def uninstall():
        for i in range(len(sys.meta_path)-1, -1, -1):
            if isinstance(sys.meta_path[i], CLFinder):
                del sys.meta_path[i]

    @staticmethod
    def config(
        single_precision_constant:bool=False,
        denorms_are_zero:bool=False,
        fp32_correctly_rounded_divide_sqrt:bool=False,
        opt_disable:bool=False,
        strict_aliasing:bool=False,
        uniform_work_group_size:bool=False,
        no_subgroup_ifp:bool=False,
        mad_enable:bool=False,
        no_signed_zeros:bool=False,
        unsafe_math_optimizations:bool=False,
        finite_math_only:bool=False,
        fast_relaxed_math:bool=False,
        w:bool=False,
        Werror:bool=False,
        cl_std:Optional[float]=None,
        kernel_arg_info:bool=False,
        g:bool=False,
        create_library:bool=False,
        enable_link_options:bool=False,
        x_spir:bool=False,
        spir_std:Optional[float]=None,
        type_checked:bool=False,
        options:Optional[BuildOptions] = None
    ):
        if options is None:
            options = BuildOptions(
                single_precision_constant,
                denorms_are_zero,
                fp32_correctly_rounded_divide_sqrt,
                opt_disable,
                strict_aliasing,
                uniform_work_group_size,
                no_subgroup_ifp,
                mad_enable,
                no_signed_zeros,
                unsafe_math_optimizations,
                finite_math_only,
                fast_relaxed_math,
                w,
                Werror,
                cl_std,
                kernel_arg_info,
                g,
                create_library,
                enable_link_options,
                x_spir,
                spir_std
            )

        for finder in sys.meta_path:
            if isinstance(finder, CLFinder):
                finder.build_options = options
                finder.type_checked = type_checked