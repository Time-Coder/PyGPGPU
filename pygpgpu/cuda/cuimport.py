import sys
import os
import importlib.util
from types import ModuleType
from typing import Optional

from .oop.program import Program
from .oop.build_options import BuildOptions
from .oop.devices import Devices


class CULoader:
    def __init__(self, fullname:str, cu_path:str, type_checked:bool, build_options:BuildOptions, generate_pyi:bool):
        self.fullname:str = fullname
        self.cu_path:str = cu_path
        self.type_checked:bool = type_checked
        self.build_options:BuildOptions = build_options
        self.generate_pyi:bool = generate_pyi

    def create_module(self, spec):
        return None

    def exec_module(self, module:ModuleType):
        Devices[0].default_context.make_current()
        program = Program(self.cu_path, type_checked=self.type_checked, options=self.build_options)
        for struct_name, struct_type in program.structs.items():
            setattr(module, struct_name, struct_type)

        for kernel_name, kernel in program.kernels.items():
            setattr(module, kernel_name, kernel)

        if self.generate_pyi:
            program._nvrtc_program._program_parser.generate_pyi()


class CUFinder:

    def __init__(self):
        self.build_options:BuildOptions = BuildOptions()
        self.type_checked:bool = False
        self.generate_pyi:bool = True

    def find_spec(self, fullname:str, path, target=None):
        cu_path = self.get_cu_path(fullname)
        if not cu_path:
            return None

        loader = CULoader(fullname, cu_path, self.type_checked, self.build_options, self.generate_pyi)
        spec = importlib.util.spec_from_file_location(fullname, cu_path, loader=loader)
        return spec

    @staticmethod
    def get_cu_path(fullname:str):
        parts = fullname.split('.')
        for path in sys.path:
            file_path = path
            for part in parts:
                file_path += f"/{part}"

            file_path += ".cu"
            if os.path.isfile(file_path):
                return file_path

        return None


class cuimport:

    @staticmethod
    def install():
        for finder in sys.meta_path:
            if isinstance(finder, CUFinder):
                return

        sys.meta_path.insert(0, CUFinder())

    @staticmethod
    def uninstall():
        for i in range(len(sys.meta_path) - 1, -1, -1):
            if isinstance(sys.meta_path[i], CUFinder):
                del sys.meta_path[i]

    @staticmethod
    def config(
        maxrregcount:Optional[int]=0,
        ftz:Optional[bool]=None,
        prec_sqrt:Optional[bool]=None,
        prec_div:Optional[bool]=None,
        fmad:Optional[bool]=None,
        use_fast_math:Optional[bool]=None,
        std:Optional[object]=None,
        w:Optional[bool]=None,
        options:Optional[BuildOptions]=None,
        type_checked:bool=False,
        generate_pyi:bool=True
    ):
        if options is None:
            options = BuildOptions(
                maxrregcount=maxrregcount,
                ftz=ftz,
                prec_sqrt=prec_sqrt,
                prec_div=prec_div,
                fmad=fmad,
                use_fast_math=use_fast_math,
                std=std,
                w=w
            )

        for finder in sys.meta_path:
            if isinstance(finder, CUFinder):
                finder.build_options = options
                finder.type_checked = type_checked
                finder.generate_pyi = generate_pyi
