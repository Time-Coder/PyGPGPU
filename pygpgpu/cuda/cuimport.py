import sys
import os
import importlib.util
from types import ModuleType
from typing import Optional

from ..utils import local_size_for_global
from .oop.program import Program
from .oop.build_options import BuildOptions
from .oop.devices import Devices


class CUKernelWrapper:
    """手写 .cu kernel 的中括号包装, 与 opencl 手写 kernel 用法对齐::

        k(a, out)                    # 默认设备, work size 自动
        k["geforce"](a, out)         # 设备(编号 / 名称子串 / "名:序号")
        k[grid, block](a, out)       # grid/block(block None 为自动)
        k["geforce"][grid, block](a, out)   # 顺序不限
    """

    def __init__(self, cu_path:str, name:str, kernel, nvrtc_program, type_checked:bool, options:BuildOptions):
        self._cu_path:str = cu_path
        self._name:str = name
        self._nvrtc_program = nvrtc_program
        self._type_checked:bool = type_checked
        self._options:BuildOptions = options
        self._kernels = {Devices[0]: kernel}
        self._device = None
        self._total = None
        self._local = None

    def _kernel_for(self, device):
        kernel = self._kernels.get(device)
        if kernel is None:
            device.default_context.make_current()
            program = Program(self._cu_path, type_checked=self._type_checked,
                              options=self._options, nvrtc_program=self._nvrtc_program)
            kernel = program.kernels[self._name]
            self._kernels[device] = kernel
        return kernel

    def _apply_sizes(self, kernel):
        """总数语义(与 opencl/@kernel 统一): total=总线程数, local=每组大小,
        cuda 的 gridDim(块数) 由 total/local 换算。"""
        if self._total is None:
            return kernel

        def norm(v):
            if v is None:
                return None
            return (v,) if isinstance(v, int) else tuple(int(x) for x in v)

        total = norm(self._total)
        local = norm(self._local)
        if local is None:
            local = local_size_for_global(total)
        if len(local) != len(total):
            raise ValueError(
                f"global {total} and local {local} work sizes must have the same dimension")
        for g, l in zip(total, local):
            if g % l != 0:
                raise ValueError(
                    f"global work size {total} must be divisible by local {local}; "
                    f"use None for an automatically derived local size")
        grid = tuple(g // l for g, l in zip(total, local))
        self._total = self._local = None
        return kernel[grid, local]

    def __getitem__(self, item):
        # 设备与尺寸都记在包装内, 调用时统一换算(保证链式任意顺序都是总数语义)
        if isinstance(item, str):
            self._device = Devices.device(item)
        elif isinstance(item, tuple) and len(item) == 2:
            self._total, self._local = item
        else:
            self._total, self._local = item, None
        return self

    def __call__(self, *args, **kwargs):
        device = (self._device if self._device is not None else Devices[0])
        kernel = self._apply_sizes(self._kernel_for(device))
        self._device = None
        return kernel(*args, **kwargs)

    def __getattr__(self, name:str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._kernels[Devices[0]], name)

    def __repr__(self)->str:
        return f"CUKernelWrapper({self._name})"


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
            setattr(module, kernel_name, CUKernelWrapper(
                self.cu_path, kernel_name, kernel,
                program._nvrtc_program, self.type_checked, self.build_options))

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
