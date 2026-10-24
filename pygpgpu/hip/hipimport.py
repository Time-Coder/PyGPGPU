import sys
import os
import importlib.util
from types import ModuleType
from typing import Optional

from ..utils import local_size_for_global
from .oop.program import Program
from .oop.build_options import BuildOptions
from .oop.devices import Devices


class HIPKernelWrapper:
    """手写 .hip kernel 的中括号包装(总数语义, 与 opencl/cuda 手写用法对齐)。

        k(args)                          # 默认设备, work size 自动
        k["gfx1036"](args)               # 设备(编号/名称子串/名:序号)
        k[total, local](args)            # local 可 None(自动)
        k["gfx1036"][total, local](args) # 顺序不限
    """

    def __init__(self, hip_path:str, name:str, kernel, compiler, type_checked:bool, options:BuildOptions):
        self._hip_path:str = hip_path
        self._name:str = name
        self._compiler = compiler
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
            program = Program(self._hip_path, type_checked=self._type_checked,
                              options=self._options, compiler=self._compiler)
            kernel = program.kernels[self._name]
            self._kernels[device] = kernel
        return kernel

    def _apply_sizes(self, kernel):
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
        return f"HIPKernelWrapper({self._name})"


class HIPLoader:
    def __init__(self, fullname:str, hip_path:str, type_checked:bool, build_options:BuildOptions, generate_pyi:bool):
        self.fullname:str = fullname
        self.hip_path:str = hip_path
        self.type_checked:bool = type_checked
        self.build_options:BuildOptions = build_options
        self.generate_pyi:bool = generate_pyi

    def create_module(self, spec):
        return None

    def exec_module(self, module:ModuleType):
        from .oop.hip_compiler import HipCompiler
        Devices[0].default_context.make_current()
        compiler = HipCompiler(self.hip_path, options=self.build_options)
        program = Program(self.hip_path, type_checked=self.type_checked,
                          options=self.build_options, compiler=compiler)
        parser = compiler.parser
        for struct_name, struct_type in program.structs.items():
            setattr(module, struct_name, struct_type)

        for kernel_name, kernel in program.kernels.items():
            setattr(module, kernel_name, HIPKernelWrapper(
                self.hip_path, kernel_name, kernel,
                compiler, self.type_checked, self.build_options))

        if self.generate_pyi:
            parser.generate_pyi()


class HIPFinder:

    def __init__(self):
        self.build_options:BuildOptions = BuildOptions()
        self.type_checked:bool = False
        self.generate_pyi:bool = True

    def find_spec(self, fullname:str, path, target=None):
        hip_path = self.get_hip_path(fullname)
        if not hip_path:
            return None

        loader = HIPLoader(fullname, hip_path, self.type_checked, self.build_options, self.generate_pyi)
        spec = importlib.util.spec_from_file_location(fullname, hip_path, loader=loader)
        return spec

    @staticmethod
    def get_hip_path(fullname:str):
        parts = fullname.split('.')
        for path in sys.path:
            file_path = path
            for part in parts:
                file_path += f"/{part}"

            file_path += ".hip"
            if os.path.isfile(file_path):
                return file_path

        return None


class hipimport:

    @staticmethod
    def install():
        for finder in sys.meta_path:
            if isinstance(finder, HIPFinder):
                return

        sys.meta_path.insert(0, HIPFinder())

    @staticmethod
    def uninstall():
        for i in range(len(sys.meta_path) - 1, -1, -1):
            if isinstance(sys.meta_path[i], HIPFinder):
                del sys.meta_path[i]

    @staticmethod
    def config(type_checked:bool=False, options:Optional[BuildOptions]=None, generate_pyi:bool=True):
        for finder in sys.meta_path:
            if isinstance(finder, HIPFinder):
                if options is not None:
                    finder.build_options = options
                finder.type_checked = type_checked
                finder.generate_pyi = generate_pyi
