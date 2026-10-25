import sys
import os
import importlib.util
from types import ModuleType
from typing import Optional

from ..utils import local_size_for_global, device_limits
from .oop.program import Program
from .oop.build_options import BuildOptions
from .oop.devices import Devices


class CUKernelWrapper:
    """Bracket-style wrapper for handwritten .cu kernels, matching the handwritten opencl kernel usage::

        k(a, out)                    # default device, automatic work size
        k["geforce"](a, out)         # device (index / name substring / "name:ordinal")
        k[grid, block](a, out)       # grid/block (a None block means automatic)
        k["geforce"][grid, block](a, out)   # either order
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
        """Total-work-item semantics (unified with opencl/@kernel): total = number of threads, local = size per group;
        the cuda gridDim (number of blocks) is derived from total/local."""
        if self._total is None:
            return kernel

        def norm(v):
            if v is None:
                return None
            return (v,) if isinstance(v, int) else tuple(int(x) for x in v)

        total = norm(self._total)
        local = norm(self._local)
        if local is None:
            device = (self._device if self._device is not None else Devices[0])
            max_group, max_items = device_limits(device)
            local = local_size_for_global(
                total, max_work_group_size=max_group, max_work_item_sizes=max_items)
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
        # Record device and sizes on the wrapper and convert them together at call time (so any chaining order keeps total-work-item semantics)
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
