from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .translator import (
    FunctionTranslator,
    PyKernelSyntaxError,
    infer_ctype,
    is_ptr,
    short_hash,
    type_sig,
)


class PyKernel:
    """`@kernel` 装饰器的实现:调用时把 python 函数翻译为后端源码并编译执行。"""

    def __init__(self, func:Callable, backend:str):
        self.raw_func:Callable = func
        self.backend:str = backend
        self.is_pykernel:bool = True
        self.__name__:str = func.__name__
        self.__doc__:str = func.__doc__ or ""
        self._cache:Dict[str, Tuple[Any, List[Tuple[int, int]]]] = {}  # sig key -> (kernel, shape sources)
        self._source_sig:str = self._source_signature()

    def __call__(self, *args:Any, global_size:Optional[int]=None)->Any:
        arg_types:List[Any] = [infer_ctype(a) for a in args]
        sig:str = type_sig(arg_types)
        entry:Optional[Tuple[Any, List[str]]] = self._cache.get(sig)

        if entry is None:
            entry = self._compile(args, arg_types, sig)
            self._cache[sig] = entry
        kernel_obj, shape_sources = entry

        launch_args:List[Any] = self._build_launch_args(args, arg_types, shape_sources)
        return kernel_obj(*launch_args)

    # ---------- 源码生成 ----------

    def _source_signature(self)->str:
        """python 源码签名:源码变化 -> 重新生成(缓存目录中内容不同则重编译)。"""
        try:
            source:str = inspect.getsource(self.raw_func)
        except OSError:
            source = ""
        return short_hash(f"{source}\nbackend={self.backend}")

    def _resolve_global(self, name:str)->Any:
        return self.raw_func.__globals__.get(name)

    def _c_kernel_name(self, sig:str)->str:
        return f"pyk_{self.__name__}_{short_hash(sig + '_' + self.backend)}"

    def _compile(self, args:List[Any], arg_types:List[Any], sig:str)->Tuple[Any, List[str]]:
        c_name:str = self._c_kernel_name(sig)
        device_cache:Dict[Any, Any] = {}

        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, arg_types, self.backend, self._resolve_global, device_cache, is_kernel=True)
        translator.translate()
        kernel_code:str = translator.render_kernel(c_name)

        dependency_codes:List[str] = []
        for entry in device_cache.values():
            dependency_codes.append(entry["code"])

        source:str = ("\n".join(dependency_codes) + "\n\n" + kernel_code) if dependency_codes else kernel_code

        source_path:str = self._write_source(sig, source)
        kernel:Any = self._backend_compile(source_path, c_name)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        shape_sources:List[Tuple[int, int]] = [(param_index[array_name], dim) for array_name, dim in translator.shapes_used]
        return kernel, shape_sources

    def _cache_dir(self)->str:
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        base_dir:str = (os.path.dirname(os.path.abspath(source_file)) if source_file else os.getcwd())
        folder:str = ("__clcache__" if self.backend == "opencl" else "__cucache__")
        return os.path.join(base_dir, folder, "pykernels")

    def _write_source(self, sig:str, source:str)->str:
        cache_dir:str = self._cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        extension:str = ("cl" if self.backend == "opencl" else "cu")
        path:str = os.path.join(cache_dir, f"pyk_{self.__name__}_{sig}_{self._source_sig}.{extension}")

        existing:Optional[str] = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        if existing != source:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                file.write(source)
        return path

    def _backend_compile(self, source_path:str, c_name:str):
        if self.backend == "opencl":
            from ..opencl.oop.program import Program
            from ..opencl.oop.platforms import Platforms
            device:Any = Platforms[0].devices[0]
            program:Any = Program(device.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        from ..cuda.oop.program import Program as CUProgram
        from ..cuda.oop.devices import Devices
        Devices[0].default_context.make_current()
        program:Any = CUProgram(source_path)
        return program.kernels[c_name]

    # ---------- 启动 ----------

    def _build_launch_args(self, args:List[Any], arg_types:List[Any], shape_sources:List[Tuple[int, int]])->List[Any]:
        launch_args:List[Any] = list(args)
        for param_index, dim in shape_sources:
            launch_args.append(int(args[param_index].shape[dim]))
        return launch_args

    def __repr__(self)->str:
        return f"PyKernel({self.__name__}, backend={self.backend})"


def kernel(func:Optional[Callable]=None, *, backend:str="opencl"):
    """`@kernel` 装饰器。

    用法::

        from pygpgpu.kernel import kernel
        import numpy as np

        @kernel
        def add(a, b, out):
            for i in range(out.shape[0]):
                out[i] = a[i] + b[i]

        add(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) * 2, out)

    - 最外层 `for i in range(<数组参数>.shape[0])` 循环是 work-item 循环,
      work size 自动取该数组第一维;
    - 函数调用的普通 python 函数会被翻译为设备函数(按实参类型单态化);
    - math.* 与 abs/min/max/round 映射到设备内建;
    - 不支持的语法抛 SyntaxError。
    """
    def decorator(f:Callable)->PyKernel:
        return PyKernel(f, backend)

    if func is not None:
        return decorator(func)
    return decorator
