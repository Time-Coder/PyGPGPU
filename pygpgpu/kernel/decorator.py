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
    scan_self_attrs,
    short_hash,
    type_sig,
)


class _BoundKernelMethod:
    """实例方法绑定形态: instance.kernel_method(...)"""

    def __init__(self, kernel:"PyKernel", instance:Any):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance

    def __call__(self, *args:Any, global_size:Optional[Any]=None)->Any:
        return self._kernel._invoke(self._instance, args, global_size)


class PyKernel:
    """`@kernel` 装饰器的实现:调用时把 python 函数翻译为后端源码并编译执行。

    支持装饰类方法: self 在翻译时剔除, 方法体里的 self.xxx 属性引用被转化为
    附加参数 _self_xxx, 调用时从实例取值传入; 也支持 staticmethod(两种装饰顺序)。
    """

    def __init__(self, func:Callable, backend:str):
        if isinstance(func, classmethod):
            raise PyKernelSyntaxError("@kernel cannot decorate classmethods (cls semantics are host-only)")
        if isinstance(func, staticmethod):
            func = func.__func__

        self.raw_func:Callable = func
        self.backend:str = backend
        self.is_pykernel:bool = True
        self.__name__:str = func.__name__
        self.__doc__:str = func.__doc__ or ""

        arg_names:List[str] = list(func.__code__.co_varnames[:func.__code__.co_argcount])
        self._is_method:bool = bool(arg_names and arg_names[0] == "self")
        self._self_param:str = (arg_names[0] if self._is_method else "")
        self._self_attrs:List[str] = (scan_self_attrs(func, self._self_param) if self._is_method else [])

        self._cache:Dict[str, Tuple[Any, List[Tuple[int, int]], List[Tuple[int, int]], List[int]]] = {}
        self._source_sig:str = self._source_signature()

    def __get__(self, obj:Any, objtype:Optional[type]=None)->Any:
        # 描述符协议: 实例访问返回绑定形态, 类访问/静态方法返回自身
        if obj is None or not self._is_method:
            return self
        return _BoundKernelMethod(self, obj)

    def __call__(self, *args:Any, global_size:Optional[Any]=None)->Any:
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            return self._invoke(args[0], args[1:], global_size)
        return self._invoke(None, args, global_size)

    def _invoke(self, instance:Any, args:Tuple[Any, ...], global_size:Optional[Any])->Any:
        self_values:List[Any] = []
        if self._is_method:
            if instance is None:
                raise TypeError(f"{self.__name__}() requires an instance")
            for name in self._self_attrs:
                if not hasattr(instance, name):
                    raise PyKernelSyntaxError(
                        f"instance of {type(instance).__name__} has no attribute '{name}' "
                        f"required by kernel {self.__name__}")
                value:Any = getattr(instance, name)
                if callable(value) and not isinstance(value, np.ndarray):
                    raise PyKernelSyntaxError(
                        f"self.{name} is a method; calling self.{name}() inside a kernel is not supported")
                self_values.append(value)

        arg_types:List[Any] = [infer_ctype(a) for a in args]
        self_types:List[Any] = []
        for name, value in zip(self._self_attrs, self_values):
            try:
                self_types.append(infer_ctype(value))
            except PyKernelSyntaxError as e:
                raise PyKernelSyntaxError(f"self.{name}: {e}") from None

        sig:str = type_sig(arg_types + self_types)
        entry = self._cache.get(sig)
        if entry is None:
            entry = self._compile(args, arg_types, self_types, sig)
            self._cache[sig] = entry
        kernel_obj, shape_sources, work_size_dims, local_sizes = entry

        full:List[Any] = list(args) + list(self_values)
        launch_args:List[Any] = self._build_launch_args(full, arg_types + self_types, shape_sources)
        if self.backend == "opencl":
            # cuda 的 local memory 是 __shared__ 静态声明,opencl 才需要传字节数
            for local_bytes in local_sizes:
                launch_args.append(local_bytes)
        if global_size is not None:
            gs:Tuple[int, ...] = ((global_size,) if isinstance(global_size, int) else tuple(int(d) for d in global_size))
            kernel_obj = kernel_obj[(gs, None)]
        elif work_size_dims:
            gs = tuple(int(full[pi].shape[dim]) for pi, dim in work_size_dims)
            kernel_obj = kernel_obj[(gs, None)]
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

    def _compile(self, args:List[Any], arg_types:List[Any], self_types:List[Any], sig:str)->Tuple[Any, List[str]]:
        if self.backend == "cuda":
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise PyKernelSyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead")

        c_name:str = self._c_kernel_name(sig)
        device_cache:Dict[Any, Any] = {}
        self_attr_types:Dict[str, Any] = dict(zip(self._self_attrs, self_types))

        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), self.backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types)
        translator.translate()
        kernel_code:str = translator.render_kernel(c_name)

        dependency_codes:List[str] = []
        for entry in device_cache.values():
            dependency_codes.append(entry["code"])

        source:str = ("\n".join(dependency_codes) + "\n\n" + kernel_code) if dependency_codes else kernel_code

        source_path:str = self._write_source(sig, source)
        kernel:Any = self._backend_compile(source_path, c_name)

        # 原子操作目标:静态分析无法识别指针强转写入,直接修正为可写
        from ..constants import KernelFlags
        for target_name in translator.atomic_targets:
            if target_name in kernel._args:
                kernel._args[target_name].kernel_flags = (kernel._args[target_name].kernel_flags | KernelFlags.Writed)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        shape_sources:List[Tuple[int, int]] = [(param_index[array_name], dim) for array_name, dim in translator.shapes_used]
        work_size_dims:List[Tuple[str, int]] = [(param_index[name], dim) for name, dim in translator.work_size_dims]
        # local memory: opencl 启动时按参数位置传字节数; cuda 是 __shared__ 声明无需传
        ctype_sizes:Dict[str, int] = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2,
                                      "int": 4, "uint": 4, "long": 8, "ulong": 8,
                                      "float": 4, "double": 8}
        local_sizes:List[int] = [ctype_sizes.get(ctype, 4) * count for _, ctype, count in translator.local_buffers]
        return kernel, shape_sources, work_size_dims, local_sizes

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
