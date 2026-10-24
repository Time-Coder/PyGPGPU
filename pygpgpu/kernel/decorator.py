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


# ---------- 中括号选择器解析 ----------

_BACKEND_ALIASES:Dict[str, str] = {
    "opencl": "opencl", "ocl": "opencl", "cl": "opencl",
    "cuda": "cuda", "cu": "cuda",
}


def _parse_selector(sel:str)->Tuple[Optional[str], Optional[str]]:
    """解析 "后端" / "后端:设备" / "设备" 形式的选择器。

    返回 (backend, device);未指定的部分为 None(由默认值/先前中括号兜底)。
    """
    if not isinstance(sel, str):
        raise TypeError(f"kernel bracket selector must be a string like 'opencl:gfx1036', got {type(sel).__name__}")
    sel = sel.strip()
    if not sel:
        raise ValueError("empty kernel bracket selector")
    if ":" in sel:
        head, rest = sel.split(":", 1)
        backend:Optional[str] = _BACKEND_ALIASES.get(head.strip().lower())
        if backend is not None:
            return backend, (rest.strip() or None)
        # 冒号后不是后端名 -> 整串是设备标识(如 "geforce:0")
        return None, sel
    backend = _BACKEND_ALIASES.get(sel.lower())
    if backend is not None:
        return backend, None
    return None, sel


def _parse_work_item(item:Any)->Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """解析 work size 中括号: int / (global,) / (global, local), None 表示自动。"""
    def norm(v:Any)->Optional[Tuple[int, ...]]:
        if v is None:
            return None
        if isinstance(v, int):
            return (v,)
        if isinstance(v, tuple):
            t:Tuple[int, ...] = tuple(int(x) for x in v)
            if not t:
                raise ValueError("work size tuple cannot be empty")
            return t
        raise TypeError(f"work size entry must be int or tuple of ints, got {type(v).__name__}")

    if isinstance(item, int):
        return norm(item), None
    if isinstance(item, tuple):
        if len(item) == 1:
            return norm(item[0]), None
        if len(item) == 2:
            return norm(item[0]), norm(item[1])
    raise TypeError("work size bracket must be n, (global,), or (global, local); use None for auto")


def _resolve_device(backend:str, device_sel:Optional[str])->Tuple[Any, str]:
    """按后端解析设备: None/编号/名称子串("名字:序号" 消歧), 返回 (设备对象, 缓存键)。"""
    if backend == "opencl":
        from ..opencl.oop.platforms import Platforms
        devices:List[Any] = [d for pf in Platforms for d in pf.devices]
    else:
        from ..cuda.oop.devices import Devices
        devices = list(Devices)

    if not devices:
        raise RuntimeError(f"no {backend} devices found")

    if device_sel is None or device_sel.strip() == "":
        return devices[0], devices[0].name

    sel:str = device_sel.strip()
    index:int = 0
    if ":" in sel:
        name_part, idx_part = sel.rsplit(":", 1)
        if idx_part.strip().isdigit() and name_part.strip():
            sel = name_part.strip()
            index = int(idx_part)

    if sel.isdigit():
        i:int = int(sel)
        if not (0 <= i < len(devices)):
            raise ValueError(f"device index {i} out of range (0..{len(devices)-1}) for backend '{backend}'")
        return devices[i], devices[i].name

    low:str = sel.lower()
    matches:List[Any] = [d for d in devices if low in d.name.lower() or low in getattr(d, "vendor", "").lower()]
    if not matches:
        raise ValueError(
            f"no {backend} device matches '{device_sel}'; available: " + ", ".join(d.name for d in devices))
    if index >= len(matches):
        raise ValueError(f"device match index {index} out of range for '{sel}' ({len(matches)} matches)")
    return matches[index], matches[index].name


class _KernelCall:
    """中括号配置形态: k["cuda"][(64, 8)](args) / k[(64, 8)]["cuda"](args), 顺序不限。"""

    def __init__(self, kernel:"PyKernel", instance:Any,
                 backend:Optional[str]=None, device:Optional[str]=None,
                 global_size:Optional[Tuple[int, ...]]=None, local_size:Optional[Tuple[int, ...]]=None):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance
        self._backend:Optional[str] = backend
        self._device:Optional[str] = device
        self._global_size:Optional[Tuple[int, ...]] = global_size
        self._local_size:Optional[Tuple[int, ...]] = local_size

    def __getitem__(self, item:Any)->"_KernelCall":
        if isinstance(item, str):
            backend, device = _parse_selector(item)
            return _KernelCall(
                self._kernel, self._instance,
                (backend if backend is not None else self._backend),
                (device if device is not None else self._device),
                self._global_size, self._local_size)
        global_size, local_size = _parse_work_item(item)
        return _KernelCall(
            self._kernel, self._instance, self._backend, self._device,
            (global_size if global_size is not None else self._global_size),
            (local_size if local_size is not None else self._local_size))

    def __call__(self, *args:Any)->Any:
        instance:Any = self._instance
        call_args:Tuple[Any, ...] = args
        if self._kernel._is_method and instance is None:
            # 类形式: Klass.method["cuda"](instance, args...)
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance = args[0]
            call_args = args[1:]
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device, self._global_size, self._local_size)


class _BoundKernelMethod:
    """实例方法绑定形态: instance.kernel_method(...)"""

    def __init__(self, kernel:"PyKernel", instance:Any):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance

    def __getitem__(self, item:Any)->_KernelCall:
        return _KernelCall(self._kernel, self._instance)[item]

    def __call__(self, *args:Any)->Any:
        return self._kernel._invoke(self._instance, args, None, None, None, None)


class PyKernel:
    """`@kernel` 装饰器的实现:调用时把 python 函数翻译为后端源码并编译执行。

    支持装饰类方法: self 在翻译时剔除, 方法体里的 self.xxx 属性引用被转化为
    附加参数 _self_xxx, 调用时从实例取值传入; 也支持 staticmethod(两种装饰顺序)。

    后端/设备/work size 通过中括号在调用时指定(与手写 kernel 用法统一),
    可用 0/1/2 个中括号, 顺序不限::

        k(a, out)                       # 默认后端/设备, work size 自动
        k["cuda"](a, out)               # 指定后端
        k["opencl:gfx1036"](a, out)     # 后端 + 设备(编号或名称子串, "名:序号"消歧)
        k[(64, 8)](a, out)              # global/local(None 为自动)
        k["cuda"][(64, 8)](a, out)      # 组合, 顺序不限
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

    def __getitem__(self, item:Any)->_KernelCall:
        return _KernelCall(self, None)[item]

    def __call__(self, *args:Any)->Any:
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            return self._invoke(args[0], args[1:], None, None, None, None)
        return self._invoke(None, args, None, None, None, None)

    def _invoke(self, instance:Any, args:Tuple[Any, ...],
                backend_sel:Optional[str], device_sel:Optional[str],
                global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]])->Any:
        backend:str = (backend_sel or self.backend)
        device_obj:Any
        device_key:str
        device_obj, device_key = _resolve_device(backend, device_sel)

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
        cache_key:str = f"{backend}|{device_key}|{sig}"
        entry = self._cache.get(cache_key)
        if entry is None:
            entry = self._compile(args, arg_types, self_types, sig, backend, device_obj)
            self._cache[cache_key] = entry
        kernel_obj, shape_sources, work_size_dims, local_sizes = entry

        full:List[Any] = list(args) + list(self_values)
        launch_args:List[Any] = self._build_launch_args(full, arg_types + self_types, shape_sources)
        if backend == "opencl":
            # cuda 的 local memory 是 __shared__ 静态声明,opencl 才需要传字节数
            for local_bytes in local_sizes:
                launch_args.append(local_bytes)
        if global_size is None and work_size_dims:
            global_size = tuple(int(full[pi].shape[dim]) for pi, dim in work_size_dims)
        if global_size is None and local_size is not None:
            raise ValueError(
                "local work size requires a global size: specify both in the bracket "
                "k[(global, local)] or rely on a magic loop to derive the global size")
        if global_size is not None or local_size is not None:
            kernel_obj = kernel_obj[(global_size, local_size)]
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

    def _c_kernel_name(self, sig:str, backend:str)->str:
        return f"pyk_{self.__name__}_{short_hash(sig + '_' + backend)}"

    def _compile(self, args:List[Any], arg_types:List[Any], self_types:List[Any], sig:str,
                 backend:str, device_obj:Any)->Tuple[Any, List[str]]:
        if backend == "cuda":
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise PyKernelSyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead")

        c_name:str = self._c_kernel_name(sig, backend)
        device_cache:Dict[Any, Any] = {}
        self_attr_types:Dict[str, Any] = dict(zip(self._self_attrs, self_types))

        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types)
        translator.translate()
        kernel_code:str = translator.render_kernel(c_name)

        dependency_codes:List[str] = []
        for entry in device_cache.values():
            dependency_codes.append(entry["code"])

        source:str = ("\n".join(dependency_codes) + "\n\n" + kernel_code) if dependency_codes else kernel_code

        source_path:str = self._write_source(sig, source, backend)
        kernel:Any = self._backend_compile(source_path, c_name, backend, device_obj)

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

    def _cache_dir(self, backend:str)->str:
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        base_dir:str = (os.path.dirname(os.path.abspath(source_file)) if source_file else os.getcwd())
        folder:str = ("__clcache__" if backend == "opencl" else "__cucache__")
        return os.path.join(base_dir, folder, "pykernels")

    def _write_source(self, sig:str, source:str, backend:str)->str:
        cache_dir:str = self._cache_dir(backend)
        os.makedirs(cache_dir, exist_ok=True)
        extension:str = ("cl" if backend == "opencl" else "cu")
        path:str = os.path.join(cache_dir, f"pyk_{self.__name__}_{sig}_{self._source_sig}.{extension}")

        existing:Optional[str] = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        if existing != source:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                file.write(source)
        return path

    def _backend_compile(self, source_path:str, c_name:str, backend:str, device_obj:Any):
        if backend == "opencl":
            from ..opencl.oop.program import Program
            program:Any = Program(device_obj.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        from ..cuda.oop.program import Program as CUProgram
        device_obj.default_context.make_current()
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
