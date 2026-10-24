"""`@kernel` 内核里可用的设备内建函数命名空间。

用法::

    from pygpgpu.kernel import gpu

    @kernel
    def k(out):
        i = gpu.get_global_id(0)
        gpu.atomic_add(out, i, 1)

翻译器识别对 `gpu` 的调用并按后端映射到 OpenCL/CUDA 内建。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import ast
import numpy as np


class PyKernelSyntaxError(SyntaxError):
    """builtins 模块独立携带的错误类型(延迟导入会与 translator 循环)。"""



class GPUNamespace:
    """标记类型:翻译器识别 `gpu.xxx(...)` 调用并映射到设备内建。"""

    def __init__(self)->None:
        self.__name__ = "pygpgpu.kernel.builtins"

    def __getattr__(self, name:str)->Callable:
        def _not_callable(*args, **kwargs)->None:
            from .translator import PyKernelSyntaxError
            raise PyKernelSyntaxError(
                f"gpu.{name} is only available inside a @kernel function body "
                f"(it maps to a device builtin, not a host function)")
        return _not_callable


gpu = GPUNamespace()


# ---------- 翻译期使用的映射元数据 ----------

# 索引类: (gpu 名, opencl 名, cuda 表达式模板), dim 参数必须是整数字面量
INDEX_BUILTINS:Dict[str, Tuple[str, str]] = {
    "get_global_id": ("get_global_id", "(int)(blockIdx{c} * blockDim{c} + threadIdx{c})"),
    "get_local_id": ("get_local_id", "(int)threadIdx{c}"),
    "get_group_id": ("get_group_id", "(int)blockIdx{c}"),
    "get_global_size": ("get_global_size", "(int)blockDim{c} * gridDim{c}"),
    "get_local_size": ("get_local_size", "(int)blockDim{c}"),
    "get_num_groups": ("get_num_groups", "(int)gridDim{c}"),
}

# 原子操作: (gpu 名, opencl 名, cuda 名); 签名 (array, index, value), 仅整型元素
ATOMIC_BUILTINS:Dict[str, Tuple[str, str]] = {
    "atomic_add": ("atomic_add", "atomicAdd"),
    "atomic_sub": ("atomic_sub", "atomicSub"),
    "atomic_min": ("atomic_min", "atomicMin"),
    "atomic_max": ("atomic_max", "atomicMax"),
    "atomic_and": ("atomic_and", "atomicAnd"),
    "atomic_or": ("atomic_or", "atomicOr"),
    "atomic_xor": ("atomic_xor", "atomicXor"),
    "atomic_xchg": ("atomic_xchg", "atomicExch"),
}

# 数学/向量内建: (gpu 名, opencl 名, cuda 名或 None 表示需特殊生成)
GENERIC_BUILTINS:Dict[str, Tuple[str, str]] = {
    "barrier": ("barrier", "__syncthreads()"),
    "rsqrt": ("rsqrt", "rsqrtf"),
    "exp2": ("exp2", "exp2f"),
    "clamp": ("clamp", ""),      # cuda 需特殊生成
    "mix": ("mix", ""),          # cuda 需特殊生成 (a*(1-t)+b*t)
    "mad": ("mad", ""),          # cuda (a*b+c)
    "saturate": ("clamp", ""),   # cuda 需特殊生成 (clamp 到 0..1)
    "step": ("step", ""),        # cuda 需特殊生成 (x < edge ? 0 : 1)
    "dot": ("dot", ""),          # 向量: opencl 原生, cuda 逐分量求和
    "cross": ("cross", ""),
    "normalize": ("normalize", ""),
    "length": ("length", ""),
    "distance": ("distance", ""),
}


def dim_component(dim:int)->str:
    if dim not in (0, 1, 2):
        raise ValueError(f"dimension index must be 0, 1 or 2, got {dim}")
    return (".x", ".y", ".z")[dim]


def emit_index(name:str, dim:int, backend:str)->str:
    opencl_name, cuda_template = INDEX_BUILTINS[name]
    component:str = dim_component(dim)
    if backend == "opencl":
        return f"{opencl_name}({dim})"
    return cuda_template.format(c=component)


def emit_barrier(backend:str)->str:
    if backend == "opencl":
        return "barrier(CLK_GLOBAL_MEM_FENCE)"
    return "__syncthreads()"


def emit_atomic(name:str, array_code:str, index_code:str, value_code:str, elem_type:str, backend:str)->str:
    opencl_name, cuda_name = ATOMIC_BUILTINS[name]
    if backend == "opencl":
        return f"{opencl_name}((volatile __global {elem_type} *)(&({array_code})[{index_code}]), {value_code})"
    return f"{cuda_name}(&({array_code})[{index_code}], {value_code})"
