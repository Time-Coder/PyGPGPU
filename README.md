# PyGPGPU

装饰一个 Python 函数, 它就运行在 GPU 上——同一份代码, 三个后端 (**OpenCL / CUDA / HIP**)。

```python
import numpy as np
from pygpgpu.kernel import kernel

@kernel
def saxpy(x, y, out, alpha):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

x = np.arange(1_000_000, dtype=np.float32)
out = np.zeros_like(x)

saxpy(x, x, out, 2.0)              # 默认后端, 自动上 GPU
saxpy["cuda"](x, x, out, 2.0)      # 中括号切后端
saxpy["hip:gfx1036"](x, x, out, 2.0)   # 后端:设备
```

不需要写任何 C 代码、不需要配置构建系统; 首次调用时自动翻译/编译并缓存, 之后零开销调用。已有手写 kernel 也没关系——`.cl`/`.cu`/`.hip` 文件可以像 Python 模块一样直接 `import`。

## 特性

- **`@kernel` 装饰器** —— 直接装饰 Python 函数(含类方法/静态方法), 调用时按实参类型翻译为后端源码并编译(单态化);
- **手写 kernel 文件 import** —— `.cl` / `.cu` / `.hip` 文件像 Python 模块一样 `import`, kernel 与 struct 类型成为模块属性, 自动生成 `.pyi` 类型提示;
- **底层 OOP API** —— Devices/Context/Buffer/Program/Kernel, 需要精细控制时使用。
- **统一的中括号语法**指定后端/设备/work size(与手写 kernel 完全一致, 顺序不限):
  ```python
  k(args)                          # 全自动
  k["cuda"](args)                  # 后端 (opencl/cuda/hip, 含别名 ocl/cu/rocm/amd)
  k["opencl:gfx1036"](args)        # 后端:设备
  k[(1024, 256)](args)             # work size: (总数, 每组), None 为自动
  k["cuda"][(1024, 256)](args)     # 组合
  ```
- **总数语义**(OpenCL/SYCL 风格): `(global, local)` 中的 global 是**总 work-item 数**, CUDA/HIP 的 grid 块数由 `global / local` 自动换算, 三个后端行为严格一致。
- **静态读写分析**: 解析 kernel 函数体, 推断每个指针参数是只读/只写/读写——只写参数跳过上传, 只读参数跳过回读, 连续调用间数据留在设备上(gnp 数组零中间拷贝)。
- **向量类型**: 50 个向量类型(float4/int3/...), 支持 `.x/.y/.z/.w` 成员与 swizzle、构造、算术(标量广播)、比较、逐分量 math。
- **双面函数库** `pygpgpu.vec_types.funcs`: `dot/cross/normalize/clamp/...` 在宿主 Python 直接可运行, 在 `@kernel` 里自动翻译为设备内建。
- **设备内建命名空间** `gpu.*`: `get_global_id/barrier/atomic_add/dot/...` 覆盖 Python 没有对应物的 GPU 内建, 按后端自动映射。
- **多维遍历**: `np.ndindex(shape)` / `np.ndenumerate(arr)` 魔法循环(2D/3D), `enumerate(arr)` 一维索引+值快照。
- **三层缓存**: 解析元信息 / 编译产物 / 生成源码, 全部按 mtime 自动失效, 源码不变则零重编译。

## 安装

```bash
pip install -e .
```

Python ≥ 3.12。GPU 运行时按需自备(任一即可):

| 后端 | 需要安装 | 说明 |
|---|---|---|
| OpenCL | GPU 驱动(通常已带) | 零额外安装, 支持 NVIDIA/AMD/Intel 多平台 |
| CUDA | NVIDIA 驱动(含 NVRTC) | Toolkit 不必需, 驱动自带 nvrtc 即可 |
| HIP | AMD HIP SDK | Windows 上编译走 SDK 自带 clang; 若 MSVC/Windows SDK 不在默认路径, 用 `PYGPGPU_HIP_INCLUDES` 指定 include 目录(os.pathsep 分隔) |

## 快速上手

### @kernel 装饰器

```python
import math
import numpy as np
from pygpgpu.kernel import kernel, gpu
from pygpgpu.vec_types import funcs, float4

def norm2(v):
    return funcs.dot(v, v)

@kernel
def normalize_all(vecs, out):
    for i, v in enumerate(vecs):          # 一维: 索引 + float4 值快照
        out[i] = funcs.normalize(v)       # 设备端逐分量展开

@kernel
def matrix_scale(m, out, k):
    for y, x in np.ndindex(out.shape):    # 二维 NDRange
        out[y, x] = m[y, x] * k

@kernel(backend="cuda")
def histogram(data, counts):
    i = gpu.get_global_id(0)
    gpu.atomic_add(counts, data[i], 1)    # 原子操作, 结果自动回读
```

支持的能力: 标量/数组/向量参数, if/while/for-range, 算术/比较/位运算, `math.*` 与 `np.*` 数学函数, 类方法(`self.xxx` 属性自动转为参数), 静态方法。不支持的语法抛出带行号的 `SyntaxError`。

### 手写 kernel 文件

已有 C 风格 kernel? `.cl` / `.cu` / `.hip` 文件(对应后端原生语法)可以像 Python 模块一样直接 import。写一个 `vecadd.cl`:

```c
__kernel void vecadd(__global const float* a, __global const float* b,
                     __global float* out, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = a[i] + b[i];
}
```

```python
from pygpgpu.opencl import climport    # import pygpgpu.cuda / pygpgpu.hip 同理
import numpy as np
import vecadd                          # 就像 import Python 模块

a = np.random.rand(1_000_000).astype(np.float32)
out = np.zeros_like(a)
vecadd.vecadd(a, a, out, np.int32(len(a)))    # 自动推断 work size
vecadd.vecadd[(1024, 256)](a, a, out, np.int32(len(a)))   # 手动指定
```

struct 定义自动映射为 ctypes 类型; kernel 旁自动生成 `.pyi`, IDE 补全可用。

### 类方法

```python
class Model:
    def __init__(self):
        self.weights = np.random.rand(1024).astype(np.float32)
        self.lr = 0.01

    @kernel
    def apply(self, x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * self.weights[i] + self.lr

m = Model()
m.apply(x, out)                     # self.weights/self.lr 自动作为 kernel 参数传入
m.apply["cuda"](x, out)             # 中括号照常可用
```

### 拷贝优化

kernel 对每个指针参数做函数体级静态分析(`points[i] = ...` 写、`x = a[i]` 读、指针别名传播、跨函数过程间分析):

```python
@kernel
def k(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i] * 2.0   # a 只读, b 只读, out 只写
# 运行时: a/b 上传后跳过回读, out 跳过上传只回读
```

配合 `pygpgpu.numpy`(gnp) 数组, 连续 kernel 调用之间数据完全留在设备端, 仅在宿主访问时才同步。

## 设备选择

中括号里的设备标识三种形式(全部后端一致):

| 形式 | 含义 |
|---|---|
| `"0"`, `"1"` | 后端设备列表编号(OpenCL 跨平台扁平编号) |
| `"gfx1036"`, `"geforce"` | 名称/厂商/**架构名**子串匹配(不区分大小写) |
| `"geforce:1"` | 名称匹配多个时取第 n 个 |

## 缓存

| 后端 | 目录 | 内容 |
|---|---|---|
| OpenCL | `__clcache__/` | 解析 meta + 分设备二进制 |
| CUDA | `__cucache__/` | 解析 meta + 分设备 cubin/ptx |
| HIP | `__hipcache__/` | 解析 meta + 分设备代码对象 |

缓存按源文件(及全部 `#include`)mtime 自动失效; `@kernel` 生成的源码按 Python 源码哈希区分。这些目录应加入 `.gitignore`。

## 已知限制

- `@kernel` 不支持: 字符串/列表/字典、闭包、递归、kernel 返回值、多维 shape 动态索引;
- 向量比较/swizzle 写、三元以上嵌套等按跨后端一致性优先做了限制(报错信息会给出替代写法);
- fp16 (`np.float16`) 参数在 CUDA/HIP 后端不可用(建议 float32);
- HIP 后端的编译依赖 AMD HIP SDK 自带 clang; 本机 hipRTC(ROCm 7.2 Windows)对任何输入均失败, 故未使用。

## License

MIT
