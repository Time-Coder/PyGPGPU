# PyGPGPU

[English](README.md) | **中文**

装饰一个 Python 函数, 它就运行在 GPU 上——同一份代码, 三个后端 (**OpenCL / CUDA / HIP**), 外加一个纯 Python 解释器后端。

```python
import numpy as np
from pygpgpu import kernel

@kernel
def saxpy(x, y, out, alpha):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

x = np.arange(1_000_000, dtype=np.float32)
out = np.zeros_like(x)

saxpy(x, x, out, 2.0)              # 未指定后端: 按 cuda -> hip -> opencl 探测第一个可用后端
saxpy["cuda"](x, x, out, 2.0)      # 中括号切后端
saxpy["hip:gfx1036"](x, x, out, 2.0)   # 后端:设备
saxpy["python"](x, x, out, 2.0)    # 同一个函数, 在解释器里串行运行(不需要 GPU)
```

不需要写任何 C 代码、不需要配置构建系统; 首次调用时自动翻译/编译并缓存, 之后零开销调用。已有手写 kernel 也没关系——`.cl`/`.cu`/`.hip` 文件可以像 Python 模块一样直接 `import`。

## 特性

- **`@kernel` 装饰器** —— 直接装饰 Python 函数(含类方法/静态方法), 调用时按实参类型翻译为后端源码并编译(单态化);
- **普通 Python 子函数** —— kernel 里调用的子函数不需要任何装饰器、不需要类型标注: 它会被翻译成设备函数, 并在调用点按实参类型单态化; 子函数继续调子函数也支持(递归展开)。
- **同一进程内三后端并存** —— 由中括号在每次调用时选择后端, 所以一个进程(甚至同一个 kernel 对象)可以混用 OpenCL / CUDA / HIP; 每个 "后端 + 设备" 组合单独单态化并缓存。
- **纯 Python 后端** —— `k["python"]` 用解释器运行同一份 `@kernel` 函数体, 在 CPU 上串行执行: 不需要 GPU、不需要编译器、不需要任何运行时; 适合测试、小规模数据和作为 GPU 语义的参照实现。
- **手写 kernel 文件 import** —— `.cl` / `.cu` / `.hip` 文件像 Python 模块一样 `import`, kernel 与 struct 类型成为模块属性, 自动生成 `.pyi` 类型提示;
- **底层 OOP API** —— Devices/Context/Buffer/Program/Kernel, 需要精细控制时使用。
- **统一的中括号语法**指定后端/设备/work size(与手写 kernel 完全一致, 顺序不限):
  ```python
  k(args)                          # 全自动
  k["cuda"](args)                  # 后端 (opencl/cuda/hip/python, 含别名 ocl/cu/rocm/amd/py)
  k["opencl:gfx1036"](args)        # 后端:设备
  k[(1024, 256)](args)             # work size: (总数, 每组), None 为自动
  k["cuda"][(1024, 256)](args)     # 组合
  ```
- **总数语义**(OpenCL/SYCL 风格): `(global, local)` 中的 global 是**总 work-item 数**, CUDA/HIP 的 grid 块数由 `global / local` 自动换算, 三个后端行为严格一致。
- **静态读写分析**: 解析 kernel 函数体, 推断每个指针参数是只读/只写/读写——只写参数跳过上传, 只读参数跳过回读, 连续调用间数据留在设备上(gnp 数组零中间拷贝)。
- **向量类型**: 50 个向量类型(float4/int3/...), 支持 `.x/.y/.z/.w` 成员与 swizzle、构造、算术(标量广播)、比较、逐分量 math。
- **双面函数库** `pygpgpu.funcs`(即 `pygpgpu.vec_types.funcs`): `dot/cross/normalize/clamp/...` 在宿主 Python 直接可运行, 在 `@kernel` 里自动翻译为设备内建。
- **设备内建命名空间** `gpu.*`: `get_global_id/barrier/atomic_add/dot/...` 覆盖 Python 没有对应物的 GPU 内建, 按后端自动映射。
- **多维遍历**: `np.ndindex(shape)` / `np.ndenumerate(arr)` 魔法循环(2D/3D), `enumerate(arr)` 一维索引+值快照。
- **三层缓存**: 解析元信息 / 编译产物 / 生成源码, 全部按 mtime 自动失效, 源码不变则零重编译。

## 安装

```bash
pip install pygpgpu
```

Python ≥ 3.7。GPU 运行时按需自备(任一即可):

| 后端 | 需要安装 | 说明 |
|---|---|---|
| OpenCL | GPU 驱动(通常已带) | 零额外安装, 支持 NVIDIA/AMD/Intel 多平台 |
| CUDA | NVIDIA 驱动(含 NVRTC) | Toolkit 不必需, 驱动自带 nvrtc 即可 |
| HIP | AMD HIP SDK | Windows 上编译走 SDK 自带 clang; 若 MSVC/Windows SDK 不在默认路径, 用 `PYGPGPU_HIP_INCLUDES` 指定 include 目录(os.pathsep 分隔) |
| Python | 无 | 在解释器里串行执行 `@kernel` 函数体; 用 `k["python"]` 显式选择, 不参与自动探测 |

`@kernel` 未指定 `backend` 时, 按 **cuda -> hip -> opencl** 顺序探测, 取第一个枚举到设备的后端。

## 快速上手

常用 API 都可以直接从 `pygpgpu` 顶层导入, 不用记子包名:

```python
from pygpgpu import kernel, gpu, funcs, float4       # @kernel / 设备内建 / 双面函数库 / 向量类型
from pygpgpu import available_devices, get_device
from pygpgpu import climport, cuimport, hipimport    # 手写 .cl/.cu/.hip 文件的 import 钩子
```

子包路径(`pygpgpu.kernel` / `pygpgpu.vec_types` / `pygpgpu.opencl` / ...)依然可用。顶层 `pygpgpu.kernel` 指的是 `@kernel` 装饰器, 要取子模块里的东西请写 `from pygpgpu.kernel import ...`。

三个 import 钩子是延迟加载的: 只有真正调用它们时才加载对应后端, 所以 `import pygpgpu` 本身不需要任何 GPU 运行时。

### @kernel 装饰器

```python
import math
import numpy as np
from pygpgpu import kernel, gpu, funcs, float4

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
    gpu.work_size(data)                   # gid 写法需要声明迭代域
    i = gpu.get_global_id(0)
    gpu.atomic_add(counts, data[i], 1)    # 原子操作, 结果自动回读
```

支持的能力: 标量/数组/向量参数, if/while/for-range, 算术/比较/位运算, `math.*` 与 `np.*` 数学函数, 类方法(`self.xxx` 属性自动转为参数), 静态方法。不支持的语法抛出带行号的 `SyntaxError`。

kernel 需要一个迭代域才能推导 work size。要么用魔法循环(`for i in range(x.shape[0])`, 循环边界就是迭代域); 要么在基于 `gpu.get_global_id` 的写法里声明 `gpu.work_size(x)` —— 这条语句只在 host 侧生效, 会被翻译器擦除, 并贡献 `x` 的全部维度。用 `k[(total, local)]` 中括号指定也可以。三者都没有时调用会直接报错, 而不是猜测。

### 非阻塞启动

装饰后的 kernel 与手写 kernel 一样提供 `submit` / `async_call`, 所以 `@kernel` 函数不用写任何 C 代码也能异步启动:

```python
import asyncio
from pygpgpu import kernel

@kernel
def saxpy(x, y, out, alpha):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

# concurrent.futures
future = saxpy.submit(x, y, out, 2.0)
future.result()                       # 回读完成后才返回

# asyncio
await saxpy.async_call(x, y, out, 2.0)

# 中括号同样适用
saxpy["cuda"].submit(x, y, out, 2.0)
saxpy["opencl"][(len(x), 256)].async_call(x, y, out, 2.0)
```

输出数组由完成回调写回, 所以只能在 future 完成之后再读: 提前访问会与传输竞争。`submit` 与 `async_call` 只差 future 的类型(`concurrent.futures.Future` 与 `asyncio.Future`)。

CUDA 与 HIP 后端有一点需要留意: 一次启动需要一条 stream, 而这些 stream 是池化的, 不是每次调用都新建。主线程的同步调用使用后端的默认(NULL)stream, 不分配任何资源; 其它线程的同步调用各自缓存一条 stream, 放在每个 context 上限为 16 条的 LRU 表里, 所以反复创建/结束线程也不会累积 stream。异步启动从另一个上限为 16 条的非阻塞 stream 池中取用: 当它们全部在飞行中时, `submit`/`async_call` 会立即返回, 等有 stream 空出来后在工作线程上完成, 而同步调用则直接等待。若某条 stream 所属线程的 context 已经不存在, 该 stream 会被丢弃而不再复用, 因为 CUDA/HIP 的 stream 属于创建它时的那个线程的 context。OpenCL 后端不受影响。 stream 还会跨线程、跨两个池之间回收: 新线程会接管一个空闲的缓存 stream 而不是新建, 缓存满时把最久未用的条目改挂到当前线程而不是销毁, 一个池在新建之前会先取走另一个池的空闲 stream, 因此两个池共用同一份存活 stream 预算。

### 纯 Python 解释器后端

`k["python"]`(别名 `"py"` / `"host"`)把装饰后的函数交给解释器执行, 而不是编译给设备。它不做任何翻译, 因此函数体不必限制在 GPU 翻译器接受的语法内, 也不需要安装任何东西: 没有驱动、没有缓存目录、没有设备。

```python
@kernel
def saxpy(x, y, out, alpha):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

saxpy["python"](x, y, out, 2.0)      # 普通 Python, 一次调用, 串行
saxpy["py"].submit(x, y, out, 2.0)   # 同一次启动, 放到线程池里执行
```

它的定位是正确性和便利性, 不是速度: 整个 kernel 在 CPU 上串行跑完。启动方式同样支持 `submit`(返回 `concurrent.futures.Future`)与 `async_call`(返回 `asyncio.Future`), 两者都由 `ThreadPoolExecutor` 执行; 单次启动仍是串行的, 所以并发启动的多次调用必须写不同的输出数组。

两种 kernel 写法保持各自语义, 这也是它与普通调用的区别所在:

| kernel 写法 | `k["python"](...)` 的行为 |
|---|---|
| 魔法循环(`for i in range(x.shape[0])`、`enumerate(x)`、`np.ndindex`/`np.ndenumerate`) | 函数只调用一次: 循环本身已经覆盖了所有元素。此时 `(total, local)` 中括号不改变任何东西, 因为决定迭代范围的是循环, 不是启动规模 |
| gid 写法(`gpu.get_global_id`), 迭代域来自 `gpu.work_size(x)` | 按声明的迭代域(`x` 的完整 shape)每个工作项跑一遍函数体, `gpu.get_global_id` 返回当前索引——也就是按 `np.ndindex` 顺序, 每个元素一步 |
| gid 写法, 迭代域来自 `k[(total, local)]` 中括号 | 同上, 按中括号的每个工作项跑一遍, 因此函数体必须像在 GPU 上一样, 自己针对真实元素个数做越界保护 |

这类函数体里的 `gpu.*` 会解析到 host 侧实现: 索引内建、原子操作(退化为普通读-改-写, 串行执行使其结果精确)、`gpu.barrier()`(空操作)以及向量数学函数族(直接调用 `pygpgpu.funcs` 的 host 实现)。有两种东西无法模拟, 会直接报错: `gpu.local`(组内共享内存需要并行工作项)与魔法循环函数体里的 `gpu.get_global_id`(那里没有"当前工作项", 请改用循环变量)。

向量数组可以照常使用: 读取 `v = points[i]` 得到向量实例, `out[i] = v.xyzy * 2.0` 写回各分量。

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
from pygpgpu import climport             # cuimport / hipimport 同理
import numpy as np
import vecadd                          # 就像 import Python 模块

a = np.random.rand(1_000_000).astype(np.float32)
out = np.zeros_like(a)
vecadd.vecadd(a, a, out, np.int32(len(a)))    # 自动推断 work size
vecadd.vecadd[(1_000_192, 256)](a, a, out, np.int32(len(a)))   # 手动指定(总数, 每组)
```

注意那个 `n` 参数: 手写 kernel 必须按真实元素个数做守卫, 因为启动的 work size 会被向上取整到
local size 的整数倍。`get_global_size(0)` 返回的是取整后的启动尺寸、不是缓冲区长度, 所以不能拿它当边界。

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

## 与其他库的用法对比

同一段计算, 用本库和另外三个 "从 Python 调用 GPU" 的项目
([Taichi](https://www.taichi-lang.org/)、[Numba](https://numba.pydata.org/)、
[NVIDIA Warp](https://github.com/NVIDIA/warp)) 分别怎么写。代码只保留有差异的部分,
具体语法请以各项目文档为准。示例里都带了一个共享的子函数, 这里是 API 差距最明显的地方:
本库它就是普通 Python 函数, 而其他几个需要再加一个装饰器和类型标注。

**PyGPGPU** —— 数组直接进、直接出, 不用管启动规模, 不用管缓冲, 子函数也不需要任何装饰器:

```python
from pygpgpu import kernel

def square(v):                          # 普通 Python: 不加装饰器, 不写类型标注
    return v * v

def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def weight(v):                          # 子函数里还能继续调子函数
    return clamp01(square(v)) * 0.5 + 0.25

@kernel
def apply_all(x, out):
    for i in range(out.shape[0]):       # 循环上界就是 work size
        out[i] = weight(x[i])

apply_all(x, out)                       # numpy 数组直接传
apply_all["hip"](x, out)                # 同一个函数, 换个后端
```

**Taichi** —— 先声明 field, 再手动把数据拷进拷出; 子函数要 `@ti.func`, 且参数必须标注类型:

```python
import taichi as ti
ti.init(arch=ti.gpu)

x = ti.field(ti.f32, n)
y = ti.field(ti.f32, n)
out = ti.field(ti.f32, n)
x.from_numpy(nx)
y.from_numpy(ny)

@ti.func                                  # 子函数要自己的装饰器
def weight(v: ti.f32) -> ti.f32:
    return ti.max(0.0, ti.min(1.0, v * v)) * 0.5 + 0.25

@ti.kernel
def saxpy(a: ti.f32):
    for i in range(n):
        out[i] = x[i] * a + y[i] + weight(x[i])

saxpy(2.0)
res = out.to_numpy()
```

**Numba** —— block/thread 数自己算, 边界守卫自己写, 设备内存自己管; 子函数要 `@cuda.jit(device=True)`:

```python
from numba import cuda

@cuda.jit(device=True)                    # 子函数要 device=True
def weight(v):
    return min(1.0, max(0.0, v * v)) * 0.5 + 0.25

@cuda.jit
def saxpy(x, y, out, a):
    i = cuda.grid(1)
    if i < out.size:                      # 边界守卫要自己写
        out[i] = x[i] * a + y[i] + weight(x[i])

d_x = cuda.to_device(nx)
d_y = cuda.to_device(ny)
d_out = cuda.device_array_like(nx)
threads = 256
saxpy[(n + threads - 1) // threads, threads](d_x, d_y, d_out, 2.0)
res = d_out.copy_to_host()
```

**Warp** —— 参数要标类型, 启动规模显式传, 子函数要 `@wp.func`:

```python
import warp as wp
wp.init()

@wp.func                                  # 子函数要自己的装饰器
def weight(v: float) -> float:
    return wp.min(1.0, wp.max(0.0, v * v)) * 0.5 + 0.25

@wp.kernel
def saxpy(x: wp.array(dtype=float), y: wp.array(dtype=float),
          out: wp.array(dtype=float), a: float):
    i = wp.tid()
    out[i] = x[i] * a + y[i] + weight(x[i])

wp.launch(saxpy, dim=n, inputs=[x, y, out, 2.0])
```

| | PyGPGPU | Taichi | Numba | Warp |
|---|---|---|---|---|
| 输入数据 | numpy 数组直接用 | 包进 `ti.field`, 手动 `from_numpy`/`to_numpy` | 手动 `cuda.to_device` + `copy_to_host` | 包进 `wp.array` |
| 子函数 | **普通 Python——不加装饰器、不写类型标注** | `@ti.func` + 标注参数 | `@cuda.jit(device=True)` | `@wp.func` + 标注参数 |
| 同进程多后端 | **三个都能用, 可自由混用** | 只有一个, 由 `ti.init(arch=...)` 固定 | 仅 CUDA | 仅 CUDA |
| 启动规模 | 由循环上界推导(或 `gpu.work_size(x)` / 中括号) | 由 `range` 推导 | block × thread 手算, 外加守卫 | 显式传 `dim=` |
| 边界守卫 | 自动生成 | 自动生成 | 自己写 | 自己写 |
| 后端 | OpenCL, CUDA, HIP | CPU, CUDA, Vulkan 等 | CPU, CUDA | CPU, CUDA |
| AMD GPU | HIP(原生 AMD 代码生成)与 OpenCL | 经 Vulkan | 不支持 | 不支持 |
| 切换后端 | 中括号 `k["hip"]`, 源码不变 | `ti.init(arch=...)`, 源码不变 | 不适用 | 不适用 |
| 安装体积 | 仅 Python + ctypes: 不需 LLVM, 不需 CUDA Toolkit | 自带运行时 | 需要 LLVM | 需要 CUDA Toolkit |
| 手写 `.cl`/`.cu`/`.hip` | 像 Python 模块一样 `import`, 生成 `.pyi` | 不支持 | 需走 `cuda` 模块的 PTX/cubin 路径 | 不支持 |
| 静态读写分析 | 支持——按参数跳过上传/回读 | 不支持 | 不支持 | 不支持 |

### 同一进程内同时使用多个后端

Taichi 在 `ti.init(arch=...)` 那一刻就把后端固定了; Numba 和 Warp 更是只支持 CUDA。本库由中括号
在每次调用时指定后端, 所以同一个进程——甚至同一个 kernel 对象——可以用上机器上所有的后端:

```python
@kernel
def apply_all(x, out):
    for i in range(out.shape[0]):
        out[i] = square(x[i]) + 1.0

apply_all["cuda"](x, out)      # NVIDIA
apply_all["hip"](x, out)       # AMD
apply_all["opencl"](x, out)    # OpenCL 枚举到的设备
apply_all["python"](x, out)    # 解释器, 在 CPU 上执行(不需要设备)
apply_all(x, out)              # 不指定: 按 cuda -> hip -> opencl 探测
```

每个 "后端 + 设备" 组合都单独单态化和缓存, 所以在它们之间来回切只是多一次字典查找——首次调用之后
不会重新编译。用它可以在同一台机器上横向对比不同厂商(见 `examples/08_compare_backends.py`),
也可以在首选 GPU 忙时切到第二块卡。

反过来, 这几个项目更强的地方, 选型前值得了解:

- **自动微分**: Taichi、Warp 可以对 kernel 求导, 本库不支持;
- **稀疏与不规则数据结构**: Taichi 有 sparse grid / SNode tree, 本库传的是稠密 numpy 数组(稀疏要自己处理);
- **CPU 回退**: 三者都能跑 CPU; 本库只有显式的 `k["python"]` 解释器后端跑 CPU, 而且它是串行的, 定位是测试而非吞吐;
- **最低解释器版本**: 本库支持 Python 3.7 及以上, 64 位与 32 位 Windows 均可; 其他几个需要更新的解释器。

这张表依据各项目文档与上面的代码片段整理; 它们的 API 迭代很快, 依赖某一格之前请核对当前文档。

## 设备选择

中括号里的设备标识三种形式(全部后端一致):

| 形式 | 含义 |
|---|---|
| `"0"`, `"1"` | 后端设备列表编号(OpenCL 跨平台扁平编号) |
| `"gfx1036"`, `"geforce"` | 名称/厂商/**架构名**子串匹配(不区分大小写) |
| `"geforce:1"` | 名称匹配多个时取第 n 个 |

辅助函数:

```python
from pygpgpu import available_devices, get_device

available_devices()          # 所有可用 "后端:设备" 字符串, 如 ['cuda:0', 'hip:0', 'opencl:0']
dev = get_device("cuda:0")   # 直接取对应后端的 Device 对象
```

## 缓存

| 后端 | 目录 | 内容 |
|---|---|---|
| OpenCL | `__clcache__/` | 解析 meta + 分设备二进制 |
| CUDA | `__cucache__/` | 解析 meta + 分设备 cubin/ptx |
| HIP | `__hipcache__/` | 解析 meta + 分设备代码对象 |
| Python | 无 | 不生成也不编译任何东西, 函数体直接执行 |

缓存按源文件(及全部 `#include`)mtime 自动失效; `@kernel` 生成的源码按 Python 源码哈希区分。这些目录应加入 `.gitignore`。

## 进度输出

各后端会打印它做了什么——翻译、编译、数据拷贝, 以及每次内核启动, 并带上实际运行的后端名:

```
launch OpenCL Kernel('pyk_saxpy_fb5e5907') on Device('NVIDIA GeForce RTX 4070 Laptop GPU')
       with global_work_size=(1024,), local_work_size=(256,)
launch CUDA Kernel('pyk_saxpy_e6a555c8') on Device('NVIDIA GeForce RTX 4070 Laptop GPU')
       with grid_dim=(3907, 1, 1), block_dim=(256, 1, 1)
```

日志默认开启, 由每个后端各一个开关控制(`CL.print_info`、`CUDA.print_info`、`NVRTC.print_info`、`HIP.print_info`、`HipCompiler.print_info`)。一次全部关闭/打开:

```python
from pygpgpu import set_print_info

set_print_info(False)      # 此后静默
set_print_info(True)       # 恢复详细输出(默认)
```

`set_print_info` 只作用于**已经导入**的后端模块, 所以它绝不会仅仅为了静默输出而加载某个 GPU 运行时。在调用之后才导入的后端会重新回到详细模式——请在你用到的后端都导入后再调用, 或者直接单独设置某个开关(`pygpgpu.opencl.driver.CL.print_info = False`)。

## 已知限制

- `@kernel` 不支持: 字符串/列表/字典、闭包、递归、kernel 返回值、多维 shape 动态索引;
- 向量比较/swizzle 写、三元以上嵌套等按跨后端一致性优先做了限制(报错信息会给出替代写法);
- fp16 (`np.float16`) 参数在 CUDA/HIP 后端不可用(建议 float32);
- HIP 后端的编译依赖 AMD HIP SDK 自带 clang; 本机 hipRTC(ROCm 7.2 Windows)对任何输入均失败, 故未使用;
- 纯 Python 后端(`k["python"]`)不模拟 `gpu.local`(组内共享内存), 它是串行执行的, 没有并行工作项的概念。

## License

MIT
