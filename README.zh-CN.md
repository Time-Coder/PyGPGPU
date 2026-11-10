# PyGPGPU

[English](README.md) | **中文**

装饰一个 Python 函数, 它就运行在 GPU 上——同一份代码, 四个后端 (**OpenCL / CUDA / HIP / Metal**), 外加一个纯 Python 解释器后端。

```python
import numpy as np
from pygpgpu import kernel

@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

x = np.arange(1_000_000, dtype=np.float32)
out = np.zeros_like(x)

saxpy(x, x, out, 2.0)              # 未指定后端: 按 cuda -> hip -> metal -> opencl 探测第一个可用后端
saxpy["cuda"](x, x, out, 2.0)      # 中括号切后端
saxpy["hip:gfx1036"](x, x, out, 2.0)   # 后端:设备
saxpy["python"](x, x, out, 2.0)    # 同一个函数, 在解释器里串行运行(不需要 GPU)
```

不需要写任何 C 代码、不需要配置构建系统; 首次调用时自动翻译/编译并缓存, 之后零开销调用。已有手写 kernel 也没关系——`.cl`/`.cu`/`.hip` 文件可以像 Python 模块一样直接 `import`。

## 特性

- **`@kernel` 装饰器** —— 直接装饰 Python 函数(含类方法/静态方法), 调用时按实参类型翻译为后端源码并编译(单态化);
- **普通 Python 子函数** —— kernel 里调用的子函数不需要任何装饰器, 类型注解也是可选的: 它会被翻译成设备函数, 并在调用点按实参类型单态化; 子函数继续调子函数也支持(递归展开)。经由名字取得的子函数同样支持: `module.func(...)`、`SomeClass.static_method(...)`、`self.static_method(...)`, 以及同一个类里 `self.other_method(...)` 这种实例方法互调。
- **同一进程内各后端并存** —— 由中括号在每次调用时选择后端, 所以一个进程(甚至同一个 kernel 对象)可以混用 OpenCL / CUDA / HIP / Metal; 每个 "后端 + 设备" 组合单独单态化并缓存。
- **纯 Python 后端** —— `k["python"]` 用解释器运行同一份 `@kernel` 函数体, 在 CPU 上串行执行: 不需要 GPU、不需要编译器、不需要任何运行时; 适合测试、小规模数据和作为 GPU 语义的参照实现。
- **手写 kernel 文件 import** —— `.cl` / `.cu` / `.hip` / `.metal` 文件像 Python 模块一样 `import`, kernel 与 struct 类型成为模块属性, 自动生成 `.pyi` 类型提示;
- **底层 OOP API** —— Devices/Context/Buffer/Program/Kernel, 需要精细控制时使用。
- **统一的中括号语法**指定后端/设备/work size(与手写 kernel 完全一致, 顺序不限):
  ```python
  k(args)                          # 全自动
  k["cuda"](args)                  # 后端 (opencl/cuda/hip/metal/python, 含别名 ocl/cu/rocm/amd/mtl/py)
  k["opencl:gfx1036"](args)        # 后端:设备
  k[(1024, 256)](args)             # work size: (总数, 每组), None 为自动
  k["cuda"][(1024, 256)](args)     # 组合
  ```
- **所有后端统一用 CUDA 的中括号**: (blocks_per_grid, threads_per_block), 也就是 CUDA/HIP 用户本来就这么写的形式。k[(4, 256)] 表示 4 个 block、每个 256 个 thread(共 1024 个 work item);k[(n,)] 是只给总个数、block 尺寸自动推导,None 为全自动。同一个中括号在每个后端、@kernel 和低层 oop API 上都成立,block 数不再需要能被什么整除:总量本就是 locks * threads。
- **静态读写分析**: 解析 kernel 函数体, 推断每个指针参数是只读/只写/读写——只写参数跳过上传, 只读参数跳过回读, 连续调用间数据留在设备上(gnp 数组零中间拷贝)。
- **`pygpgpu.numpy` —— 常驻设备的数组**: 形如 numpy 的数组(`gnp.zeros`、`gnp.arange` 等)可以把字节留在设备上, 跨越任意多次 kernel 调用; 公开的 `to_device`/`to_host` 是异步拷贝, 跨设备经宿主中转, 零拷贝只在驱动能保证的后端兑现(cuda/hip/metal), 不能保证时抛出说明原因的异常(opencl); 并支持与 `torch.Tensor` 双向互转。
- **向量类型**: 50 个向量类型(float4/int3/...), 支持 `.x/.y/.z/.w` 成员与 swizzle、构造、算术(标量广播)、比较、逐分量 math。
- **双面函数库** `pygpgpu.funcs`(即 `pygpgpu.vec_types.funcs`): `dot/cross/normalize/clamp/...` 在宿主 Python 直接可运行, 在 `@kernel` 里自动翻译为设备内建。
- **设备内建直接从顶层导出**: `get_global_id`、`barrier`、`atomic_add`、`dot` 等都是普通函数, 从 `pygpgpu` 直接导入即可。在 `@kernel` 函数体里, 翻译器按**它解析到的函数对象**识别(而非名字), 再映射成各后端的调用, 所以同一份函数体在 OpenCL / CUDA / HIP 和解释器后端上都能跑。
- **多维遍历**: `np.ndindex(shape)` / `np.ndenumerate(arr)` 魔法循环(2D/3D), `enumerate(arr)` 一维索引+值快照。
- **三层缓存**: 解析元信息 / 编译产物 / 生成源码, 全部按 mtime 自动失效, 源码不变则零重编译。
- **用 ctypes 绑定的宿主端 CUDA 库** —— cuBLAS、cuBLASLt、cuSOLVER、cuFFT、cuSPARSE 以及 CUDA runtime, 形态与驱动一致(惰性加载的 DLL、签名表、库自己的错误信息), 其上还有一层面向 numpy 的封装(`gemm`、`axpy`、`scal`、`dot`、`rfft`、`csrmv`)。详见下文"宿主端 CUDA 库"。
- **同样方式绑定的宿主端 HIP 库** —— hipBLAS、hipBLASLt、hipFFT、hipSPARSE、hipSOLVER。两个 BLAS 库的签名表与 CUDA 那边共用; 另外三个必须有自己的一份, 因为 HIP 的 plan handle 是指针, 且多个入口的参数个数不同。**数值**则处处是 HIP 自己的(它的 `HIPBLAS_OP_N` 是 111, 不是 cuBLAS 的 0)。详见下文"宿主端 HIP 库"。

## 安装

```bash
pip install pygpgpu
```

Python ≥ 3.7。GPU 运行时按需自备(任一即可):

| 后端 | 需要安装 | 说明 |
|---|---|---|
| OpenCL | GPU 驱动(通常已带) | 零额外安装, 支持 NVIDIA/AMD/Intel 多平台 |
| CUDA | NVIDIA 驱动(含 NVRTC) | Toolkit 不必需, 驱动自带 nvrtc 即可 |
| HIP | AMD HIP SDK | 编译走 SDK 自带 clang: Windows 上是 `<ROCm>/bin/clang.exe`, 自动发现 MSVC/Windows SDK 头文件(若不在默认路径, 用 `PYGPGPU_HIP_INCLUDES` 指定 include 目录, os.pathsep 分隔); Linux 上是 `<ROCm>/bin/clang`(或 `PATH` 上的 `hipcc`), 系统头文件由它自己找到 |
| Metal | Apple 硬件上的 macOS | Metal 后端走框架自带的运行期编译器, 不需要 Xcode 命令行工具; 它必须有真实 Apple GPU, Mac 虚拟机上没有 Metal 设备, 因此该后端在那里不可用 |
| Python | 无 | 在解释器里串行执行 `@kernel` 函数体; 用 `k["python"]` 显式选择, 不参与自动探测 |

`@kernel` 未指定 `backend` 时, 按 **cuda -> hip -> metal -> opencl** 顺序探测, 取第一个枚举到设备的后端。把 Metal 排在 OpenCL 之前, 是因为 macOS 上的 OpenCL 已废弃且只暴露 CPU 设备。

## 快速上手

常用 API 都可以直接从 `pygpgpu` 顶层导入, 不用记子包名:

```python
from pygpgpu import kernel, funcs, float4             # @kernel / 双面函数库 / 向量类型
from pygpgpu import get_global_id, atomic_add        # 设备内建也是顶层普通函数
from pygpgpu import list_devices, get_device, config  # 设备查询 / 进程级设置
from pygpgpu import kernel_import                    # 手写 .cl/.cu/.hip 文件的 import 钩子
```

子包路径(`pygpgpu.kernel` / `pygpgpu.vec_types` / `pygpgpu.opencl` / ...)依然可用。顶层 `pygpgpu.kernel` 指的是 `@kernel` 装饰器, 要取子模块里的东西请写 `from pygpgpu.kernel import ...`。

`kernel_import` 的导入与安装都不需要 GPU 运行时; 合并模块里的某个 kernel 在首次调用时才加载它自己那个后端。

进程级设置都是同一个对象 `pygpgpu.config` 的属性 —— 没有需要另外查找的 `set_...()` 函数:

```python
import numpy as np
import pygpgpu

pygpgpu.config.print_info = False                    # 进度日志
pygpgpu.config.type_map = {float: np.float64}        # python float 使用的宽度
pygpgpu.config.default_device = ["cuda:0", "hip:0"]  # 不带中括号的 k(...) 依次尝试
pygpgpu.config.kernel_import.type_checked = True     # 手写 kernel 的 import 钩子
```

每个属性都在它所属的章节里说明 —— 类型宽度见 “为 `int` 和 `float` 选择宽度”, 默认设备见 “选择默认设备”, 日志开关见 “进度输出”, import 钩子见 “手写 kernel 文件”。

### @kernel 装饰器

```python
import math
import numpy as np
from pygpgpu import kernel, funcs, float4, atomic_add, get_global_id

def norm2(v: float4) -> float:
    return funcs.dot(v, v)

@kernel
def normalize_all(vecs: np.ndarray, out: np.ndarray) -> None:
    for i, v in enumerate(vecs):          # 一维: 索引 + float4 值快照
        out[i] = funcs.normalize(v)       # 设备端逐分量展开

@kernel
def matrix_scale(m: np.ndarray, out: np.ndarray, k: float) -> None:
    for y, x in np.ndindex(out.shape):    # 二维 NDRange
        out[y, x] = m[y, x] * k

@kernel(backend="cuda", domain="data")
def histogram(data: np.ndarray, counts: np.ndarray) -> None:
    i = get_global_id(0)
    atomic_add(counts, data[i], 1)    # 原子操作, 结果自动回读
```

支持的能力: 标量/数组/向量参数, if/while/for-range, 算术/比较/位运算, `math.*` 与 `np.*` 数学函数, 类方法(`self.xxx` 属性自动转为参数), 静态方法, 以及按名字调用辅助函数 —— 模块级函数、`module.func(...)`、`SomeClass.static_method(...)`, 以及同一个类里 `self.method(...)` 这种实例方法互调。不支持的语法抛出带行号的 `SyntaxError`。

凡是需要数组参数的地方, 都接受 **Python 的数值 list 或 tuple** —— `k([1, 2, 3], out)` 可用, 且 `for x in items: total += x` 能遍历它 —— 因为它是按 numpy 自己的规则转换的(`int64`、`float64`、`complex128`), 与它代表的 ndarray 完全一致。**对象** list 则是另一回事: 它是一组元素, 元素类型由合成的结构体描述, 场景里的多个物体就是这样传给 kernel 的。作为对象成员的**数值 list**(`bag.values[0]`)会变成定长数组字段(`T * N`), 长度取自调用者传入的那个 list。

`break` 与 `continue` 只在真正有循环可作用的地方成立: 写在嵌套的串行 `for`/`while`(或 `for x in arr`)里时, 含义与 Python 完全一致。直接写在魔法循环体内则没有"可继续的循环"可言 —— 每次迭代是一个独立的 work item —— 所以 `continue` 变成"本 work item 到此为止"(下一次迭代是另一个线程, 它本来就会做自己的活), 而 `break` 会抛 `SyntaxError`: 停掉魔法循环等于同时停掉其他所有 work item, 设备无法表达。需要提前结束当前 work item 的剩余部分时, 用 `if` 包住即可。

整数上的 `//` 与 `%` 遵循 Python 的向下取整语义, 负操作数也一样: `-7 // 2` 是 `-4`, `-7 % 2` 是 `1`, 而 C 会给出 `-3` 和 `-1`。翻译器把这个一步修正生成为一个小设备函数, 于是 kernel 与在解释器里跑同一个函数的结果一致; 无符号宽度不需要修正, 直接沿用 C 的运算符。

### 为 `int` 和 `float` 选择宽度

Python 的 `int` 是任意精度, `float` 是 64 位 double, 但 kernel 参数必须是固定宽度, 所以翻译器各取 32 位。装不下的实参会抛 `ValueError`, 并指出该用什么写法携带这个值 —— 绝不会静默截断:

```python
@kernel
def f(step, out): ...          # `step` 是 `int`; f(2 ** 31, out) 会抛 ValueError

f(np.int64(2 ** 31), out)      # numpy 标量自带宽度, 所以这样可以
```

想给 Python 内置类型换默认宽度时, 在装饰器上或通过 `pygpgpu.config` 进程级指定:

```python
import numpy as np
from pygpgpu import config, kernel

@kernel(float=np.float64, int=np.int64)   # python float/complex -> double, python int -> long
def accurate(scale, step, out): ...

config.type_map = {float: np.float64}     # 之后装饰的每个 kernel 都生效
```

`config.type_map` 以 Python 内置类型为键(只能出现 `int` 和 `float`), 值就是装饰器关键字接受的那些类型对象; 赋值整体替换旧映射, 赋 `None` 或 `{}` 恢复内置的 32 位宽度。读回的是解析后的设备类型名, 所以未设置的进程读到 `{}`, 上面的赋值读回是 `{float: 'double'}`。

值是类型对象 —— numpy dtype 或标量类型(`np.int64`、`np.dtype("uint32")`)、ctypes 标量类(`ctypes.c_double`) —— 也可以是设备类型名(`"long"`)。可选集合是封闭的, 且两个键不同:

| 键 | 允许 | 对应设备类型 |
|---|---|---|
| `int` | `np.int8`、`np.int16`、`np.int32`、`np.int64`、`np.uint8`、`np.uint16`、`np.uint32`、`np.uint64` | `char`、`short`、`int`、`long`、`uchar`、`ushort`、`uint`、`ulong` |
| `float` | `np.float32`、`np.float64` | `float`、`double` |

其他一律抛 `ValueError` —— `bool` 不是整数宽度, `half` 在 CUDA/HIP 上不支持, 字符串是**设备**类型名(`np.int8` 可用而 `"int8"` 不可用)。数值装不进所选宽度时同样拒绝: 无符号宽度收到负数会报错, 超出该宽度范围的值也一样, 而不是回绕。

只有 `int` 和 `float` 可映射, 因为 numpy 标量自带宽度并保持不变(`np.uint32(x)` 仍是 `uint`)。Python `complex` 同样没有自己的宽度, 所以跟随 `float=` —— 默认的 32 位 `float` 对应 `complex64`, `float=np.float64` 对应 `complex128`; `np.complex64`/`np.complex128` 保持自己的宽度。这个选择覆盖 Python complex 进入函数体的每一种方式: 参数、字面量(`3 + 4j`), 以及 `complex(re, im)`。

这个选择在装饰时确定 —— 给 `config.type_map` 赋值不会改变已经存在的 kernel —— 并且是 kernel 身份的一部分, 所以两种映射绝不会共用同一个已编译二进制。

映射也会作用到**对象的成员**: 普通对象会变成结构体, 成员就是结构体的字段。因此在 `int=np.int64` 下 `obj.n = 2 ** 31` 按 64 位传输; 而在默认 32 位宽度下同一个值会抛 `ValueError`, 而不是回绕。调用者**显式声明**过的成员类型(`_fields_` 里的 `ctypes.c_int64`)保持其自身宽度 —— 映射不会覆盖写明的东西。

从手写 `.cl`/`.cu`/`.hip` **导入**的 kernel 没有映射: 它的宽度就是源码声明的那样, 超出该宽度的标量会抛出带参数名的 `ValueError`, 而不是传到设备上回绕。

类型注解是可选项: 本 README 里的代码片段带上注解只为便于阅读, 设备端类型来自调用时的实参, 所以同样的代码不写注解也能跑(`self` 不是 kernel 参数, 给它加注解没有任何影响)。

kernel 需要一个迭代域才能推导 work size。要么用魔法循环(`for i in range(x.shape[0])`, 循环边界就是迭代域); 要么在基于 `get_global_id` 的写法里用装饰器指明数组: `@kernel(domain="x")`。迭代域会贡献 `x` 的全部维度, 用 `k[(blocks, threads)]` 中括号指定也可以。三者都没有时调用会直接报错, 而不是猜测。

### 数组元素上的 `+=`

即使多个工作项会撞到同一个元素, `arr[i] op= value` 也遵循 Python 语义: `bins[image[i, j, k]] += 1` 会累计每一个像素, 不需要手写 `atomic_add`。翻译器在无法证明该元素只属于一个工作项时, 会生成设备端原子操作 —— 能证明的情形是下标恰好由魔法循环自己的索引变量构成, 于是 `out[i] += x[i]` 仍是普通的(更快)读改写。

```python
@kernel
def histogram(image: np.ndarray):
    bins: np.ndarray = np.zeros((3, 256), dtype=np.int32)   # 在 host 侧创建
    for i, j, k in np.ndindex(image.shape):
        bins[k, image[i, j, k]] += 1    # 每个像素一次原子加: 不会丢计数
    return bins
```

只有存在对应原子原语的运算符会自动处理(`+=`、`-=`、`&=`、`|=`、`^=`)。有竞争的 `*=`/`/=`/`**=`, 或者往浮点元素上做有竞争的累加, 会抛出 `SyntaxError` 而不是悄悄丢数据: OpenCL 后端没有可移植的浮点原子操作, 请改用整型缓冲区, 或者先按工作项局部累加再做归约(见 `examples/09_reduction.py`)。

### 累加到单个标量

手写 kernel 里是把单个元素的指针(`__global int* total`)交给 `atomicAdd`。这里有三种等价写法:

```python
from pygpgpu import atomic_add, kernel

# 1) 单元素数组就是那个指针
@kernel
def count(x: np.ndarray, total: np.ndarray):
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            atomic_add(total, 0, 1)     # 也可以直接写: total[0] += 1

# 2) 声明为 nonlocal/global 的外部标量会变成设备端指针,
#    调用结束后结果写回外层作用域
@kernel
def count_global(x: np.ndarray):
    global TOTAL
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            TOTAL += 1                  # 设备端是原子操作, 解释器里是原生语义

# 3) 在函数体里创建计数器并作为返回值交回
@kernel
def count_and_return(x: np.ndarray):
    total: np.ndarray = np.zeros(1, dtype=np.int32)
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            total[0] += 1
    return total                        # 单元素数组, 调用方读 [0]
```

三种写法在每个后端上都精确: 成千上万个工作项同时累加同一个位置也不会丢。外部标量永远是**共享**位置, 所以 `TOTAL += 1` 一定是原子操作; 而对它调用 `atomic_add(scalar, 0, 1)` 会被拒绝, 并提示改用前者——原子操作需要一个可下标的数组。

### 在 kernel 里打印

在 OpenCL 与 CUDA 后端上, kernel 函数体里的 `print()` 可以直接用: 它会被翻译成设备端 `printf`, 打印出与 Python 相同的那一行, 调试 kernel 不必再手写 C。

```python
@kernel
def inspect(a: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        print("i =", i, "a[i] =", a[i], "ok =", a[i] > 0, "v =", float2(a[i], a[i] * 2))
        out[i] = a[i]
```

参数之间用一个空格分隔, 行末换行, 与 `print` 一致。每个值按 `str()` 的写法输出: bool 为 `True`/`False`, 整数为十进制, 浮点数为 Python 的最短往返形式(`2.0`、`0.1`、`1e+20`), 向量为 `float4(1.0, 2.0, 3.0, 4.0)`。`None` 打印为 `None`, 字面量字符串会变成固定文本。

有三点需要注意:

- 一行会展开成 **每个分支叶子一个 `printf`**, 由算出的下标选择。原因是设备上每个 printf 调用点都有自己的输出缓冲区, 而缓冲区是按调用点排空回收的: 一行若拆成多个 printf 调用, 打印出来会与其他工作项交错拼接。因此一个 bool 参数会让该行的 printf 数量翻倍(一个 `True`、一个 `False`), 一个浮点参数会乘以它的位数预算; 一行参数过多时会触发保护并提示拆成多次调用;
- 行与行之间的顺序是设备的调度顺序, 而不是循环顺序, 因为工作项是并发执行的。每一行本身保持完整;
- **HIP 后端不支持 `print()`**: 在被测的 ROCm 7.2 机器上, 启动带这些 printf 调用点的 kernel 会让 AMD 驱动丢失设备(随后 Windows 会重置显示驱动)。因此在为 HIP 编译时, kernel 体内的 `print()` 会直接报错: 想看输出请用 `opencl` 或 `cuda` 后端运行, 或者把结果写进数组再回到 host 查看。

纯 Python 后端(`k["python"]`)无需翻译: 它的 `print` 就是普通内建函数, 因此关键字参数和打印数组在那里同样可用。

### 非阻塞启动

装饰后的 kernel 与手写 kernel 一样提供 `submit` / `async_call`, 所以 `@kernel` 函数不用写任何 C 代码也能异步启动:

```python
import asyncio
from pygpgpu import kernel

@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
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

一次启动需要一条执行通道, 三个后端都是池化复用而不是每次调用新建。在 CUDA 与 HIP 上这条通道是 stream: 主线程的同步调用使用后端的默认(NULL)stream, 不分配任何资源; 其它线程的同步调用各自缓存一条 stream, 放在每个 context 上限为 16 条的 LRU 表里, 所以反复创建/结束线程也不会累积 stream。异步启动从另一个上限为 16 条的非阻塞 stream 池中取用: 当它们全部在飞行中时, `submit`/`async_call` 会立即返回, 等有 stream 空出来后在工作线程上完成, 而同步调用则直接等待。若某条 stream 所属线程的 context 已经不存在, 该 stream 会被丢弃而不再复用, 因为 CUDA/HIP 的 stream 属于创建它时的那个线程的 context。 OpenCL 同样池化命令队列, 只有一点不同: 它没有默认队列, 所以主线程也要缓存一条, 而不是走共享的 NULL 通道。队列放在每个 context 上限为 16 条的 LRU 表里, 同样具备接管 / 改挂 / 跨池借用的策略, 异步启动使用另一个上限为 16 条的池。现在一百次阻塞启动除了已缓存的那条之外不会再创建任何队列。 stream 还会跨线程、跨两个池之间回收: 新线程会接管一个空闲的缓存 stream 而不是新建, 缓存满时把最久未用的条目改挂到当前线程而不是销毁, 一个池在新建之前会先取走另一个池的空闲 stream, 因此两个池共用同一份存活 stream 预算。

### 纯 Python 解释器后端

`k["python"]`(别名 `"py"` / `"host"`)把装饰后的函数交给解释器执行, 而不是编译给设备。它不做任何翻译, 因此函数体不必限制在 GPU 翻译器接受的语法内, 也不需要安装任何东西: 没有驱动、没有缓存目录、没有设备。

```python
@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

saxpy["python"](x, y, out, 2.0)      # 普通 Python, 一次调用, 串行
saxpy["py"].submit(x, y, out, 2.0)   # 同一次启动, 放到线程池里执行
```

它的定位是正确性和便利性, 不是速度: 整个 kernel 在 CPU 上串行跑完。启动方式同样支持 `submit`(返回 `concurrent.futures.Future`)与 `async_call`(返回 `asyncio.Future`), 两者都由 `ThreadPoolExecutor` 执行; 单次启动仍是串行的, 所以并发启动的多次调用必须写不同的输出数组。

两种 kernel 写法保持各自语义, 这也是它与普通调用的区别所在:

| kernel 写法 | `k["python"](...)` 的行为 |
|---|---|
| 魔法循环(`for i in range(x.shape[0])`、`enumerate(x)`、`np.ndindex`/`np.ndenumerate`) | 函数只调用一次: 循环本身已经覆盖了所有元素。此时 `(blocks, threads)` 中括号不改变任何东西, 因为决定迭代范围的是循环, 不是启动规模 |
| gid 写法(`get_global_id`), 迭代域来自 `@kernel(domain="x")` | 按声明的迭代域(`x` 的完整 shape)每个工作项跑一遍函数体, `get_global_id` 返回当前索引——也就是按 `np.ndindex` 顺序, 每个元素一步 |
| gid 写法, 迭代域来自 `k[(blocks, threads)]` 中括号 | 同上, 按中括号的每个工作项跑一遍, 因此函数体必须像在 GPU 上一样, 自己针对真实元素个数做越界保护 |

这类函数体里的内建函数会解析到 host 侧实现: 索引内建、原子操作(退化为普通读-改-写, 串行执行使其结果精确)、`barrier()`(空操作)以及向量数学函数族(直接调用 `pygpgpu.funcs` 的 host 实现)。有两种东西无法模拟, 会直接报错: `local`(组内共享内存需要并行工作项)与魔法循环函数体里的 `get_global_id`(那里没有"当前工作项", 请改用循环变量)。

向量数组可以照常使用: 读取 `v = points[i]` 得到向量实例, `out[i] = v.xyzy * 2.0` 写回各分量。

### 一份函数体, 两种行为: KERNEL_TRANSLATING

解释器从头到尾照原样执行函数体, 所以当一个 kernel 在两边需要**不同**代码时, 差异无处安放 -- 而对一个设备看不到的值写普通 if, 会被拒绝。KERNEL_TRANSLATING 就是这个出口, 借鉴 	yping.TYPE_CHECKING: 它在 Python 里是 False, 在翻译函数体时是 True, 因此条件在**翻译期**就被定下, 只有一个分支会被生成:

`python
@kernel(domain="out")
def measure(data: np.ndarray, out: np.ndarray):
    i = get_global_id(0)
    if KERNEL_TRANSLATING:
        out[i] = data[i] * 2.0          # 设备: 真正的计算
    else:
        out[i] = len(str(i))            # Python: 设备无法执行的仅主机表达式
`

分支会像宏一样被内联 -- 生成的源码里不会留下任何 if -- 而被丢弃的分支根本不会被翻译, 所以仅限 Python 的代码对设备零成本。识别的只是**名字**, 因此 kernel 也可以自己声明一个 KERNEL_TRANSLATING, 不必从这里 import。

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
from pygpgpu import kernel_import
import numpy as np

kernel_import.install()
import vecadd                          # 一个 .cl 文件 = 只有一个后端的模块

a = np.random.rand(1_000_000).astype(np.float32)
out = np.zeros_like(a)
vecadd.vecadd(a, a, out, np.int32(len(a)))        # 自动推断 work size
vecadd.vecadd[(3907, 256)](a, a, out, np.int32(len(a)))   # (blocks, threads)
```

注意那个 `n` 参数: 手写 kernel 必须按真实元素个数做守卫, 因为启动的 work size 会被向上取整到
local size 的整数倍。`get_global_size(0)` 返回的是取整后的启动尺寸、不是缓冲区长度, 所以不能拿它当边界。

struct 定义自动映射为 ctypes 类型; kernel 旁自动生成 `.pyi`, IDE 补全可用。钩子自身的设置在
`config.kernel_import`, 每一项都在 kernel 模块被导入时读取:

```python
from pygpgpu import config

config.kernel_import.type_checked = True    # 按生成的 stub 检查每次调用的 dtype
config.kernel_import.generate_pyi = False   # 不在源码旁写 .pyi
config.kernel_import.options = {"cuda": build_options}   # 后端 -> BuildOptions
```

`type_checked` 把错误的 dtype 变成调用处的 `TypeError`, 而不是设备上的失败; `generate_pyi`
默认为 `True`。给 `options` 赋值会整体替换映射, 新映射没有提到的后端回到各自的构建默认值。

### 一个 kernel, 多个后端

同名的 `.cl` / `.cu` / `.hip` 文件就是同一个模块。把 `vecops.cl`、`vecops.cu`、
`vecops.hip` 放进一个包目录(带 `__init__.py`), 然后:

```python
from pygpgpu import kernel_import

kernel_import.install()
from kernel_import import vecops        # 存放三份源文件的包

vecops.add_offset(a, b, out, offset, n)      # 默认设备, 用法与 @kernel 相同
vecops.add_offset["cuda"](a, b, out, ...)    # 中括号选择后端
vecops.add_offset["opencl:1"][(blocks, threads)](...)   # 设备 + work size, 顺序任意
```

三份文件如何合成一个模块:

* 同名且**参数表相同**的 kernel 是同一个函数的多个后端实现 -- 中括号选择实现, 调用保持
  kernel 自己的签名;
* **参数表不同**则为重载 -- pyi 里每个参数表一个 `@overload`, 调用时由实参类型选定实现
  (不带括号时按与 `@kernel` 相同的 `cuda -> hip -> metal -> opencl` 探测);
* 只在一个文件里声明的 kernel 依然存在, 中括号指到没有实现的后端时在括号处抛 `KeyError`。

合并模块同样生成 `.pyi`: 每个 kernel 一个 `class Kernel_<name>`, 每个不同参数表一个
`__call__`, struct 取第一份声明它的文件。

### 科学计算函数库

设备端的数学函数和别的内建函数一样, 就是普通的名字:

```python
from pygpgpu import erf, tgamma, fast_exp, normal_cdf

@kernel
def integrate(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = erf(x[i]) * tgamma(x[i] + 1.0)
```

分三组, 区别很重要:

* **所有设备后端都有**的函数 -- `erf`、`erfc`、`tgamma`、`lgamma`、`rint`、`remainder`、
  `fmin`、`fmax`、`fdim`、`copysign`、`nextafter`、`ldexp`、`fract`、`sign`, 以及
  `native_*` 一族。它们在 OpenCL、CUDA、HIP、Metal 上都能跑。
* **CUDA 与 HIP 有、OpenCL 没有**的函数 -- 快速近似(`fast_exp`、`fast_log`、`fast_sin` ...)、正确舍入
  形式(`exact_recip`、`exact_divide`、`exact_sqrt` ...)、特殊函数(`erf_inverse`、
  `normal_cdf`、`reciprocal_hypot`、`sin_pi`、`cos_pi`、贝塞尔函数)以及转换
  (`float_to_int_nearest`、`int_to_float` ...)。用到它们的 kernel 就是 **CUDA 或 HIP 的 kernel**:
  在 OpenCL 上翻译器会**指名道姓**地拒绝, 消息里写明是哪个函数没有 OpenCL 对应物,
  而不是丢给你一句编译器的 "undeclared identifier"。
* `mad` / `fma` 一族与向量运算 -- 这些 pygf 已经发布, 这里用的就是 pygf 自己的对象
  (见 `pygpgpu.funcs`)。

同一个名字在 CUDA 与 HIP 上答案相同, 但两者不是同一块设备, 差别值得知道:

* HIP 没有 `__exp2f`, 所以那里的 `native_exp2` 写作 `exp2f` -- 普通拼写, HIP 有。
* HIP **完全没有**向零舍入的内建(`__frcp_rz`、`__fdiv_rz`、`__fsqrt_rz`), 所以
  `exact_recip_rz`、`exact_divide_rz`、`exact_sqrt_rz` 在 HIP 上由包发射的一个设备端辅助函数
  算出来: 它取正确舍入的值, 用融合乘加判断精确结果落在它的哪一侧, 若在另一侧就朝零方向挪
  一个最小单位。结果与 CUDA 自己的内建**逐位相同**(除法是唯一的例外 -- 两块硬件的除法单元在
  最后一位上舍入不同, 但**舍入方向**两边一致)。

有两个名字值得提醒, 因为以设备端文档为准时, 拼写本身会骗人:

* CUDA 的 `rhypotf` 是斜边的**倒数**(`1/sqrt(x²+y²)`), 所以这里叫 `reciprocal_hypot`;
  斜边本身叫 `hypot`。
* CUDA 的 `sinpif` 是 `sin(πx)`, **不是** sinc 函数, 所以这里叫 `sin_pi`。

### 给已有的设备端函数起个名字

不是内建的设备端函数 -- 你自己头文件里的 `__device__` 函数, 或者你想换个拼写的编译器
内建 -- 可以注册:

```python
from pygpgpu import device_function

@device_function(cuda="my_fast_rcp", opencl="native_recip",
                 headers={"cuda": "myclib.h"})
def my_fast_rcp(x: float) -> float:
    """宿主端实现; 设备端调用注册表里指名的那个符号。"""
    return 1.0 / x

@kernel
def use_it(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = my_fast_rcp(x[i])
```

设备端调用表里指名的符号; 头文件会被复制进缓存并写进生成的源码, 所以你不必知道缓存在哪
(写成完整 `#include` 行的头文件则原样带进源码, 系统头文件如 `#include <cuda_fp16.h>` 需要这样)。
某个后端没有对应符号时, 会像 CUDA-only 内建那样按名字拒绝。

函数体是**宿主端**的实现 -- `python` 后端跑的就是它 -- 所以两侧必须一致, 与内建函数的
要求完全一样。

### 类方法

```python
class Model:
    def __init__(self) -> None:
        self.weights = np.random.rand(1024).astype(np.float32)
        self.lr = 0.01

    @kernel
    def apply(self, x: np.ndarray, out: np.ndarray) -> None:
        for i in range(out.shape[0]):
            out[i] = x[i] * self.weights[i] + self.lr

m = Model()
m.apply(x, out)                     # self.weights/self.lr 自动作为 kernel 参数传入
m.apply["cuda"](x, out)             # 中括号照常可用
```

### 按名字调用辅助函数

kernel 可以调用用普通 Python 写的辅助函数。除了直接用裸名字调模块级函数, 辅助函数也可以经由名字取得:

```python
import mylib

def soften(v: float) -> float:            # 模块级辅助函数
    return v * 0.5

class Model:
    def __init__(self) -> None:
        self.level = 1.0            # 被 gain() 读取, 所以会成为 kernel 参数

    @staticmethod
    def scale(v: float) -> float:         # 静态方法: 没有接收者
        return v * 3.0

    def gain(self) -> float:              # 实例方法: 读取 self.level
        return self.level + 1.0

    @kernel
    def apply(self, x: np.ndarray, out: np.ndarray) -> None:
        for i in range(out.shape[0]):
            out[i] = soften(x[i]) * self.scale(x[i]) + self.gain() + mylib.bias(x[i])
```

- `self.method(...)` 调用同一个类的另一个方法。它和其他辅助函数一样被翻译, 区别只在于**它**读取的 `self` 属性(上面例子里的 `self.level`)也会成为 kernel 参数 —— 嵌套调用会递归收集。
- `SomeClass.static_method(...)` 和 `self.static_method(...)` 可用于任意类的静态方法, 不需要接收者。
- `module.func(...)` 可用于经由模块对象取到的普通函数。
- 设备端不支持: 调用 `@kernel` 装饰的方法、`classmethod`(设备上没有 `cls`)、以及**其它实例**的方法 —— 设备函数没有对象模型, 请把那个对象的数组作为显式 kernel 参数传进来。属性也不能写回(`self.x = ...` 会被拒绝, 请写成数组元素)。

### 拷贝优化

kernel 对每个指针参数做函数体级静态分析(`points[i] = ...` 写、`x = a[i]` 读、指针别名传播、跨函数过程间分析):

```python
@kernel
def k(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i] * 2.0   # a 只读, b 只读, out 只写
# 运行时: a/b 上传后跳过回读, out 跳过上传只回读
```

配合 `pygpgpu.numpy`(gnp) 数组, 连续 kernel 调用之间数据完全留在设备端, 仅在宿主访问时才同步。

### 常驻设备的数组: `pygpgpu.numpy`

`import pygpgpu.numpy as gnp` 给到的就是 numpy —— 同样的构造器、同样的运算、同样的 API,
只多一个对 kernel 而言关键的区别: 数据可以留在设备上, 不必每次启动后都回到 CPU。

```python
import pygpgpu.numpy as gnp
from pygpgpu import kernel

@kernel
def saxpy(x, y, out, alpha):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]

a = gnp.arange(1_000_000, dtype=gnp.float32)
b = gnp.arange(1_000_000, dtype=gnp.float32)
out = gnp.empty(1_000_000, dtype=gnp.float32)

saxpy(a, b, out, 2.0)      # 上传 a、b、out, 每个一次
saxpy(a, b, out, 2.0)      # 什么都不动: 它们已经在那台设备上
print(out[0])              # ……这一句才把字节取回来

a.to_device("opencl:1")    # 显式、异步地拷到另一台设备
a.to_host()                # 异步取回; 下一次真正的读取会等它
```

在宿主这一侧 gnp 数组就是普通 numpy 数组 —— `a + 1`、`np.sum(a)`、`a[0]`、`a.T`、
`a.tolist()` 与 `numpy.ndarray` 完全一致 —— 所以唯一改变的只是: kernel 调用不再每次付出一次往返拷贝。

**`to_device` / `to_host` 是异步的。** 两者都返回一个 `Transfer`, 它同时是屏障(`.wait()`)、
future(`.future()`、`await transfer`)和这次拷贝的描述。引擎不会 "以防万一" 地等待, 只在真正依赖结果的
地方等:

* 读取宿主字节之前, 若还有在途拷贝仍在写它们;
* 在某个缓冲区上传还没落地时, 又要往它上面启动;
* 每次在宿主侧真正读取数组之前 —— `out[0]`、`a + 1`、`tolist()` —— 这也正是为什么没被等待的
  `to_host()` 依然正确。

`to_device` 接受与 kernel 方括号里相同的选择器(`"cuda"`、`"opencl:1"`、`"hip:gfx1036"`,
`(backend, device)` 对, 或 `Device` 对象), `None` 表示普通 `k(...)` 调用会用的默认设备。

**跨设备一律经由宿主中转。** 两个后端的缓冲区类型不同, 即使同一后端的两个设备也可能属于不同 context,
所以设备到设备就是一次下载加一次上传。最新的内容获胜: 每个设备副本带着它持有的那份字节的 epoch,
宿主读取时从 "走得最远" 的那台设备取回 —— 因此在设备 A 上跑一次、再在设备 B 上跑一次, 然后读宿主,
得到的是 B 的结果。

**零拷贝**让设备直接读写数组自己的内存, 而不是它的副本:

```python
ok, why = gnp.zero_copy_support("opencl")     # (False, "...CL_MEM_USE_HOST_PTR...")
a.to_device("cuda:0", zero_copy=True)         # cuda/hip/metal: 数组本身就是那个缓冲区
```

只有在驱动**确实**能保证时才兑现 —— CUDA/HIP 的映射锁页内存(`cuMemHostRegister` +
`cuMemHostGetDevicePointer`)、Metal 的 `StorageModeShared` —— 其余情况下**抛异常并说明原因**。
OpenCL 属于后者: `CL_MEM_USE_HOST_PTR` 允许运行时保留一份私有副本, kernel 写入可能永远到不了调用者的
数组, 所以这个请求直接抛异常, 而不是过一会儿给出错误答案。

**PyTorch tensor 双向互转**, 宿主侧的两种形式可以共享同一块缓冲区:

```python
array = gnp.ndarray.from_torch(tensor, zero_copy=True)   # CPU tensor: 同一块存储
tensor = array.to_torch(zero_copy=True)

gnp.ndarray.from_torch(cuda_tensor)     # 设备 tensor: 经由宿主中转
array.to_torch()                       # tensor 自己持有的一份拷贝
```

`from_torch(zero_copy=True)` 拒绝 CUDA tensor: 那块设备存储归 torch 所有, 而 gnp 数组的字节在宿主内存,
两者之间并不存在 "零拷贝", 于是调用直接说明这一点。`to_torch` 会先把常驻设备的数组取回宿主,
因为它的宿主字节按设计就是过期的。

有两点值得知道:

* **是 `gnp` 的构造器才产生 gnp 数组。** `np.zeros(...)` 返回普通 `numpy.ndarray`(numpy 自己的函数),
  行为正常但没有设备常驻; `gnp.zeros(...)` 才返回 gnp 数组。混用只是悄悄失去优化, 这也正是两个命名空间
  分开的原因。
* **视图共用同一份设备副本。** `a[1]`、`a.T`、`a.reshape(...)` 都是视图, 视图的字节属于父数组的缓冲区,
  所以 kernel 写 `a[1]` 会把整个 `a` 标成设备上已变更, 而不是只标那一行(多标是安全的, 少标才会丢数据)。

## 与其他库的用法对比

同一段计算, 用本库和另外三个 "从 Python 调用 GPU" 的项目
([Taichi](https://www.taichi-lang.org/)、[Numba](https://numba.pydata.org/)、
[NVIDIA Warp](https://github.com/NVIDIA/warp)) 分别怎么写。代码只保留有差异的部分,
具体语法请以各项目文档为准。示例里都带了一个共享的子函数, 这里是 API 差距最明显的地方:
本库它就是普通 Python 函数, 而其他几个需要再加一个装饰器和类型标注。

**PyGPGPU** —— 数组直接进、直接出, 不用管启动规模, 不用管缓冲, 子函数也不需要任何装饰器:

```python
from pygpgpu import kernel

def square(v):                          # 普通 Python: 不需要装饰器, 也不需要写类型标注
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
def apply_all(x: np.ndarray, out: np.ndarray) -> None:
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
| 子函数 | **普通 Python——不需要装饰器, 注解可选** | `@ti.func` + 标注参数 | `@cuda.jit(device=True)` | `@wp.func` + 标注参数 |
| 同进程多后端 | **三个都能用, 可自由混用** | 只有一个, 由 `ti.init(arch=...)` 固定 | 仅 CUDA | 仅 CUDA |
| 启动规模 | 由循环上界推导(或 `domain="x"` / 中括号) | 由 `range` 推导 | block × thread 手算, 外加守卫 | 显式传 `dim=` |
| 边界守卫 | 自动生成 | 自动生成 | 自己写 | 自己写 |
| 后端 | OpenCL, CUDA, HIP, Metal | CPU, CUDA, Vulkan 等 | CPU, CUDA | CPU, CUDA |
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
def apply_all(x: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = square(x[i]) + 1.0

apply_all["cuda"](x, out)      # NVIDIA
apply_all["hip"](x, out)       # AMD
apply_all["opencl"](x, out)    # OpenCL 枚举到的设备
apply_all["python"](x, out)    # 解释器, 在 CPU 上执行(不需要设备)
apply_all(x, out)              # 不指定: 按 cuda -> hip -> metal -> opencl 探测
```

每个 "后端 + 设备" 组合都单独单态化和缓存, 所以在它们之间来回切只是多一次字典查找——首次调用之后
不会重新编译。用它可以在同一台机器上横向对比不同厂商(见 `examples/08_compare_backends.py`),
也可以在首选 GPU 忙时切到第二块卡。

反过来, 这几个项目更强的地方, 选型前值得了解:

- **自动微分**: Taichi、Warp 可以对 kernel 求导, 本库不支持;
- **稀疏与不规则数据结构**: Taichi 有 sparse grid / SNode tree, 本库传的是稠密 numpy 数组(稀疏要自己处理);
- **CPU 回退**: 三者都能跑 CPU; 本库只有显式的 `k["python"]` 解释器后端跑 CPU, 而且它是串行的, 定位是测试而非吞吐;
- **最低解释器版本**: 本库支持 Python 3.7 及以上, 64 位与 32 位 Windows 以及 Linux x86-64 均可; 其他几个需要更新的解释器。Linux 已在 WSL 下端到端验证(CUDA 走 `/usr/lib/wsl` 的驱动直通, OpenCL 走发行版自带的 runtime), 因此那里的驱动名与库发现路径是真实跑过的分支, 而不是未测代码。

这张表依据各项目文档与上面的代码片段整理; 它们的 API 迭代很快, 依赖某一格之前请核对当前文档。

### 返回值与外层变量

kernel 每个工作项执行一次, 因此没有唯一的 C 返回值可以带回。翻译器给每个返回位置一个独立的设备参数——
标量与向量是一元素缓冲区, 数组则直接用数组对象本身——再由第一个走到 `return` 的工作项用原子比较交换
认领结果。宿主在启动完成后读回这些槽位, 返回的就是胜出工作项给出的值:

```python
@kernel
def first_peak(x: np.ndarray):
    for i in range(x.shape[0]):
        if x[i] > 10.0:
            return x[i], i          # 元组
    return None                     # 没有任何工作项返回时就是 None

value, index = first_peak[backend](x)
```

支持的返回类型与支持的参数类型完全一致; 每条 `return` 语句的形状可以不同(某个分支 `return x[i]`,
另一个 `return x[i], i`); 裸 `return` 得到 `None`。`submit()` 与 `async_call()` 在 future 完成
(即读回之后)交出同样的值。魔法循环之后的值返回会被拒绝: 那里返回的是某个工作项的局部值, 而不是函数的
结果。

函数体从外部读取的变量同样成为参数, 因此外层值改变后下一次调用立刻可见——若内联, kernel 会变陈旧:

```python
SCALE = 2.0

@kernel
def scaled(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * SCALE

scaled[backend](x, out)     # SCALE 作为参数传入
SCALE = 100.0               # 下一次调用用 100.0
scaled[backend](x, out)
```

用 `nonlocal`/`global` 声明的外层标量是可写的: 它以指针传入, 内部每次使用都是解引用, 调用结束后值会写回
其作用域——`nonlocal` 写回闭包 cell, `global` 写回模块全局。外层数组也是指针参数, 若 kernel 写了它, 会
被读回。重绑定外层数组(与写其元素相对)会被拒绝, 因为该对象与调用方共享。helper 函数与调用它的 kernel
受到完全相同的处理。

魔法循环之上的语句还可以用 numpy 创建数据——`np.zeros`、`np.ones`、`np.empty`、`np.full`、
`np.arange`、`np.linspace`、`*_like`、`np.float32(...)` 以及其他分配类构造器。这些语句由解释器执行,
因此数组的形状与 dtype 与 numpy 给出的完全一致, 并作为参数传给 kernel:

```python
@kernel
def normalized(x: np.ndarray, out: np.ndarray):
    total = np.zeros(1, dtype=np.float32)      # 在宿主创建
    for i in range(x.shape[0]):
        total[0] = total[0] + x[i]
    for i in range(out.shape[0]):
        out[i] = x[i] / total[0]

@kernel
def doubled(x: np.ndarray):
    result = np.zeros_like(x)                  # 创建后原样返回
    for i in range(x.shape[0]):
        result[i] = x[i] * 2.0
    return result
```

被返回的创建对象复用它本来就在使用的那个参数, 因此调用方拿到的是宿主自己的对象, 而不是副本。魔法循环
之下的创建会被拒绝: 那会为每个工作项创建一个对象, 而不是一个。

### 图像

`Image` 是所有后端共用的图像类型。kernel 里像数组一样下标, 该下标在 CUDA 上翻译为
`tex2D`/`surf2Dwrite`, 在 OpenCL 上翻译为 `read_imagef`/`write_imagef`, 在 `python` 后端
由宿主实现执行, 因此同一份 kernel 在各后端得到相同结果:

```python
from pygpgpu import Image, kernel

@kernel
def brighten(src, dst):
    for y, x in np.ndindex(dst.shape):
        dst[y, x] = src[y, x] * 0.5

src = Image("photo.png", filter="linear")
dst = Image(shape=(height, width), dtype=np.float32)

brighten["cuda"](src, dst)      # 或 "opencl"、"python"
out = dst.read()
```

可以从路径、数组或尺寸构造。通道数是独立参数, 不会从形状推断, 因此 `(4, 4)` 是 4x4 灰度图
而不是 4 像素的 RGBA:

```python
Image(np.zeros((4, 4)))                    # 4x4, 1 通道
Image(np.zeros((4, 4)), channels=4)        # 4x4, 末轴为 4 通道
Image(shape=(4, 4))                        # 分配 4x4, 稍后填充
Image(shape=(4, 4, 3), channels=2)         # 4x4x3 体素, 2 通道
```

通道数必须是 1、2 或 4——设备单次采样能承载的宽度。`channels=3` 会被拒绝并提示补齐到 4,
因为 RGB 不是设备图像格式。

坐标遵循 CUDA 的纹元空间: 纹元 i 覆盖 [i, i+1), 因此整数下标取该纹元中心, 小数下标取纹元
之间的位置, 由线性滤波插值。两者对整个图像固定, 因此 kernel 不需要 sampler 参数; 并且都以枚举
给出(直接传成员值的字符串同样可用):

```python
from pygpgpu import ImageFilter, ImageAddress

Image(arr, filter=ImageFilter.LINEAR, address=ImageAddress.CLAMP)
Image(arr, filter="linear")                       # 等价写法
```

`ImageFilter` 为 `POINT` 或 `LINEAR`。`ImageAddress` 为 `CLAMP`、`WRAP` 或 `MIRROR`, 但只有
`CLAMP` 可移植: 在 cuda 或 opencl 后端上使用 `WRAP`/`MIRROR` 的图像会在启动时报错, 因为在
纹元空间坐标下设备采样器对这两种模式只读取边缘纹元(在开发时所有可用 GPU 上实测), 结果会与
`python` 后端不同。`python` 后端实现了全部三种模式, 需要环绕或镜像的 kernel 可以在那里运行。

HIP 后端拒绝图像: 它复用 CUDA 代码生成, 其中没有纹理对象类型, 且在不支持纹理的硬件上
驱动会拒绝 `hipTexObjectCreate`。

## 宿主端 CUDA 库

CUDA *驱动*(`pygpgpu/cuda/driver`)是底层 API: context、stream、module、内存。CUDA *数学库*
则是以独立 DLL 形式随它一起安装的, 负责真正的数学计算。它们用 ctypes 绑定在
`pygpgpu/cuda/libraries` 下, 形态与驱动一致 —— 惰性加载的 DLL、每个库一张签名表, 以及
被转换成 `RuntimeError`(带上该库自己的说明文字)的错误码:

```python
from pygpgpu.cuda.libraries import driver_for

CUBLAS = driver_for("cublas")          # cuBLAS 的入口点
CUFFT  = driver_for("cufft")           # cuFFT 的入口点
```

| 库 | 已绑定的操作 |
|---|---|
| **cuBLAS** | `gemm`(float32/float64)、`axpy`、`scal`、`dot`、`GemmEx`, 以及 handle 与 stream 管理 |
| **cuBLASLt** | 通用 `matmul`(由启发式挑选算法)、矩阵布局、epilogue |
| **cuFFT** | `plan1d`、`exec`(R2C / C2R / C2C)、销毁 plan、绑定 stream |
| **cuSPARSE** | `csrmv`(createCsr / createDnVec / SpMV)、稠密与稀疏描述符 |
| **cuSOLVER** | `potrf`(Cholesky)、`gesvd`(SVD)、`getrf`(LU), 以及各自的 workspace 查询 |
| **CUDA runtime** | `cudaMalloc`、`cudaFree`、`cudaMemcpy`、`cudaMemset`、设备数量与属性 |

在原始入口点之上还有一层很薄的、面向 numpy 的封装, 调用时传数组而不是设备指针:

```python
import numpy as np
from pygpgpu.cuda_libraries import gemm, axpy, dot, rfft, csrmv

a = np.random.rand(3, 4).astype(np.float32)
b = np.random.rand(4, 2).astype(np.float32)

c = gemm(a, b)                  # a @ b, 跑在 GPU 上
d = axpy(2.0, a[0], b[0])       # 逐元素
e = dot(a[0], b[0])             # 一个标量
f = rfft(a[0])                  # n/2+1 个复数, 与 np.fft.rfft 一致
g = csrmv(row_offsets, column_indices, values, x)   # 稀疏乘法
```

同样的名字也可以作为模块属性访问, 两种写法都成立:

```python
from pygpgpu import cuda_libraries
cuda_libraries.gemm(a, b)

from pygpgpu.cuda_libraries import gemm
gemm(a, b)
```

封装层采用 numpy 的 **row-major** 布局, 返回的也是同样的布局; cuBLAS 读的是 column-major,
这个差异在内部处理掉了, 所以 `gemm(a, b)` 就是 `a @ b`。一维的情形遵循 numpy 自己 `@` 的
规则, 矩阵与向量相乘不需要调用方自己 reshape。

**必须先有 current 的 CUDA context。** 库 handle 归属于创建它时 current 的那个 context,
因此封装层会在没有 current context 时自行置上一个; 但如果你直接使用原始入口点, 就需要
自己先置好:

```python
from pygpgpu import get_device

device = get_device("cuda:0")
device.default_context.make_current()      # 在第一次调用库之前
```

否则调用会以 `CUDA_ERROR_INVALID_CONTEXT` 失败, 而错误信息里会写明这一点。

**这些是宿主端操作, 不是设备内建。** 一次 `gemm` 是把整个数组搬到设备上跑一次库调用;
而 `@kernel` 的函数体是逐线程的算术。这就是为什么 `@kernel` 里不能调用
`cuda_libraries.gemm` —— 两者是不同层次, 混用意味着每个 work item 都去启动一次库调用。
对应的设备端版本(逐元素乘法、逐元素点积)是顶层已经导出的设备内建。

这些库是**惰性加载**的: `import pygpgpu` 完全不会加载任何 CUDA 库, 每个库的 DLL 都在第一次
真正用到它的调用时才加载。没有装 CUDA Toolkit 的机器依然可以 import 这个包, 失败会在调用
时以加载器自己的错误暴露出来。

**cuSOLVER 需要消费级显卡上并没有的 kernel。** NVIDIA 并未为每种架构都提供 cuSOLVER 的稠密
分解 kernel, 没有这些 kernel 的卡不是在准备阶段失败, 而是在执行阶段失败: 调用返回一个失败码,
同时 context 报告非法内存访问。在开发这个功能的机器上实测(GeForce RTX 4070 Laptop, 计算能力
8.9, CUDA 12.6), `potrf`、`getrf`、`gesvd`、`gesvdj` 和 `syevj` 都是这样失败的,
`compute-sanitizer` 把出错点定位在 cuSOLVER 自己的 DLL 内部。绑定本身是正确的, 测试会带着
这个原因跳过, 而不是失败。

## 宿主端 HIP 库

hipBLAS 与 hipBLASLt 是 AMD 对 cuBLAS 与 cuBLASLt 的移植, 所以绑定方式相同, 访问方式也是同一
种层次:

```python
from pygpgpu import hip_libraries

c = hip_libraries.gemm(a, b)          # 在 GPU 上做 a @ b
d = hip_libraries.axpy(2.0, x, y)     # y = 2*x + y
e = hip_libraries.dot(x, y)           # 一个标量
```

两种自然写法都可用, 与 CUDA 那边完全一致:

```python
from pygpgpu import hip_libraries
hip_libraries.gemm(a, b)

from pygpgpu.hip_libraries import gemm
gemm(a, b)
```

`gemm`、`lt_gemm`、`axpy`、`scal`、`dot` 接收 numpy 数组并按 numpy 的行主序返回 —— hipBLAS
读的是列主序, 差异在内部处理, 所以 `gemm(a, b)` 就是 `a @ b`, 包括 numpy 对 `@` 的 1 维
规则和 `transpose_a`/`transpose_b` 开关。另有 `ensure_context`、`allocate`、`free`、
`to_device`、`to_host`、`synchronize`、`create_handle`、`destroy_handle`、`set_stream`,
所以原始入口也能直接用而不必另建一套分配器: `allocate` 走的就是后端已经在用的 HIP runtime。

**API 是 cuBLAS 的, 但数值不是 —— 这就是陷阱。** hipBLAS 的 `HIPBLAS_OP_N` 是 **111**,
而 `CUBLAS_OP_N` 是 0; 它的状态码独立编号(`HIPBLAS_STATUS_ALLOC_FAILED` 是 2, cuBLAS 的是 3);
它的 compute type 是 0..10 的密集编号, 而 cuBLAS 的是 64..77。所以这里的枚举和状态消息是
hipBLAS 自己的([`pygpgpu/hip/libraries/_enums.py`](pygpgpu/hip/libraries/_enums.py)), 从 ROCm
的头文件里读出。把 cuBLAS 的数值传给 hipBLAS 调用, 一次普通的 gemm 就会返回
`HIPBLAS_STATUS_INVALID_ENUM`。

而**签名表**确实就是 cuBLAS 的: 参数表完全相同, 所以它由 CUDA 那份**派生**而来而不是重写一遍,
并且每条都做了**拷贝**而非共享(共享的话, 一个库把绑好的函数缓存进去, 另一个库就会用到它)。
只有名字不同, 而且并不统一: hipBLAS 去掉了 CUDA 12 的 ABI 版本后缀(`cublasSgemm_v2` 是
`hipblasSgemm`), hipBLASLt 的入口是 `hipblasLt*` 而它的 DLL 叫 `hipblaslt.dll`。

**hipBLASLt 需要你的设备上可能没有的 kernel 库。** rocBLASLt 从按架构命名的 Tensile 文件
(`TensileLibrary_lazy_<arch>.dat`)加载 kernel, 而 ROCm 只为独立显卡提供这些文件 —— 在开发
这个功能的机器上(集成显卡 Radeon 610M, `gfx1036`)该文件并不存在, 所以每次 `lt_gemm` 都在
rocBLASLt 内部失败, 而绑定本身没问题。`gemm` 不依赖它。消息里会说明这一点, 因为状态码是
`INVALID_VALUE`, 否则会让人去怀疑参数。

**hipFFT、hipSPARSE、hipSOLVER 也已绑定**, 而且它们的签名表是**照 ROCm 自己的头文件写的**,
不是从 CUDA 那份派生的, 因为签名确实不同:

| | cuFFT / cuSPARSE / cuSOLVER | HIP |
|---|---|---|
| `ExecuteC2C` | 3 个参数 | **4** 个 —— 多了变换方向 |
| `GetSize1d` | 4 个参数 | **5** 个 —— `nx, type, batch, workSize` |
| `GetVersion`(sparse) | 1 个参数 | **2** 个 —— handle 在前 |
| `CreateDnMat` | 6 个参数 | **7** 个 —— 多一个 `order` |
| `GetVersion`(solver) | 有 | **没有** —— hipSOLVER 不提供 |
| plan handle | `int` | **一个指针** |

最后一行最要紧: 在 AMD 平台上 HIP 把 `hipfftHandle` 拼成 `struct hipfftHandle_t*`
(`int` 那种拼法在 `#ifdef __HIP_PLATFORM_NVIDIA__` 里面), 而 cuFFT 的 `cufftHandle` 是
`int`。照抄 CUDA 的表会在库里读 8 字节指针的地方传进一个 4 字节整数 —— 那是第一次调用里
的访问违例, 不是一个错答案。

`rfft` 与 `irfft` 建在这之上, 答案与 numpy 一致:

```python
from pygpgpu import hip_libraries

spectrum = hip_libraries.rfft(signal)          # n//2+1 个复数, 同 np.fft.rfft
back = hip_libraries.irfft(spectrum, n)        # 逆变换, 缩放与 numpy 相同
```

**这三个里有两个并非每台设备都能跑, 其中一个还是静默失败。** rocSPARSE 与 rocSOLVER 从
DLL 旁边的运行时目录里找 kernel, 而 ROCm 7.2 的 Windows 安装为 rocBLAS 和 rocFFT 建了该
目录, 为这两个**都没建**(`bin/rocblas/library` 存在, `bin/rocsparse` 与 `bin/rocsolver`
不存在)。后果不同, 两者都值得知道:

* `hipsparseCreate` 返回 `HIPSPARSE_STATUS_INTERNAL_ERROR` 并留下空句柄, 所以什么都建不起来
  —— 一个干脆的失败。
* `hipsolverDnCreate` 却**成功**, 随后一次分解返回**成功、`devInfo == 0`, 而矩阵原样未动**。
  调用者拿到的是错答案, 且没有任何错误。若你用 hipSOLVER, 请核对数值而不是状态码;
  `hipsolverDnSgetrf_bufferSize` 对一个需要工作区的矩阵返回 0, 是那个便宜的判断依据。

这两个库的测试带着上述原因跳过而不是失败, 因为绑定是验证过的, 是**库**在这台机器上跑不了。
hipFFT 不受影响 —— 上面的 `rfft`/`irfft` 已对着 numpy 验证。

与 CUDA 那些一样, 这些是**宿主端**操作: 一次 `gemm` 把整块数组搬到设备上跑一次库调用, 而
`@kernel` 的函数体是逐线程的算术。kernel 里不能调用它们, 而且这些库是惰性加载的 ——
`import pygpgpu` 不加载任何 HIP 库。


## 设备选择

中括号里的设备标识三种形式(全部后端一致):

| 形式 | 含义 |
|---|---|
| `"0"`, `"1"` | 后端设备列表编号(OpenCL 跨平台扁平编号) |
| `"gfx1036"`, `"geforce"` | 名称/厂商/**架构名**子串匹配(不区分大小写) |
| `"amd"`、`"nvidia"`、`"intel"` | 厂商简称: 与驱动上报的厂商字符串匹配, 因此 `opencl:amd` 能找到厂商写作 `Advanced Micro Devices, Inc.` 的设备 |
| `"geforce:1"` | 名称匹配多个时取第 n 个 |

辅助函数:

```python
from pygpgpu import list_devices, get_device

list_devices()          # 所有可用 "后端:设备" 字符串, 如 ['cuda:0', 'hip:0', 'opencl:0']
dev = get_device("cuda:0")   # 直接取对应后端的 Device 对象
```

### 选择默认设备

没有中括号的调用会按 `cuda -> hip -> metal -> opencl` 使用第一个可用后端。`config.default_device`
可以替换这个顺序:

```python
from pygpgpu import config, kernel

config.default_device = ["opencl:cpu", "cuda"]   # 先试 CPU 设备, 不可用则回退到 cuda
k(x, out)                                        # 无中括号: 按该列表依次尝试

config.default_device = "gfx1036"                # 只写设备名: 会在各后端中查找
config.default_device = "cuda"                   # 单个选择器等价于只有一项的列表
config.default_device = None                     # 恢复内置顺序
```

使用与中括号相同的选择器写法, 逐项尝试直到某一项可用, 不可用的项会被跳过。该列表在
调用时读取, 因此对赋值之前装饰的 kernel 同样生效; 读取 `config.default_device` 得到生效
的列表 —— 未设置时就是内置探测顺序 `cuda -> hip -> metal -> opencl`。
用 `@kernel(backend=...)` 显式指定后端的 kernel 不受影响。

## 缓存

| 后端 | 目录 | 内容 |
|---|---|---|
| OpenCL | `__clcache__/` | 解析 meta + 分设备二进制 |
| CUDA | `__cucache__/` | 解析 meta + 分设备 cubin/ptx |
| HIP | `__hipcache__/` | 解析 meta + 分设备代码对象 |
| Metal | `__metalcache__/` | 生成的 MSL 源码, 首次启动时编译 |
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
from pygpgpu import config

config.print_info = False      # 此后静默
config.print_info = True       # 恢复详细输出(默认)
```

`config.print_info` 只作用于**已经导入**的后端模块, 所以它绝不会仅仅为了静默输出而加载某个 GPU 运行时。它同时记录进程级默认值, 因此在赋值**之后**才导入的后端同样从静默开始 —— 两者的先后顺序无关紧要, 读取 `config.print_info` 得到的是默认值而不是某个后端的开关。只改单个后端时, 直接设置它自己的属性(`pygpgpu.opencl.driver.CL.print_info = False`), 这不会影响共享的默认值。

## 已知限制

- `@kernel` 不支持: 字符串/字典/集合、递归、多维 shape 动态索引。**Python 的数值 list 或 tuple 是支持的**(它变成它所代表的那个数组, 见上文 "支持的能力"), 对象的数值 list 成员同样支持; 对象 list 作为结构体集合也支持。**不支持**的是把嵌套 list 逐层迭代: `for row in grid:` 再 `for cell in row:` 要求每个元素是一个切片, 而 `for x in arr` 永远只产出单个元素 —— 所以二维 Python list 必须整体传入, 用 `grid[i, j]` 索引。闭包与模块全局变量**是**支持的: 外层变量成为 kernel 参数, 用 `nonlocal`/`global` 声明过的会在调用后写回(见上文 "返回值与外层变量");
- kernel 里的 `print()` 逐个参数处理 `True`/`False`、整数、浮点数与向量, `sep`/`end` 可用但需为字符串字面量（格式串固化在设备代码中），`file`/`flush` 不支持, 也不能直接打印数组(请打印某个元素)。含大量浮点数的一行会展开成一个按位数选择的 `switch`, 特别长的一行会触达代码体积保护, 请拆成多次调用。HIP 后端直接拒绝 `print()`: 在被测运行时上, 这类 kernel 会让驱动丢失设备;
- 向量比较/swizzle 写、三元以上嵌套等按跨后端一致性优先做了限制(报错信息会给出替代写法);
- HIP 后端不支持 `Image`: HIP 复用 CUDA 代码生成, 其中没有纹理对象类型, 并且在不支持纹理的硬件上 `hipTexObjectCreate` 会被拒绝(返回 "operation not supported", 例如 ROCm 7.2 Windows 下的集成 Radeon)。翻译器会抛出指明后端的 `SyntaxError`; 图像 kernel 请使用 cuda、opencl 或 python;
- `@kernel` 函数上的类型注解会被接受但忽略: 设备类型来自调用时的实参, 因为同一个 kernel 会按实参类型签名分别特化。注解供编辑器与类型检查器使用, 不参与翻译;
- `@kernel` 签名可以带默认值, 调用也可以用关键字传参: 启动前会先绑定为位置顺序, 并补上省略的默认值。`*args`、`**kwargs` 与仅限关键字参数仍不支持;
- fp16 (`np.float16`) 参数在 CUDA/HIP 后端不可用(建议 float32);
- HIP 后端的编译依赖 AMD HIP SDK 自带 clang; 本机 hipRTC(ROCm 7.2 Windows)对任何输入均失败, 故未使用;
- 纯 Python 后端(`k["python"]`)不模拟 `local`(组内共享内存), 它是串行执行的, 没有并行工作项的概念。

在 CUDA 上，数组 API（`Array` 的 `cuMemcpy2D` 上传/回读）在每个受支持的解释器上都正常。`cuSurfObjectCreate` 与 `cuTexObjectCreate` 偶尔会对一份逐字节合法的描述符返回 `CUDA_ERROR_INVALID_VALUE`；失败集中在进程最初若干次调用，之后不再出现。在开发此项目的机器上实测（驱动 13.4，RTX 4070）：约每一百次创建失败一次，对已失败的描述符重试无法挽回，而为新建数组重新构造描述符通常可以。由于一个 `Image` 每次 launch 需要绑定一个 surface 和一个 texture，同一进程内连续启动多个图像 kernel 的场景可能损失若干次 launch。`python` 与 OpenCL 后端不受影响：OpenCL 的图像在每次 launch 时创建，其代码路径未出现过该失败，而 `python` 后端完全不接触设备。`examples/11_texture2d.py` 在驱动拒绝时报出原因而不是抛异常。
## License

MIT
