# 示例 (examples)

[English](README.md) | **中文**

这里存放覆盖 PyGPGPU 各种用法的可直接运行示例。每个文件都是独立的脚本,
在仓库根目录下用 `python examples/xxx.py` 运行(不要在会破坏 `inspect.getsource`
的环境中用 `exec`/stdin 运行)。示例内的注释与输出文本按仓库语言规范统一使用英文。

## 依赖

- 一条 GPU 后端即可运行 `01`–`05`(未指定后端时按 `cuda -> hip -> opencl` 探测);
- `06` 需要手写 kernel 导入钩子对应的运行时(默认 OpenCL, 几乎全平台可用);
- `07` 使用底层 OOP API, 默认 OpenCL 设备;
- `08` 三后端对比只在机器上实际存在的后端上运行, 缺哪个跳过哪个, 总能出结果;
- `09` 只用 `@kernel` + 原子操作, 任意单后端均可;
- `10` 装饰后 kernel 的 `submit`/`async_call` 非阻塞启动(任意后端);
- `11` 完全不需要 GPU: 全部运行在纯 Python 解释器后端上。

## 目录

| 文件 | 覆盖内容 |
|---|---|
| [01_basic_kernel.py](01_basic_kernel.py) | `@kernel` 最基础用法: SAXPY, 自动 work size, 中括号切后端/设备 |
| [02_backend_device_selection.py](02_backend_device_selection.py) | `list_devices()` / `get_device()` 与设备选择语法 |
| [03_nd_loop_and_class_method.py](03_nd_loop_and_class_method.py) | 多维 `np.ndindex`/`np.ndenumerate` 与类方法/静态方法; 以及按名字调用辅助函数: 同一个类的另一个方法、任意类的静态方法、模块函数 |
| [04_vector_types.py](04_vector_types.py) | 向量类型(float4/...)、swizzle、双面函数库 `pygpgpu.funcs` |
| [05_gnp_zero_copy_and_atomics.py](05_gnp_zero_copy_and_atomics.py) | `gnp` 设备驻留数组零拷贝、原子操作直方图与本地内存归约 |
| [06_handwritten_kernels.py](06_handwritten_kernels.py) | 手写 `.cl`/`.cu`/`.hip` 文件像 Python 模块一样 `import` |
| [07_low_level_oop.py](07_low_level_oop.py) | 底层 OOP API: Platforms/Device/Context/Buffer/Program/Kernel |
| [08_compare_backends.py](08_compare_backends.py) | 同一内核在 OpenCL / CUDA / HIP 上的速度对比 |
| [09_reduction.py](09_reduction.py) | 数组求和归约: 逐元素原子、跨步私有累加、本地内存树形归约三种写法 |
| [10_async_launch.py](10_async_launch.py) | 装饰后 kernel 的非阻塞启动: `submit` / `async_call` |
| [11_python_backend.py](11_python_backend.py) | 纯 Python 后端: `k["python"]`、魔法循环与 gid 写法、`submit`/`async_call` |
| [11_returns_and_scope.py](11_returns_and_scope.py) | 返回值(以参数形式传递, 首个返回者胜出)与外层作用域变量(参数化, `nonlocal` 会写回) |
| [12_print_in_kernel.py](12_print_in_kernel.py) | kernel 里的 `print()`: 自动翻译成设备端 `printf`, 输出与 Python 完全一致的文本 |
| [13_image.py](13_image.py) | 与后端无关的 `Image` 类型: kernel 里像数组一样下标(`src[y, x]`), 在 CUDA 上翻译为 `tex2D`/`surf2Dwrite`, 在 OpenCL 上翻译为 `read_imagef`/`write_imagef`, 在 `python` 后端由宿主实现执行, 三者结果逐元素一致 |
| [14_heterogeneous_data.py](14_heterogeneous_data.py) | kernel 能承载的每一种数据, 且每种都在三种角色下演示(传参 / 体内创建 / 返回值): 标量, 1/2/3 维标量数组, 按值传递的向量, 1/2/3 维向量数组, 继承自 `ctypes.Structure` 的结构体(成员含标量 / 向量 / 数组 / 嵌套结构体 / 结构体数组), 以及 1/2/3 维结构体数组 |

`@kernel` 函数体支持 docstring(第一行的字符串字面量会被翻译器自动忽略,
不会进入生成的 GPU 源码), 所以可以放心在装饰函数里写说明文字。

## 关于自动 work size

kernel 有两种方式声明自己的迭代域, 自动 work size 需要其中之一:

- **魔法循环**, 即 `for i in range(x.shape[0]): ...`(二维/三维用 `np.ndindex`/`np.ndenumerate`), 循环边界就是迭代域;
- 对基于 `get_global_id` 的 **gid 写法** kernel, 在函数体开头声明: `work_size(x)`。这条语句只在 host 侧生效——翻译器会把它擦除, 不生成任何设备代码——并且它会贡献 `x` 的全部维度, 与 `np.ndindex(x.shape)` 的语义一致。两者都没有时, 调用会直接报错, 而不是静默猜测。

有了迭代域后, `@kernel` 依据这些 extent 推导 local size, 并可能把 global size 向上取整到 local 的整数倍(这样在 shape 尴尬时——例如长度为质数——仍能保持高效的工作组划分, 而不是退化成单个 work item)。生成的 kernel 会为每个 work-item 做越界保护, 所以对 `@kernel` 是安全的。

手写 kernel 必须自己 clamp, 而且不能用 `get_global_size(0)` 来做: 它返回的是取整后的启动规模, 不是元素个数。请像 `vecadd.cl` 那样把真实元素数作为 kernel 参数传入并据此比较。

纯 Python 后端(`k["python"]`)对两种写法保持同样的语义: 魔法循环 kernel 按原样只跑一次(循环本身已覆盖所有元素), gid 写法 kernel 按其迭代域的每个工作项跑一次。它无法模拟 `local`, 因为组内共享内存需要并行的工作项。

## 关于后端初始化

`pygpgpu` 顶层导入本身不需要任何 GPU 运行时(`climport/cuimport/hipimport`
是延迟加载的)。但**导入后端子包**(`pygpgpu.cuda` / `pygpgpu.hip`)会立即
初始化对应运行时: 缺 `nvcuda.dll` / `amdhip64.dll` 就直接抛错。因此不在
运行时的机器上, 请只跑那些在默认后端或 OpenCL 上的示例; 只有 `02`/`08`
会通过 `list_devices()` 主动探测并按运行时存在性适配, 不会因缺后端失败。
