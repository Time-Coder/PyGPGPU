from .driver import CUDA, NVRTC
from .driver import _start_cuda_init, _start_nvrtc_init

# The driver and the compiler are initialized on background threads: `import pygpgpu.cuda`
# returns without waiting for the DLL load and the `cuInit` preheat, which overlap with
# whatever the importing thread does next. Every driver call waits for the initialization
# it needs (see `Func.__call__` in the driver), so nothing observes a half-initialized
# runtime. A runtime the machine lacks is recorded by the driver's init, not raised here.
_start_cuda_init()
_start_nvrtc_init()
