from .driver import HIP
from .driver import _start_hip_init

# The runtime is initialized on a background thread: `import pygpgpu.hip` returns without
# waiting for the DLL load and the `hipInit` preheat. Every driver call waits for the
# initialization it needs (see `Func.__call__` in the driver), so nothing observes a
# half-initialized runtime. A runtime the machine lacks is recorded by the driver's init,
# not raised here.
_start_hip_init()
