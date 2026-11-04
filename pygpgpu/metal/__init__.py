from .driver import MTL, _start_metal_init

# The runtime is initialized on a background thread: `import pygpgpu.metal` returns
# without waiting for the framework load and the selector preheat. Every driver call
# waits for the initialization it needs (see `Func.__call__` in the driver), so nothing
# observes a half-initialized runtime. A runtime the machine lacks (any non-Apple
# machine, and Mac virtual machines) is recorded by the driver's init, not raised here.
_start_metal_init()
