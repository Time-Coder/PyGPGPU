import traceback

import numpy as np

from pygpgpu import kernel


@kernel
def _probe(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i]


def test_probe() -> None:
    import importlib
    PC = importlib.import_module('pygpgpu.kernel.decorator.pykernel_call')
    original = PC._user_traceback

    def loud(error, func):
        import traceback as tb
        tb.print_exception(type(error), error, error.__traceback__)
        return original(error, func)

    PC._user_traceback = loud
    try:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = np.zeros(3, dtype=np.float32)
        _probe(x, out)
    finally:
        PC._user_traceback = original
