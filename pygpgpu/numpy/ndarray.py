from typing import Dict, Tuple

import numpy as np

from ..opencl.oop import Buffer


class ndarray(np.ndarray):

    def __init__(self, *args, **kwargs):
        np.ndarray.__init__(self, *args, **kwargs)

        self._buffers:Dict[Tuple[str, int], Buffer] = {}