from .driver import CUDA, NVRTC
from .cuimport import cuimport

CUDA.init()
NVRTC.init()

cuimport.install()
