# from pygpgpu.cuda import CUDA
# from pygpgpu.cuda.oop import Device, Devices

# CUDA.print_call = True

# for device in Devices:
#     print(device.default_context.api_version)

import tree_sitter_opencl as tscl
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tscl.language())
parser = Parser(PY_LANGUAGE)

code = open("test.cl").read()
ast = parser.parse(code.encode("utf-8"))
print(ast)