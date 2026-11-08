# The cparser package holds the parts of the legacy C toolchain that are still reachable:
# the preprocessor, the tree-sitter parser, and the one-off script that regenerates the
# atom-type tables.
#
# The rest of the legacy chain -- `cparser.py`, `ctokens.py`, `csyntaxtree.py`,
# `minifyc.py` and the generated `ShaderBuiltins.py` -- was deleted: nothing could import
# it (`cparser.py` needs `pygpgpu.GLInfo`, which does not exist, and `ShaderBuiltins.py`
# needs OpenGL, which is not a dependency), so it was unreachable dead code.
from .cpreprocessor import CPreprocessor
