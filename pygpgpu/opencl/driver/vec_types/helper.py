"""The stub writer for the 8/16-wide vector classes this package defines.

``python -m pygpgpu.opencl.driver.vec_types.helper`` regenerates the ``.pyi``
files from the ``genVec8``/``genVec16`` templates. The swizzle generation is
pygf's (one implementation for both libraries) through the shared helpers in
``pygpgpu.vec_types.helper``; this module keeps only the table -- the classes
whose ``.py`` lives here -- and the one shape those stubs declare, single
components only (``s0`` and, on the 16-wide classes, the hex spellings ``sA``).
"""
from __future__ import annotations

import os
import re
from typing import List

from pygpgpu.vec_types.helper import generate_swizzle_defines

# The classes whose .py lives beside this module, with the scalar annotation
# their components answer and the letter sets the components are spelled by.
VEC_INFOS:List[tuple] = [
    ('char8',    'int', ['xyzw', 'rgba', '01234567']),
    ('char16',   'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('uchar8',   'int', ['xyzw', 'rgba', '01234567']),
    ('uchar16',  'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('short8',   'int', ['xyzw', 'rgba', '01234567']),
    ('short16',  'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('ushort8',  'int', ['xyzw', 'rgba', '01234567']),
    ('ushort16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('int8',     'int', ['xyzw', 'rgba', '01234567']),
    ('int16',    'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('uint8',    'int', ['xyzw', 'rgba', '01234567']),
    ('uint16',   'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('long8',    'int', ['xyzw', 'rgba', '01234567']),
    ('long16',   'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('ulong8',   'int', ['xyzw', 'rgba', '01234567']),
    ('ulong16',  'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('float8',   'float', ['xyzw', 'rgba', '01234567']),
    ('float16',  'float', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ('double8',  'float', ['xyzw', 'rgba', '01234567']),
    ('double16', 'float', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
]

# Only the single components: the 8/16-wide stubs never declared swizzle
# results, and nothing reads one in a kernel (a kernel refuses the width).
_SINGLE_LENGTHS:List[int] = [1]


def write_wide_pyi() -> None:
    self_folder:str = os.path.dirname(os.path.abspath(__file__))

    for name, scalar, char_sets in VEC_INFOS:
        num:str = re.search(r"\d+$", name).group()
        header:str = (
            "from typing import Union\n\n"
            f"from .genVec{num} import genVec{num}\n\n\n"
            f"class {name}(genVec{num}):\n\n"
            "    def __len__(self)->int: ...\n\n"
            "    def __repr__(self)->str: ...\n\n"
            "    def __str__(self)->str: ...\n\n")

        with open(f"{self_folder}/{name}.pyi", "w", newline="\n") as out_file:
            swizzle_defines:str = generate_swizzle_defines(
                name, scalar, char_sets, _SINGLE_LENGTHS)
            out_file.write(header + swizzle_defines)


if __name__ == "__main__":
    write_wide_pyi()
