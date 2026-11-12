"""The vector-type stub writer: ``python -m pygpgpu.vec_types.helper`` regenerates
the ``.pyi`` files from the ``genVecN.pyi.in`` templates.

The swizzle combinations themselves are pygf's generator -- the one implementation
serves both libraries -- and this module keeps only the presentation the stubs add:
the ``s`` prefix that turns OpenCL's numeric characters into attribute names, and
the two shapes (the swizzle sets for the narrow two/four-wide types, single
components for the 8/16-wide ones).

Two things make the output *truthful* rather than merely generated, and both are
easy to regress:

  * Only the classes whose ``.py`` lives in this package get a stub. The registry
    also names the driver classes (``char3``) and the pygf re-exports
    (``float2``); a stub for either landed beside modules it does not describe,
    with imports that do not resolve.
  * A swizzle is declared only when the runtime answers it. The narrow families
    have no host 3-wide class (their vec3s are device classes in the driver
    packages), so every three-component result is refused at runtime and none is
    declared -- including the four-character sets' three-length combos, whose
    result width is three.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from pygf.helper import generate_getter_swizzles as _pygf_getter_swizzles
from pygf.helper import generate_setter_swizzles as _pygf_setter_swizzles


def generate_getter_swizzles(char_sets:List[str], lengths:List[int])->List[str]:
    result:List[str] = []
    for char_set in char_sets:
        prefix = 's' if char_set.startswith('0') else ''
        for combo in _pygf_getter_swizzles([char_set], lengths=lengths):
            result.append(prefix + combo)

    return result

def generate_setter_swizzles(char_sets:List[str], lengths:List[int])->List[str]:
    result:List[str] = []
    for char_set in char_sets:
        prefix = 's' if char_set.startswith('0') else ''
        for combo in _pygf_setter_swizzles([char_set], lengths=lengths):
            result.append(prefix + combo)

    return result

def generate_swizzle_defines(type_name:str, scalar:str, char_sets:List[str],
                             lengths:List[int])->str:
    result:str = ""
    num:str = re.search(r'\d+$', type_name).group()
    basename:str = type_name[:-len(num)]
    getter_swizzles:List[str] = generate_getter_swizzles(char_sets, lengths)
    setter_swizzles:List[str] = generate_setter_swizzles(char_sets, lengths)
    for swizzle in getter_swizzles:
        return_type_name:str = scalar
        input_type_name:str = "Union[bool, int, float]"
        n_swizzle:int = len(swizzle)
        if swizzle[0] == 's' and swizzle[1] in '0123456789ABCDEFabcdef':
            n_swizzle -= 1

        if n_swizzle > 1:
            return_type_name = basename + str(n_swizzle)
            input_type_name = f"Union[bool, int, float, genVec{n_swizzle}]"

        result += f"""
    @property
    def {swizzle}(self)->{return_type_name}: ...
"""

        if swizzle in setter_swizzles:
            result += f"""
    @{swizzle}.setter
    def {swizzle}(self, value:{input_type_name})->None: ...
"""

    return result


# The classes whose .py lives beside this module: the narrow integer families at
# widths 2 and 4. Each entry names the scalar annotation the components answer and
# the letter sets the components are spelled by. The 8/16-wide families live beside
# the only backend that declares them (`pygpgpu/opencl/driver/vec_types/`), with
# their own copy of this table and writer.
VEC_INFOS:List[tuple] = [
    ('char2',   'int', ['xy', 'rg', '01']),
    ('char4',   'int', ['xyzw', 'rgba', '0123']),
    ('uchar2',   'int', ['xy', 'rg', '01']),
    ('uchar4',   'int', ['xyzw', 'rgba', '0123']),
    ('short2',   'int', ['xy', 'rg', '01']),
    ('short4',   'int', ['xyzw', 'rgba', '0123']),
    ('ushort2',  'int', ['xy', 'rg', '01']),
    ('ushort4',  'int', ['xyzw', 'rgba', '0123']),
]

# The swizzle lengths a stub declares. Two and four for the narrow families (a
# three-component result does not exist beside them).
_NARROW_LENGTHS:List[int] = [1, 2, 4]

# The header a stub carries, now that the width bases live in pygf: the base
# import names pygf's module, and the sibling width the swizzle results name
# (a 2-wide vector answers 4-component repeats, a 4-wide one answers
# 2-component reads -- neither names a 3-wide, which does not exist here).
_NARROW_HEADERS:Dict[int, str] = {
    2: ("from typing import Union\n\n"
        "from pygf.genVec2 import genVec2\n"
        "from pygf.genVec4 import genVec4\n\n\n"),
    4: ("from typing import Union\n\n"
        "from pygf.genVec4 import genVec4\n"
        "from pygf.genVec2 import genVec2\n\n\n"),
}

def write_clmath_pyi():
    self_folder = os.path.dirname(os.path.abspath(__file__))

    for name, scalar, char_sets in VEC_INFOS:
        num:str = re.search(r'\d+$', name).group()
        sibling:str = '4' if num == '2' else '2'
        header:str = (
            "from typing import Union\n\n"
            f"from pygf.genVec{num} import genVec{num}\n"
            f"from pygf.genVec{sibling} import genVec{sibling}\n\n\n"
            f"class {name}(genVec{num}):\n\n"
            "    def __len__(self)->int: ...\n\n"
            "    def __repr__(self)->str: ...\n\n"
            "    def __str__(self)->str: ...\n\n")

        with open(f"{self_folder}/{name}.pyi", "w", newline="\n") as out_file:
            swizzle_defines:str = generate_swizzle_defines(
                name, scalar, char_sets, _NARROW_LENGTHS)
            out_file.write(header + swizzle_defines)

if __name__ == "__main__":
    write_clmath_pyi()
