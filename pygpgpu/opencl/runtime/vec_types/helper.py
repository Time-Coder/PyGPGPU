from __future__ import annotations
import itertools
from typing import List, Set, Dict
from types import ModuleType
import importlib
import os
import re


_module_map:Dict[str, ModuleType] = {}

def from_import(module_name:str, attr_name:str)->type:
    if module_name not in _module_map:
        module = None
        if module_name.startswith("."):
            module = importlib.import_module(module_name, package=__package__)
        else:
            module = importlib.import_module(module_name)
        _module_map[module_name] = module
    
    return getattr(_module_map[module_name], attr_name)

def generate_getter_swizzles(char_sets:List[str])->Set[str]:
    result:List[str] = []
    for char_set in char_sets:
        prefix = 's' if char_set.startswith('0') else ''
        for length in range(1, 4 + 1):
            for combo in itertools.product(char_set, repeat=length):
                swizzle = prefix + ''.join(combo)
                result.append(swizzle)
    
    return result

def generate_setter_swizzles(char_sets:List[str])->Set[str]:
    result:List[str] = []
    for char_set in char_sets:
        prefix = 's' if char_set.startswith('0') else ''
        for length in range(1, 4 + 1):
            if length > len(char_set):
                continue

            for combo in itertools.permutations(char_set, length):
                swizzle = prefix + ''.join(combo)
                result.append(swizzle)
    
    return result

def generate_swizzle_defines(type_name:str, dtype_name:str, char_sets:List[str], short:bool=True)->str:
    result:str = ""
    num:str = re.search(r'\d+$', type_name).group()
    basename:str = type_name[:-len(num)]
    if short:
        getter_swizzles:Set[str] = generate_getter_swizzles(char_sets)
        setter_swizzles:Set[str] = generate_setter_swizzles(char_sets)
        for swizzle in getter_swizzles:
            return_type_name:str = dtype_name
            input_type_name:str = "Union[bool, int, float]"
            n_swizzle = len(swizzle)
            if swizzle[0] == 's' and swizzle[1] in '0123456789ABCDEFabcdef':
                n_swizzle -= 1

            if n_swizzle > 1:
                return_type_name:str = basename + str(n_swizzle)
                input_type_name:str = f"Union[bool, int, float, genVec{n_swizzle}]"

            result += f"""
    @property
    def {swizzle}(self)->{return_type_name}: ...
"""
            
            if swizzle in setter_swizzles:
                result += f"""
    @{swizzle}.setter
    def {swizzle}(self, value:{input_type_name})->None: ...
"""
    else:
        return_type_name:str = dtype_name
        input_type_name:str = "Union[bool, int, float]"
        for char_set in char_sets:
            for char in char_set:
                if char in '0123456789ABCDEFabcdef':
                    char = 's' + char

                result += f"""
    @property
    def {char}(self)->{return_type_name}: ...
"""
            
                result += f"""
    @{char}.setter
    def {char}(self, value:{input_type_name})->None: ...
"""
            
    return result


def write_clmath_pyi():
    self_folder = os.path.dirname(os.path.abspath(__file__))
    vec_infos = [
        ('char2', 'str', ['xy', 'rg', '01']),
        ('char3', 'str', ['xyz', 'rgb', '012']),
        ('char4', 'str', ['xyzw', 'rgba', '0123']),
        ('char8', 'str', ['xyzw', 'rgba', '01234567']),
        ('char16', 'str', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('uchar2', 'byte', ['xy', 'rg', '01']),
        ('uchar3', 'byte', ['xyz', 'rgb', '012']),
        ('uchar4', 'byte', ['xyzw', 'rgba', '0123']),
        ('uchar8', 'byte', ['xyzw', 'rgba', '01234567']),
        ('uchar16', 'byte', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('short2', 'int', ['xy', 'rg', '01']),
        ('short3', 'int', ['xyz', 'rgb', '012']),
        ('short4', 'int', ['xyzw', 'rgba', '0123']),
        ('short8', 'int', ['xyzw', 'rgba', '01234567']),
        ('short16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('ushort2', 'int', ['xy', 'rg', '01']),
        ('ushort3', 'int', ['xyz', 'rgb', '012']),
        ('ushort4', 'int', ['xyzw', 'rgba', '0123']),
        ('ushort8', 'int', ['xyzw', 'rgba', '01234567']),
        ('ushort16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('int2', 'int', ['xy', 'rg', '01']),
        ('int3', 'int', ['xyz', 'rgb', '012']),
        ('int4', 'int', ['xyzw', 'rgba', '0123']),
        ('int8', 'int', ['xyzw', 'rgba', '01234567']),
        ('int16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('uint2', 'int', ['xy', 'rg', '01']),
        ('uint3', 'int', ['xyz', 'rgb', '012']),
        ('uint4', 'int', ['xyzw', 'rgba', '0123']),
        ('uint8', 'int', ['xyzw', 'rgba', '01234567']),
        ('uint16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('long2', 'int', ['xy', 'rg', '01']),
        ('long3', 'int', ['xyz', 'rgb', '012']),
        ('long4', 'int', ['xyzw', 'rgba', '0123']),
        ('long8', 'int', ['xyzw', 'rgba', '01234567']),
        ('long16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('ulong2', 'int', ['xy', 'rg', '01']),
        ('ulong3', 'int', ['xyz', 'rgb', '012']),
        ('ulong4', 'int', ['xyzw', 'rgba', '0123']),
        ('ulong8', 'int', ['xyzw', 'rgba', '01234567']),
        ('ulong16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('float2', 'int', ['xy', 'rg', '01']),
        ('float3', 'int', ['xyz', 'rgb', '012']),
        ('float4', 'int', ['xyzw', 'rgba', '0123']),
        ('float8', 'int', ['xyzw', 'rgba', '01234567']),
        ('float16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
        ('double2', 'int', ['xy', 'rg', '01']),
        ('double3', 'int', ['xyz', 'rgb', '012']),
        ('double4', 'int', ['xyzw', 'rgba', '0123']),
        ('double8', 'int', ['xyzw', 'rgba', '01234567']),
        ('double16', 'int', ['xyzw', 'rgba', '0123456789ABCDEFabcdef']),
    ]

    pyi_in_contents:Dict[str, str] = {}

    for vec_inf in vec_infos:
        num:str = re.search(r'\d+$', vec_inf[0]).group()
        basename:str = vec_inf[0][:-len(num)]
        in_file_name:str = f"{self_folder}/genVec{num}.pyi.in"
        if in_file_name not in pyi_in_contents:
            pyi_in_contents[in_file_name] = open(in_file_name).read()

        with open(f"{self_folder}/{vec_inf[0]}.pyi", "w") as out_file:
            swizzle_defines:str = generate_swizzle_defines(*vec_inf, short=(int(num) <= 4))
            pyi_in_content = pyi_in_contents[in_file_name].format(basename=basename)
            out_file.write(pyi_in_content + swizzle_defines)

if __name__ == "__main__":
    write_clmath_pyi()