"""Reads a Metal Shading Language source into the per-kernel argument layout.

Metal's entry-point spelling is `kernel void f(...)`, and none of the grammars
this package depends on (OpenCL C, CUDA C) accepts that: the CUDA grammar tags
`void` as an ERROR node when `kernel` precedes it, and the OpenCL grammar has no
`kernel` qualifier at all. Adding a third grammar dependency for a language
whose declarations this package already *generates itself* would buy nothing --
the generated sources follow one shape, and a handwritten `.metal` file follows
the same shape -- so the reader here is a small, explicit declaration scanner
rather than a full parser.

What it reads is exactly what a launch needs, and nothing else:

* the entry points (`kernel void name(...)`), and
* each one's parameter names and types, including the `[[thread_position_in_grid]]`
  style attribute that marks an index argument.

A parameter's type is normalized the way MSL spells it (`device const float*`,
`uint3`, `[[attribute]]`), and the trailing attribute is stripped because the
host binds arguments by position, not by attribute.

Anything the scanner cannot make sense of raises rather than guessing: a launch
that silently bound the wrong argument would be far worse than a clear refusal.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .program_info import ArgInfo, KernelInfo

# `kernel void name(` at a declaration position. The qualifier may carry the
# `[[...]]` attributes MSL allows between it and the return type, so the match
# stops at `void` and lets the attribute text be skipped by the caller.
_KERNEL_HEAD_RE:re.Pattern = re.compile(
    r"\bkernel\s+(?:void|[\w:]+)\s+([A-Za-z_]\w*)\s*\(")

# A parameter attribute such as `[[thread_position_in_grid]]`: the bracket pair is
# stripped so the inner name can be compared against the dispatch attributes.
_ATTRIBUTE_RE:re.Pattern = re.compile(r"\[\[\s*[^\]]*\]\]")
_ATTRIBUTE_INNER_RE:re.Pattern = re.compile(r"\[\[\s*([^\]]*?)\s*\]\]")

# The address-space qualifiers MSL puts in front of a pointer type. They describe
# where the pointer lives, not what the host binds, so the type is normalized
# without them -- the same way the CUDA and OpenCL paths normalize theirs.
_ADDRESS_SPACE_WORDS:Tuple[str, ...] = (
    "device", "constant", "threadgroup", "thread", "threadgroup_imageblock",
)

# The access/type qualifiers that are likewise not part of the bound value.
_ACCESS_WORDS:Tuple[str, ...] = ("const", "volatile", "restrict", "__restrict")

# The attributes that name a *dispatch-supplied* value. Such a parameter is not
# bound by the host at all -- the threadgroup dispatch fills it in -- so it is left
# out of the argument list a launch binds. Every other parameter (a buffer, a
# scalar, an extent) is bound by position.
_DISPATCH_ATTRIBUTES:frozenset = frozenset((
    "thread_position_in_grid", "thread_position_in_threadgroup",
    "thread_index_in_threadgroup", "threadgroup_position_in_grid",
    "threadgroups_per_grid", "threads_per_grid", "threads_per_threadgroup",
    "thread_index_in_threadgroup_imageblock", "thread_position_in_threadgroup_imageblock",
    "quadgroup_index_in_threadgroup", "simdgroup_index_in_threadgroup",
    "simdgroup_count", "thread_index_in_simdgroup", "threads_per_simdgroup",
))


class ProgramParser:
    """The kernel layout of one MSL source file."""

    def __init__(self)->None:
        self._file_name:str = ""
        self._clean_code:str = ""
        self._kernel_infos:Dict[str, KernelInfo] = {}
        self._structs:Dict[str, Any] = {}
        # `kernel_import` reads this off every backend's parser to build the module's
        # `struct` members (see `_BackendSource.struct_types`). MSL structures are not
        # parsed here yet -- an MSL `struct Foo { ... }` has no host counterpart the
        # scanner builds -- so the map is empty and a handwritten source that declares
        # one simply has no exported ctypes type. The attribute exists so the shared
        # import hook does not have to special-case this backend.
        self._struct_types:Dict[str, type] = {}
        # The stub generator walks `_struct_infos` (name -> StructInfo) to declare the
        # ctypes classes in the `.pyi`. Empty for the same reason.
        self._struct_infos:Dict[str, Any] = {}

    def parse(self, file_name:str, includes:Optional[List[str]]=None,
              defines:Optional[Dict[str, Any]]=None,
              options:Optional[Any]=None)->float:
        """Read one source file and record its kernel layout.

        `options` is accepted and ignored: it carries the backend's build options
        (`kernel_import` passes them to every parser), which for Metal are the
        compile flags a `MTLCompileOptions` would hold. Nothing in this scanner
        depends on them.
        """
        self._file_name = file_name
        with open(file_name, "r", encoding="utf-8") as source:
            self._clean_code = source.read()
        self._scan(self._clean_code)
        return os.path.getmtime(file_name)

    def _scan(self, code:str)->None:
        for match in _KERNEL_HEAD_RE.finditer(code):
            name:str = match.group(1)
            parameters, _closing = self._read_parameters(code, match.end())
            kernel_info = KernelInfo(name)
            # The attribute parameters are NOT part of `KernelInfo.args`. That map is
            # the host-visible signature: `kernel_import` builds the module's call
            # signature from it, the stub generator writes one `__call__` per entry,
            # and `Kernel._bind_args` binds buffers by its order. A dispatch-supplied
            # value is filled in by the threadgroup dispatch and never bound by the
            # host, so listing it would put a `gid` parameter on a call the caller
            # cannot supply -- which is exactly what the generated stub used to show.
            # They are kept separately, because the *device* signature still declares
            # them and the launch has to skip those slots by index.
            dispatch_names:List[str] = []
            for parameter in parameters:
                argument = self._parse_parameter(parameter)
                if argument is None:
                    continue
                arg_info = ArgInfo(
                    argument["name"], argument["type"],
                    tuple(argument["array_shape"]),
                    argument["type_qualifiers"],
                    argument["kernel_flags"])
                # The type still declares this parameter, and the launch skips its slot
                # by index -- but it is not something the host binds.
                arg_info.dispatch_supplied = argument["dispatch_supplied"]
                if argument["dispatch_supplied"]:
                    dispatch_names.append(arg_info.name)
                    continue
                kernel_info.args[arg_info.name] = arg_info
            # Recorded on the KernelInfo, beside the host-visible parameters: the
            # device signature declares them at the end, so the launch binds
            # `len(args)` buffers and leaves the tail to the dispatch.
            kernel_info.dispatch_parameters = dispatch_names
            self._kernel_infos[name] = kernel_info

    @staticmethod
    def _read_parameters(code:str, start:int)->Tuple[List[str], int]:
        """Split a parameter list into top-level declarations, honouring nesting.

        The scan tracks brackets because a parameter's type may itself carry a
        pair (`float[4]`, `[[...]]`), and splitting on a comma inside those would
        break the declaration apart.
        """
        depth:int = 0
        current:List[str] = []
        parameters:List[str] = []
        for index in range(start, len(code)):
            character:str = code[index]
            if character in "([<":
                depth += 1
            elif character in ")]>":
                if depth == 0:
                    if current:
                        parameters.append("".join(current))
                    return parameters, index
                depth -= 1
            if character == "," and depth == 0:
                parameters.append("".join(current))
                current = []
                continue
            current.append(character)
        raise ValueError(f"{os.path.basename(start and 'the source')}: unterminated parameter list")

    def _parse_parameter(self, text:str)->Optional[Dict[str, Any]]:
        """One parameter declaration -> the descriptor a launch needs."""
        attributes:List[str] = [match.strip()
                                for match in _ATTRIBUTE_INNER_RE.findall(text)]
        declaration:str = _ATTRIBUTE_RE.sub(" ", text).strip()
        if not declaration or declaration == "void":
            return None
        # The name is the last identifier before any array suffix.
        array_shape:List[int] = []
        name_match = re.search(r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\])*)\s*$",
                              declaration)
        if not name_match:
            raise ValueError(f"cannot read the parameter declaration {text.strip()!r}")
        name:str = name_match.group(1)
        for extent in re.findall(r"\[\s*(\d+)\s*\]", name_match.group(2)):
            array_shape.append(int(extent))
        type_text:str = declaration[:name_match.start()].strip()
        if not type_text:
            raise ValueError(f"the parameter {name!r} declares no type")
        # A dispatch attribute means the threadgroup dispatch supplies this value,
        # so the host binds nothing for it; recording that here is what lets the
        # launch skip the slot instead of passing a buffer the GPU would overwrite.
        is_dispatch:bool = any(attribute in _DISPATCH_ATTRIBUTES
                               for attribute in attributes)
        return {
            "name": name,
            "type": self._normalized_type(type_text),
            "array_shape": array_shape,
            "type_qualifiers": [word for word in type_text.split()
                                if word in _ACCESS_WORDS],
            "kernel_flags": None,
            "dispatch_supplied": is_dispatch,
        }

    @staticmethod
    def _normalized_type(type_text:str)->str:
        """MSL's spelling reduced to the one the driver tables are keyed by.

        Address-space and access words are dropped, so `device const float*` and
        `float*` agree; a pointer keeps its `*`, and a vector keeps its spelling
        (`float4`, `uint3`), because that is a different type rather than a
        qualifier.
        """
        words:List[str] = [word for word in type_text.split()
                           if word not in _ADDRESS_SPACE_WORDS + _ACCESS_WORDS]
        normalized:str = "".join(words)
        # `float *` -> `float*`, and `const float *` -> `float*` (the const was dropped).
        normalized = re.sub(r"\s*\*\s*", "*", normalized)
        return normalized

    # ---- public surface the Program needs ----

    @property
    def file_name(self)->str:
        return self._file_name

    @property
    def clean_code(self)->str:
        return self._clean_code

    @property
    def kernel_infos(self)->Dict[str, KernelInfo]:
        return self._kernel_infos

    @property
    def structs(self)->Dict[str, Any]:
        return self._structs
