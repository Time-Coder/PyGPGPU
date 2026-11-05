"""Stub generation for modules imported through `kernel_import`.

A merged module's stub is the union of its backends' declarations: one
`class Kernel_<name>(KernelImportWrapper)` per kernel, one `__call__`/`submit`/
`async_call` per distinct parameter list (marked `@overload` when there is more than one),
`__getitem__` overloads covering the device string and the bracket ranks, and one
`<name>: Kernel_<name>` line per kernel at the end. Structures come from the first file
that declares each one. The result is written next to the module's first source, so an
editor completes `vecops.add_offset(a, b, out, offset, n)` just as it does for a kernel
imported from a single-backend file.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


_STUB_HEADER = """from typing import override, overload, Tuple
import ctypes
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.kernel_import import KernelImportWrapper

"""

_GETITEM_OVERLOADS = """    @overload
    def __getitem__(self, device: str)->{class_name}: ...

    @overload
    def __getitem__(self, work_sizes: Tuple[int, int])->{class_name}: ...

    @overload
    def __getitem__(self, work_sizes: Tuple[Tuple[int], Tuple[int]])->{class_name}: ...

    @overload
    def __getitem__(self, work_sizes: Tuple[Tuple[int, int], Tuple[int, int]])->{class_name}: ...

    @overload
    def __getitem__(self, work_sizes: Tuple[Tuple[int, int, int], Tuple[int, int, int]])->{class_name}: ...

"""

_INTEGER_TYPES:Tuple[str, ...] = ("char", "uchar", "short", "ushort", "int", "uint",
                                  "long", "ulong")


def _annotation_for(type_str:str, backend:str)->str:
    """The annotation one parsed parameter carries, in the backend stubs' vocabulary."""
    from .kernel_import import _scalar_tables

    tables:Dict[str, object] = _scalar_tables(backend)
    stripped:str = type_str.strip()
    pointer:bool = stripped.endswith("*")
    base:str = (stripped[:-1].strip() if pointer else stripped).strip()

    if pointer:
        dtype_name:str = base
        scalar:object = tables.get("basic", {}).get(base)
        if scalar is not None:
            import numpy as np
            dtype_name = "np." + np.dtype(scalar).name
        return f"NDArray[{dtype_name}]"

    if base in _INTEGER_TYPES:
        return "int"
    if base in ("float", "double"):
        # A scalar is spelled `float`, as the backend stubs do: a caller passes a plain
        # Python number, and np.float32/np.float64 would refuse it.
        return "float"
    return base


def _distinct_parameter_lists(
        declarations:Dict[str, Tuple[Tuple[str, str], ...]]
        )->List[Tuple[str, List[Tuple[str, str]]]]:
    """Collapse equal parameter lists across backends, in declaration order.

    Two backends declaring the same list share one overload; the note names the backends
    behind each one, so the stub can show which implementation a bracket reaches.
    """
    from .kernel_import import normalized_signature

    lists:List[List[Tuple[str, str]]] = []
    normalized_lists:List[Tuple[Tuple[str, str], ...]] = []
    notes:List[str] = []
    for backend, signature in declarations.items():
        parameters:List[Tuple[str, str]] = list(signature)
        # Compare normalized spellings: OpenCL's `__global const float*` and CUDA's
        # `const float*` are one parameter list, not two overloads.
        normalized:Tuple[Tuple[str, str], ...] = normalized_signature(parameters)
        for index, existing in enumerate(normalized_lists):
            if existing == normalized:
                notes[index] += f", {backend}"
                break
        else:
            lists.append(parameters)
            normalized_lists.append(normalized)
            notes.append(backend)
    return list(zip(notes, lists))


def _call_block(parameters:List[Tuple[str, str]], backend_note:str, overload:bool)->str:
    """The three launch declarations for one parameter list."""
    # The annotation's backend only decides how a scalar type is spelled; the parameter
    # list itself came from that backend's declaration.
    backend:str = backend_note.split(",")[0].strip()
    arg_list:str = ", ".join(f"{name}: {_annotation_for(type_str, backend)}"
                             for name, type_str in parameters)
    if overload:
        note:str = f"  # {backend_note}" if backend_note else ""
        call:str = f"    @overload\n    def __call__(self, {arg_list})->None: ...{note}\n\n"
        submit:str = ("    @overload\n    def submit(self, "
                      f"{arg_list})->concurrent.futures.Future: ...\n\n")
        async_call:str = ("    @overload\n    def async_call(self, "
                          f"{arg_list})->asyncio.Future: ...\n\n")
    else:
        call = f"    @override\n    def __call__(self, {arg_list})->None: ...\n\n"
        submit = ("    @override\n    def submit(self, "
                  f"{arg_list})->concurrent.futures.Future: ...\n\n")
        async_call = ("    @override\n    def async_call(self, "
                      f"{arg_list})->asyncio.Future: ...\n\n")
    return call + submit + async_call


def generate_pyi_for(sources:Dict[str, str], backends:Dict[str, object],
                     kernels:Dict[str, object])->None:
    """Write `<stem>.pyi` next to the module's first source (probe order)."""
    from .kernel_import import _BACKEND_EXTENSIONS

    first_source:str = next(sources[backend] for backend, _ in _BACKEND_EXTENSIONS
                            if backend in sources)
    pyi_file_name:str = first_source.rsplit(".", 1)[0] + ".pyi"

    content:str = _STUB_HEADER

    # Structures, from the first backend that declares each one.
    seen_structs:set = set()
    for backend, _extension in _BACKEND_EXTENSIONS:
        source:Optional[object] = backends.get(backend)
        if source is None:
            continue
        for struct_name, struct_info in source.parser._struct_infos.items():
            if struct_name in seen_structs:
                continue
            seen_structs.add(struct_name)
            content += struct_info.declare()

    for kernel_name, wrapper in kernels.items():
        class_name:str = f"Kernel_{kernel_name}"
        declared:Dict[str, Tuple[Tuple[str, str], ...]] = wrapper.declarations()
        # The probe order decides which overload a bare call reaches first, so it is the
        # order the declarations are listed in.
        ordered:Dict[str, Tuple[Tuple[str, str], ...]] = {
            backend: declared[backend]
            for backend, _extension in _BACKEND_EXTENSIONS
            if backend in declared
        }
        overloads:List[Tuple[str, List[Tuple[str, str]]]] = _distinct_parameter_lists(ordered)

        content += f"class {class_name}(KernelImportWrapper):\n"
        for backend_note, parameters in overloads:
            content += _call_block(parameters, backend_note, overload=len(overloads) > 1)
        content += _GETITEM_OVERLOADS.format(class_name=class_name)
        content += "\n"

    for kernel_name in kernels:
        content += f"{kernel_name}: Kernel_{kernel_name}\n"

    with open(pyi_file_name, "w", encoding="utf-8") as out_file:
        out_file.write(content)
