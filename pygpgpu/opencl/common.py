from typing import Any, Dict, List, Optional

from .oop.program_wrapper import ProgramWrapper
from .oop.build_options import BuildOptions
from .oop.program_parser import ProgramParser


__program_wrappers:Dict[str, ProgramWrapper] = {}


def compile(
    file_name:str,
    includes:Optional[List[str]] = None,
    defines:Optional[Dict[str, Any]] = None,
    single_precision_constant:bool=False,
    denorms_are_zero:bool=False,
    fp32_correctly_rounded_divide_sqrt:bool=False,
    opt_disable:bool=False,
    strict_aliasing:bool=False,
    uniform_work_group_size:bool=False,
    no_subgroup_ifp:bool=False,
    mad_enable:bool=False,
    no_signed_zeros:bool=False,
    unsafe_math_optimizations:bool=False,
    finite_math_only:bool=False,
    fast_relaxed_math:bool=False,
    w:bool=False,
    Werror:bool=False,
    cl_std:Optional[float]=None,
    kernel_arg_info:bool=False,
    g:bool=False,
    create_library:bool=False,
    enable_link_options:bool=False,
    x_spir:bool=False,
    spir_std:Optional[float]=None,
    type_checked:bool=False,
    options:Optional[BuildOptions]=None,
    generate_pyi:bool=False
)->ProgramWrapper:
    if options is None:
        options:BuildOptions = BuildOptions(
            single_precision_constant,
            denorms_are_zero,
            fp32_correctly_rounded_divide_sqrt,
            opt_disable,
            strict_aliasing,
            uniform_work_group_size,
            no_subgroup_ifp,
            mad_enable,
            no_signed_zeros,
            unsafe_math_optimizations,
            finite_math_only,
            fast_relaxed_math,
            w,
            Werror,
            cl_std,
            kernel_arg_info,
            g,
            create_library,
            enable_link_options,
            x_spir,
            spir_std
        )

    key:str = ProgramParser.md5_of(includes, defines, options, file_name=file_name)
    if key not in __program_wrappers:
        program_wrapper = ProgramWrapper(file_name, includes, defines, options, type_checked)
        if generate_pyi:
            program_wrapper._program_parser.generate_pyi()
            
        __program_wrappers[key] = program_wrapper

    return __program_wrappers[key]
