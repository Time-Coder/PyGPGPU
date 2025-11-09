from typing import Optional, List


class BuildOptions:

    def __init__(self,
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
        spir_std:Optional[float]=None
    ):
        self.single_precision_constant:bool = single_precision_constant
        self.denorms_are_zero:bool = denorms_are_zero
        self.fp32_correctly_rounded_divide_sqrt:bool = fp32_correctly_rounded_divide_sqrt
        self.opt_disable:bool = opt_disable
        self.strict_aliasing:bool = strict_aliasing
        self.uniform_work_group_size:bool = uniform_work_group_size
        self.no_subgroup_ifp:bool = no_subgroup_ifp
        self.mad_enable:bool = mad_enable
        self.no_signed_zeros:bool = no_signed_zeros
        self.unsafe_math_optimizations:bool = unsafe_math_optimizations
        self.finite_math_only:bool = finite_math_only
        self.fast_relaxed_math:bool = fast_relaxed_math
        self.w:bool = w
        self.Werror:bool = Werror
        self.cl_std:Optional[float] = cl_std
        self.kernel_arg_info:bool = kernel_arg_info
        self.g:bool = g
        self.create_library:bool = create_library
        self.enable_link_options:bool = enable_link_options
        self.x_spir:bool = x_spir
        self.spir_std:Optional[float] = spir_std

    def __str__(self)->str:
        options:List[str] = []

        if self.single_precision_constant:
            options.append("-cl-single-precision-constant")

        if self.denorms_are_zero:
            options.append("-cl-denorms-are-zero")

        if self.fp32_correctly_rounded_divide_sqrt:
            options.append("-cl-fp32-correctly-rounded-divide-sqrt")

        if self.opt_disable:
            options.append("-cl-opt-disable")

        if self.strict_aliasing:
            options.append("-cl-strict-aliasing")

        if self.uniform_work_group_size:
            options.append("-cl-uniform-work-group-size")

        if self.no_subgroup_ifp:
            options.append("-cl-no-subgroup-ifp")

        if self.mad_enable:
            options.append("-cl-mad-enable")

        if self.no_signed_zeros:
            options.append("-cl-no-signed-zeros")

        if self.unsafe_math_optimizations:
            options.append("-cl-unsafe-math-optimizations")

        if self.finite_math_only:
            options.append("-cl-finite-math-only")

        if self.fast_relaxed_math:
            options.append("-cl-finite-math-only")

        if self.w:
            options.append("-w")

        if self.Werror:
            options.append("-Werror")

        if self.cl_std:
            options.append(f"-cl-std=CL{self.cl_std:.1f}")

        if self.kernel_arg_info:
            options.append("-cl-kernel-arg-info")

        if self.g:
            options.append("-g")

        if self.create_library:
            options.append("-create-library")

        if self.enable_link_options:
            options.append("-enable-link-options")

        if self.x_spir:
            options.append("-x spir")

        if self.spir_std:
            options.append(f"-spir-std={self.spir_std:.1f}")

        return " ".join(options)
