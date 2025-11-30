from typing import Optional, List, Optional
from ..driver import GPUArch, CppStd, OptInfoKind


class BuildOptions:

    def __init__(self,
        gpu_architecture: Optional[GPUArch] = None, arch: Optional[GPUArch] = None,
        device_c:Optional[bool] = None, dc:Optional[bool] = None,
        device_w:Optional[bool] = None, dw:Optional[bool] = None,
        relocatable_device_code:Optional[bool] = None, rdc:Optional[bool] = None,
        extensible_whole_program:Optional[bool] = None, ewp:Optional[bool] = None,
        device_debug:Optional[bool] = None, G:Optional[bool] = None,
        generate_line_info:Optional[bool] = None, lineinfo:Optional[bool] = None,
        ptxas_options:Optional[List[str]] = None, Xptxas:Optional[List[str]] = None,
        maxrregcount:Optional[int] = 0, ftz:Optional[bool] = None,
        prec_sqrt:Optional[bool] = None, prec_div: Optional[bool] = None,
        fmad:Optional[bool] = None, use_fast_math: Optional[bool] = None,
        extra_device_vectorization:Optional[bool] = None, modify_stack_limit: Optional[bool] = None,
        std:Optional[CppStd] = None, builtin_move_forward:Optional[bool] = None,
        builtin_initializer_list:Optional[bool] = None,
        disable_warnings:Optional[bool] = None, w:Optional[bool] = None,
        restrict:Optional[bool] = None, 
        device_as_default_execution_space:Optional[bool] = None, default_device:Optional[bool] = None,
        optimization_info:Optional[OptInfoKind] = None, opt_info:Optional[OptInfoKind] = None,
        version_ident:Optional[bool] = None, dQ:Optional[bool] = None,
        display_error_number:Optional[bool] = None, err_no:Optional[bool] = None,
        diag_error:Optional[List[int]] = None, diag_suppress:Optional[List[int]] = None,
        diag_warn:Optional[List[int]] = None
    ):
        self.gpu_architecture:Optional[GPUArch] = gpu_architecture
        self.arch:Optional[GPUArch] = arch
        self.device_c:Optional[bool] = device_c
        self.dc:Optional[bool] = dc
        self.device_w:Optional[bool] = device_w
        self.dw:Optional[bool] = dw
        self.relocatable_device_code:Optional[bool] = relocatable_device_code
        self.rdc:Optional[bool] = rdc
        self.ewp:Optional[bool] = ewp
        self.extensible_whole_program:Optional[bool] = extensible_whole_program
        self.device_debug:Optional[bool] = device_debug
        self.G:Optional[bool] = G
        self.generate_line_info:Optional[bool] = generate_line_info
        self.lineinfo:Optional[bool] = lineinfo
        self.ptxas_options:Optional[List[str]] = ptxas_options
        self.Xptxas:Optional[List[str]] = Xptxas
        self.maxrregcount:Optional[int] = maxrregcount
        self.ftz:Optional[bool] = ftz
        self.prec_sqrt:Optional[bool] = prec_sqrt
        self.prec_div:Optional[bool] = prec_div
        self.fmad:Optional[bool] = fmad
        self.use_fast_math:Optional[bool] = use_fast_math
        self.extra_device_vectorization:Optional[bool] = extra_device_vectorization
        self.modify_stack_limit:Optional[bool] = modify_stack_limit
        self.std:Optional[CppStd] = std
        self.builtin_move_forward:Optional[bool] = builtin_move_forward
        self.builtin_initializer_list:Optional[bool] = builtin_initializer_list
        self.disable_warnings:Optional[bool] = disable_warnings
        self.w:Optional[bool] = w
        self.restrict:Optional[bool] = restrict
        self.device_as_default_execution_space:Optional[bool] = device_as_default_execution_space
        self.default_device:Optional[bool] = default_device
        self.optimization_info:Optional[OptInfoKind] = optimization_info
        self.opt_info:Optional[OptInfoKind] = opt_info
        self.version_ident:Optional[bool] = version_ident
        self.dQ:Optional[bool] = version_ident
        self.display_error_number:Optional[bool] = display_error_number
        self.diag_error:Optional[List[int]] = diag_error
        self.err_no:Optional[bool] = display_error_number
        self.diag_suppress:Optional[List[int]] = diag_suppress
        self.diag_warn:Optional[List[int]] = diag_warn

    def to_list(self)->List[str]:
        options:List[str] = []

        if self.gpu_architecture is not None:
            options.append(f"--gpu-architecture={self.gpu_architecture.name}")

        if self.arch is not None:
            options.append(f"-arch={self.arch.name}")

        if self.device_c is not None:
            options.append("--device-c")

        if self.dc is not None:
            options.append("-dc")

        if self.device_w is not None:
            options.append("-dw")

        if self.relocatable_device_code is not None:
            options.append(f"--relocatable-device-code={str(self.relocatable_device_code).lower()}")

        if self.rdc is not None:
            options.append(f"-rdc={str(self.rdc).lower()}")

        if self.extensible_whole_program is not None:
            options.append("--extensible-whole-program")

        if self.ewp is not None:
            options.append("-ewp")

        if self.device_debug is not None:
            options.append("--device-debug")

        if self.G is not None:
            options.append("-G")

        if self.generate_line_info is not None:
            options.append("--generate-line-info")
        
        if self.lineinfo is not None:
            options.append("-lineinfo")

        if self.ptxas_options is not None:
            for option in self.ptxas_options:
                options.append(f"--ptxas-options {option}")

        if self.Xptxas is not None:
            for option in self.Xptxas:
                options.append(f"-Xptxas {option}")

        if self.maxrregcount is not None:
            options.append(f"--maxrregcount={self.maxrregcount}")

        if self.ftz is not None:
            options.append(f"--ftz={str(self.ftz).lower()}")

        if self.prec_sqrt is not None:
            options.append(f"--prec-sqrt={str(self.prec_sqrt).lower()}")

        if self.prec_div is not None:
            options.append(f"--prec-div={str(self.prec_div).lower()}")

        if self.fmad is not None:
            options.append(f"--fmad={str(self.fmad).lower()}")

        if self.use_fast_math is not None:
            options.append(f"--use-fast-math")

        if self.extra_device_vectorization is not None:
            options.append(f"--extra-device-vectorization")

        if self.modify_stack_limit is not None:
            options.append(f"--modify-stack-limit")

        if self.std is not None:
            options.append(f"--std={self.std.name.replace("cpp", "c++")}")

        if self.builtin_move_forward is not None:
            options.append(f"--builtin-move-forward={str(self.builtin_move_forward).lower()}")

        if self.builtin_initializer_list is not None:
            options.append(f"--builtin-initializer-list={str(self.builtin_initializer_list).lower()}")

        if self.disable_warnings is not None:
            options.append(f"--disable-warnings")

        if self.w is not None:
            options.append(f"-w")

        if self.restrict is not None:
            options.append(f"--restrict")

        if self.device_as_default_execution_space is not None:
            options.append(f"--device-as-default-execution-space")

        if self.default_device is not None:
            options.append(f"-default-device")

        if self.optimization_info is not None:
            options.append(f"--optimization-info={self.optimization_info.name}")

        if self.opt_info is not None:
            options.append(f"-opt-info={self.opt_info.name}")

        if self.version_ident is not None:
            options.append(f"--version-ident={str(self.version_ident).lower()}")

        if self.dQ is not None:
            options.append(f"-dQ={str(self.dQ).lower()}")

        if self.display_error_number is not None:
            options.append(f"--display-error-number")

        if self.diag_error is not None:
            options.append(f"--diag-error={','.join(self.diag_error)}")

        if self.diag_suppress is not None:
            options.append(f"--diag-suppress={','.join(self.diag_suppress)}")

        if self.diag_warn is not None:
            options.append(f"--diag-warn={','.join(self.diag_warn)}")

        return options
    
    def __str__(self)->str:
        return " ".join(self.to_list())
