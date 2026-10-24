from typing import Optional, List


class BuildOptions:
    """hip(clang 子进程)编译选项; 额外选项原样透传 clang。"""

    def __init__(self, extra_options:Optional[List[str]]=None, fast_math:bool=False, debug:bool=False):
        self.extra_options:List[str] = list(extra_options or [])
        self.fast_math:bool = fast_math
        self.debug:bool = debug

    def to_list(self)->List[str]:
        options:List[str] = list(self.extra_options)
        if self.fast_math:
            options.append("-ffast-math")
        if self.debug:
            options.extend(["-O0", "-g"])
        return options

    def __str__(self)->str:
        return " ".join(self.to_list())
