import os
import re
import io
from typing import Optional, List, Tuple, Dict, Set

from .pcpp import pcmd


class CPreprocessor:

    @staticmethod
    def __short_file_name(file_name:str):
        abs_file_name = os.path.abspath(file_name).replace("\\", "/")
        rel_file_name = os.path.relpath(abs_file_name).replace("\\", "/")
        if len(rel_file_name) > len(abs_file_name):
            return abs_file_name
        else:
            return rel_file_name

    @staticmethod
    def __process_line_directives(code:str):
        line_pattern = re.compile(r'^\s*#line\s+(\d+)\s+(?:"(?P<file_name>[^"]*)"|(\S+)).*$', re.MULTILINE)
        lines = code.split('\n')
        processed_lines = []
        line_mapping = {}
        current_file = None
        current_original_line = 1
        processed_line_number = 1
        
        for line in lines:
            match = line_pattern.match(line)
            if match:
                original_line = int(match.group(1))
                file_name = match.group(2) or match.group(3) or ""
                current_file = file_name
                current_original_line = original_line
            else:
                processed_lines.append(line)
                if current_file is not None:
                    line_mapping[processed_line_number] = (CPreprocessor.__short_file_name(current_file), current_original_line)
                processed_line_number += 1
                current_original_line += 1

        related_files = set()
        def add_related_file(match:re.Match):
            file_name:str = match.group("file_name")
            file_name:str = os.path.abspath(file_name).replace("\\", "/")
            related_files.add(file_name)
            return ""
        
        line_pattern.sub(add_related_file, code)
        
        return '\n'.join(processed_lines), line_mapping, related_files

    @staticmethod
    def macros_expand(file, include_paths: Optional[List[str] ] = None, defines: Optional[Dict[str, str] ] = None)->Tuple[str, Dict[int, Tuple[str, int] ], Set[str]]:
        output = io.StringIO()

        if defines is None:
            defines = {}

        if include_paths is None:
            include_paths = []

        cmds = ["", ""]

        for path in include_paths:
            cmds.append(f"-I{path}")

        for name, value in defines.items():
            arg = f"-D{name}"
            if value is not None:
                arg += f"={value}"
            cmds.append(arg)

        pcmd.CmdPreprocessor(cmds, file, output)
        return CPreprocessor.__process_line_directives(output.getvalue())

    @staticmethod
    def macros_expand_code(code: str, include_paths: Optional[List[str] ] = None, defines: Optional[Dict[str, str] ] = None)->Tuple[str, Dict[int, Tuple[str, int] ], Set[str]]:
        input = io.StringIO(code)
        input.name = "<string>"

        return CPreprocessor.macros_expand(input, include_paths, defines)

    @staticmethod
    def macros_expand_file(filename: str, include_paths: Optional[List[str] ] = None, defines: Optional[Dict[str, str] ] = None)->Tuple[str, Dict[int, Tuple[str, int] ], Set[str]]:
        input = open(filename, 'r', encoding='utf-8')

        return CPreprocessor.macros_expand(input, include_paths, defines)