import os
import re
import io
from typing import Optional, List, Tuple, Dict, Set

from .pcpp import pcmd


class CPreprocessor:

    @staticmethod
    def __short_file_name(file_name:str):
        abs_file_name = os.path.abspath(file_name).replace("\\", "/")
        try:
            rel_file_name = os.path.relpath(abs_file_name).replace("\\", "/")
        except ValueError:
            # relative paths cannot be computed across drives (e.g. the temp file is on C:, the project on E:)
            return abs_file_name
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
    def __collect_includes(file_name:str, include_paths:List[str], related:Set[str], processed:Set[str])->None:
        """Recursively collect the files involved in #include expansion.

        pcpp only emits #line directives for files that produce output lines, so
        includes containing pure macro definitions never appear in the output and
        related_files misses them (a changed include never invalidates the cache).
        processed is the dedup set, related is the output set (which may already
        contain files extracted from #line).
        """
        abs_file_name:str = os.path.abspath(file_name).replace("\\", "/")
        if abs_file_name in processed:
            return
        processed.add(abs_file_name)
        related.add(abs_file_name)

        try:
            with open(file_name, 'r', encoding='utf-8') as input:
                content:str = input.read()
        except OSError:
            return

        include_pattern = re.compile(r'^\s*#\s*include\s*(?:"(?P<quoted>[^"]+)"|<(?P<angled>[^>]+)>)', re.MULTILINE)
        search_paths:List[str] = [os.path.dirname(abs_file_name)] + list(include_paths)
        for match in include_pattern.finditer(content):
            name:str = (match.group("quoted") if match.group("quoted") is not None else match.group("angled"))
            if os.path.isabs(name):
                candidates:List[str] = [name]
            elif match.group("quoted") is not None:
                candidates = [os.path.join(path, name) for path in search_paths]
            else:
                candidates = [os.path.join(path, name) for path in include_paths]

            for candidate in candidates:
                if os.path.isfile(candidate):
                    CPreprocessor.__collect_includes(candidate, include_paths, related, processed)
                    break

    @staticmethod
    def macros_expand_file(filename: str, include_paths: Optional[List[str] ] = None, defines: Optional[Dict[str, str] ] = None)->Tuple[str, Dict[int, Tuple[str, int] ], Set[str]]:
        input = open(filename, 'r', encoding='utf-8')

        clean_code, line_map, related_files = CPreprocessor.macros_expand(input, include_paths, defines)
        # #line only yields files that produce output lines; recursively complete the full include set here
        CPreprocessor.__collect_includes(filename, include_paths or [], related_files, set())
        return clean_code, line_map, related_files