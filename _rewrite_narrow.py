"""Rewrite both stub writers to build their headers without .pyi.in templates."""
import pathlib

narrow = pathlib.Path(r"pygpgpu/vec_types/helper.py")
text = narrow.read_text(encoding="utf-8")
old = text[text.find("def write_clmath_pyi()"):]
new = '''def write_clmath_pyi():
    self_folder = os.path.dirname(os.path.abspath(__file__))

    for name, scalar, char_sets in VEC_INFOS:
        num:str = re.search(r'\\d+$', name).group()
        sibling:str = '4' if num == '2' else '2'
        header:str = (
            "from typing import Union\\n\\n"
            f"from pygf.genVec{num} import genVec{num}\\n"
            f"from pygf.genVec{sibling} import genVec{sibling}\\n\\n\\n"
            f"class {name}(genVec{num}):\\n\\n"
            "    def __len__(self)->int: ...\\n\\n"
            "    def __repr__(self)->str: ...\\n\\n"
            "    def __str__(self)->str: ...\\n\\n")

        with open(f"{self_folder}/{name}.pyi", "w", newline="\\n") as out_file:
            swizzle_defines:str = generate_swizzle_defines(
                name, scalar, char_sets, _NARROW_LENGTHS)
            out_file.write(header + swizzle_defines)

if __name__ == "__main__":
    write_clmath_pyi()
'''
text = text.replace(old, new)
narrow.write_text(text, encoding="utf-8", newline="")
print("narrow writer rewritten:", "def write_clmath_pyi" in text)
