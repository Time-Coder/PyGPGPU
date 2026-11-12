"""The driver for the damaged-file repro: plant, launch, and report.

Split from the repro module because the repro is data (a kernel + its base class) and this
is the procedure: write the module's generated file minus its forward-declaration block,
leave the digest sidecar honest, launch, and report what happened -- including whether the
file healed itself.
"""
from __future__ import annotations

import re
from pathlib import Path

_MARKER = "/* pygpgpu: device functions called here but defined elsewhere */"

_DECLARATION = re.compile(r"^\s*(?:__device__ )?\w[\w ]*\*?\s*\w+\([^)]*\);\s*$")


def generated_files(backend: str) -> list[Path]:
    """The generated files the repro's kernel lives in, for one backend.

    Only the *generated source* files match -- the ones this package writes from a render.
    HIP's binary-cache folder (`<cache>/pykernels/__hipcache__/`) holds roc-clang
    intermediates that are preprocessed copies (pcpp strips comments, so the marker never
    survives in them); they are compiled artifacts, not generated source, and the
    machinery regenerates them from the source on every compile. The backend's own cache
    root (`<backend-cache>/pykernels`) carries the same folder name, so the test is the
    directory *below* `pykernels`.
    """
    root = Path(__file__).parent / {backend: "__cucache__", "opencl": "__clcache__",
                                    "hip": "__hipcache__"}[backend]
    folder = root / "pykernels"
    if not folder.is_dir():
        return []
    hits = [p for p in folder.rglob("*.cl") if "damaged_generated_file" in p.name]
    hits += [p for p in folder.rglob("*.cu") if "damaged_generated_file" in p.name]
    hits += [p for p in folder.rglob("*.hip") if "damaged_generated_file" in p.name
             and "__hipcache__" not in p.parent.relative_to(folder).parts]
    return hits


def strip_prototype_block(path: Path) -> int:
    """Remove the forward-declaration block, leaving the digest sidecar honest."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    removed = 0
    skipping = False
    for line in lines:
        if _MARKER in line:
            skipping = True
            removed += 1
            continue
        if skipping and _DECLARATION.match(line):
            removed += 1
            continue
        skipping = False
        out.append(line)
    if removed:
        path.write_text("".join(out), encoding="utf-8")
    return removed


def marker_present(path: Path) -> bool:
    """Whether the forward-declaration block is in the file."""
    return _MARKER in path.read_text(encoding="utf-8")
