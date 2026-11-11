# The bracket-vs-domain coverage check.
from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from ..translator import (
    _unpack_count_message,
)



def _check_shape_indices(kernel_name:str, full:List[object],
                         shape_sources:List[Tuple[int, int]],
                         work_size_dims:List[Tuple[Optional[int], int]],
                         launch_meta:Dict[str, object])->None:
    """Refuse a dimension the array does not have, in the caller's own terms.

    Two different mistakes reach this function, and they have to stay apart because python
    reports them differently:

    * a magic loop whose *iteration domain* names more axes than the array has
      (``for i, j in np.ndindex(arr.shape)`` over a 1-D array) is an unpacking error:
      ``not enough values to unpack (expected 2, got 1)``. Its rank is written in the
      source while the array's shape is known only from the value passed at call time, so
      it can only be found here.
    * a *subscript* that names more indices than the array has dimensions (``a[i, i]``
      over a 1-D array) is an ``IndexError``: ``too many indices for array: array is
      1-dimensional, but 2 were indexed``. A subscript naming fewer indices than the
      dimensions (``out[i, j]`` on a 3-D array) is legal python that selects a sub-array,
      which a work item cannot represent, so that one stays refused with its own wording.

    The two used to share one check driven by ``shape_sources``, which carries both the
    loop's extents and the dimensions a subscript asks for. A body indexing ``a[i, i]``
    therefore claimed the *loop* wanted two dimensions of ``a`` and reported an unpacking
    error about a loop that never iterated ``a`` at all.
    """
    param_names:Dict[int, str] = launch_meta["param_names"]
    name_to_index:Dict[str, int] = {name: index for index, name in param_names.items()}

    def shape_of(param_index:Optional[int])->Optional[Tuple[int, ...]]:
        if param_index is None:
            return None
        shape:Optional[Tuple[int, ...]] = getattr(full[param_index], "shape", None)
        return None if shape is None else tuple(int(s) for s in shape)

    # The loop's own iteration domain, and only that: every axis it names has to exist.
    first_line:int = int(launch_meta.get("source_first_line", 1) or 1)
    for param_index, dim in sorted({(pi, d) for pi, d in work_size_dims if pi is not None}):
        if dim < 0:
            continue                      # -1 is a whole-shape domain, checked below
        shape = shape_of(param_index)
        if shape is None:
            continue
        rank:int = len(shape)
        if dim < rank:
            continue
        name = param_names.get(param_index, f"argument {param_index}")
        error:BaseException = ValueError(
            f"{kernel_name}(): {_unpack_count_message(dim + 1, rank)}: the magic loop "
            f"iterates {dim + 1} axes but '{name}' has shape {shape}, so it has "
            f"{rank} dimension{'s' if rank != 1 else ''}")
        # The loop is the statement python blames, so point there.
        loop_line:int = int(launch_meta.get("loop_line", 0) or 0)
        if loop_line:
            error.lineno = first_line + loop_line - 1  # type: ignore[attr-defined]
        raise error from None

    # A slice is a pointer at an offset, which is what python's view really is only for a
    # one-dimensional array. On a 2-D one, `a[0:2]` is two *rows*, so `part[i]` is a row
    # and not the element `a[0 + i]`; emitting the offset would answer a scalar where
    # python answers an array, silently. The rank is known here and nowhere in the
    # translator (a device pointer carries no rank), so this is where it is refused.
    for name in sorted(set(launch_meta.get("sliced_arrays") or [])):
        param_index = name_to_index.get(name, -1)
        if param_index < 0:
            continue
        shape = shape_of(param_index)
        if shape is None or len(shape) == 1:
            continue
        raise ValueError(
            f"{kernel_name}(): a slice of '{name}' names "
            f"{'rows' if len(shape) == 2 else 'sub-arrays'}, not elements: the array has "
            f"shape {shape}, so `{name}[lo:hi]` is a view of whole "
            f"{'rows' if len(shape) == 2 else 'sub-arrays'} while the device pointer it "
            f"becomes addresses elements. Slice a one-dimensional array, or index "
            f"`{name}` directly")

    # A subscript asks for *its* number of indices; that is what python counts.
    for name, count in sorted(launch_meta["subscript_indices"].items()):
        param_index:int = name_to_index.get(name, -1)
        if param_index < 0:
            continue                      # a local buffer or an image: no numpy value
        shape = shape_of(param_index)
        if shape is None:
            continue
        rank = len(shape)
        if count == rank:
            continue
        # The statement that subscripts the array is the one worth blaming, not the loop:
        # the parsed copy's line is offset by `source_first_line`, so the caret lands on
        # `a[i, i]` in the user's file, exactly where python would point.
        subscript_line:int = int(launch_meta.get("subscript_lines", {}).get(name, 0) or 0)
        first_line:int = int(launch_meta.get("source_first_line", 1) or 1)
        if count > rank:
            # numpy's own sentence and exception type, word for word: python raises
            # exactly this for `a[i, i]` on a 1-D array, so the device does too. Nothing
            # is appended -- the array is already named by the frame above, and an extra
            # clause made the two readings differ.
            error:BaseException = IndexError(
                f"too many indices for array: array is {rank}-dimensional, "
                f"but {count} were indexed")
            # What the frame stand-in needs to reproduce python's caret: the expression
            # that was too wide, and how wide the array really is.
            error.subscript_text = launch_meta.get(  # type: ignore[attr-defined]
                "subscript_texts", {}).get(name, "")
            error.array_rank = rank  # type: ignore[attr-defined]
        else:
            # Python raises nothing here -- `out[i, j]` on a 3-D array selects a sub-array --
            # so this is a device limit, not a mirror of the interpreter's. A ValueError
            # keeps it apart from the indexing errors above. A *store* through a partial
            # index is python's broadcast and is legal (see the store loop below); a read
            # has no device value to name, so it stays refused.
            error = ValueError(
                f"too few indices for '{name}': python would select a sub-array from its "
                f"shape {shape}, which a work item cannot represent. Index every dimension, "
                f"e.g. '{name}[i, j, k]' for a 3-D array")
        if subscript_line:
            # `_body_frame` reads these back to place a frame on that line, and the column
            # decides where CPython starts the caret -- on the subscript expression.
            error.lineno = first_line + subscript_line - 1  # type: ignore[attr-defined]
            error.column = int(  # type: ignore[attr-defined]
                launch_meta.get("subscript_columns", {}).get(name, 0) or 0)
        raise error from None

    # A *store* is recorded apart from a read, because its partial form is legal:
    # `image[0, 0] = value` on a 3-D array is python's broadcast, and the generated
    # store walks the sub-array's contiguous elements through the tail parameter. Only
    # the too-many direction is a mistake here -- numpy raises its own IndexError for it.
    for name, count in sorted(launch_meta.get("subscript_store_indices", {}).items()):
        param_index = name_to_index.get(name, -1)
        if param_index < 0:
            continue
        shape = shape_of(param_index)
        if shape is None:
            continue
        rank = len(shape)
        if count <= rank:
            continue
        subscript_line = int(launch_meta.get("subscript_lines", {}).get(name, 0) or 0)
        first_line = int(launch_meta.get("source_first_line", 1) or 1)
        error = IndexError(
            f"too many indices for array: array is {rank}-dimensional, "
            f"but {count} were indexed")
        error.subscript_text = launch_meta.get(  # type: ignore[attr-defined]
            "subscript_texts", {}).get(name, "")
        error.array_rank = rank  # type: ignore[attr-defined]
        if subscript_line:
            error.lineno = first_line + subscript_line - 1  # type: ignore[attr-defined]
            error.column = int(  # type: ignore[attr-defined]
                launch_meta.get("subscript_columns", {}).get(name, 0) or 0)
        raise error from None
