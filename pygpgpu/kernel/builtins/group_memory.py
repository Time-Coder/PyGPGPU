from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from ..python_backend import PythonFrame
    from . import _IndexVector



def barrier() -> None:
    """Wait for every work item of the group. A serial run has nothing to wait for."""
    return None


def local_array_per_group(shape: object, dtype: object = "float32") -> object:
    """Work-group shared memory: a zeroed array of ``shape``/``dtype``, one slice per group.

    The arguments are ``np.zeros``' own, in its own order: the **shape first** (an int or a
    tuple of ints) and the **dtype second** (defaulting to float32, as an array of no declared
    dtype would be). So ``local_array_per_group(256, np.int32)`` and
    ``local_array_per_group((64, 64), np.float32)`` read like the numpy call they replace.

    This is CUDA's ``__shared__`` / OpenCL's ``__local``, and the value stands for **all
    groups' slices at once**, keyed by group:

    * on the **host** it really is a mapping -- block id -> that group's array -- so
      ``buf[blockIdx]`` selects the slice, and the model is honest about what a shared
      buffer is;
    * on the **device** no indexing happens at all: the emitted code is a plain
      ``__shared__`` array (CUDA/HIP) or ``__local`` parameter (OpenCL), which the hardware
      already gives each group privately, so ``buf[blockIdx][j]`` folds to ``buf[j]``.

    That is why the translator requires the block id for the first subscript but emits
    nothing for it: the index is a statement about *which* group's memory is meant, and
    inside a work item only one answer is reachable.

    A body that is one work item of one known block (a gid-style kernel) has no such
    mapping to spell: it writes ``buf[j]`` directly, which is what ``local_array`` builds.
    Both spellings emit the same declaration on the device; they differ in the host model,
    because a magic-loop body stands for the whole launch while a gid-style body stands for
    one work item.

    The shape has to be a **compile-time constant**: the buffer is a fixed-size
    ``__shared__`` array (CUDA/HIP) or one ``__local`` parameter (OpenCL), so a size known
    only at launch cannot declare it.

    On the interpreter backend this returns the mapping itself, with each group's array
    created on first use, so a body that reads ``buf[blockIdx][j]`` works unchanged and a
    group never sees another group's storage. A serial run has no real parallelism, so a
    ``barrier()`` cannot order anything -- a body that *depends* on another work item having
    written first still gets a serial answer, which is the answer a device run would give
    only if the ordering happened to be the same.

    ``shared_array_per_block`` is the same buffer under CUDA's word for a group. The two are
    separate functions on purpose: one name, one signature, so an editor highlights a call to
    either as a call to that name rather than to an alias of the other.
    """
    return _group_slices(shape, dtype, "local_array_per_group")


def shared_array_per_block(shape: object, dtype: object = "float32") -> object:
    """CUDA's spelling of `local_array_per_group`: the same buffer, one slice per block.

    ``shared_array_per_block(256, np.int32)`` and
    ``local_array_per_group(256, np.int32)`` build the same device declaration and are
    indexed the same way -- by the group first, ``buf[blockIdx][j]`` -- because "block" is
    CUDA's word for an OpenCL work group. Both names are published so a CUDA reader finds
    the one they look for; the shapes, the dtype rule and the host model are documented on
    `local_array_per_group`.

    This is a function of its own rather than an alias: an alias is the same object under a
    second name, which an editor shows as a reference to the first, and a kernel that means
    to call the CUDA spelling then reads as if it called the OpenCL one.
    """
    return _group_slices(shape, dtype, "shared_array_per_block")


def _group_slices(shape: object, dtype: object, what: str) -> object:
    """The host value both per-group spellings answer: a mapping of every group's slice."""
    from ..host_runtime import require_frame

    frame: object = require_frame(what)
    return _GroupSlices(shape, dtype, frame, what)


class _GroupSlices:
    """One group's zeroed array per block id, created on demand.

    Subscripting is the whole point -- ``buf[blockIdx]`` -- and a block id outside the launch
    is refused rather than silently allocated, because that would hand out storage for a
    group the launch does not have.
    """

    __slots__ = ("_shape", "_dtype", "_frame", "_what", "_slices")

    def __init__(self, shape: Union[int, Tuple[int, ...]], dtype: DTypeLike,
                 frame: "PythonFrame", what: str)->None:
        if isinstance(shape, int):
            shape = (shape, )
        self._shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self._dtype: DTypeLike = dtype
        self._frame: "PythonFrame" = frame
        self._what: str = what
        self._slices: Dict[Tuple[int, ...], "np.ndarray[Any, Any]"] = {}

    def __getitem__(self, group: Union[int, Tuple[int, ...], "_IndexVector"])->"np.ndarray[Any, Any]":
        import numpy as np

        key: Tuple[int, ...] = _group_key(group, self._what)
        groups: List[int] = []
        for axis in range(len(self._frame.global_size)):
            local: int = self._frame.local_size[axis] if axis < len(self._frame.local_size) else 1
            groups.append(-(-self._frame.global_size[axis] // max(1, local)))
        for axis, count in enumerate(groups):
            if axis < len(key) and key[axis] >= count:
                raise IndexError(
                    f"no work group {key} in this launch: it has {tuple(groups)} group(s), so "
                    f"a {self._what}() buffer has no storage for it")
        if key not in self._slices:
            self._slices[key] = np.zeros(self._shape, dtype=self._dtype)
        return self._slices[key]

    def __repr__(self)->str:
        return f"<{self._what}{self._shape} dtype={self._dtype}>"


def _group_key(group: Union[int, Sequence[int], "_IndexVector"], what: str)->Tuple[int, ...]:
    """The block id a `buf[blockIdx]` subscript names, as a tuple of axis indices."""
    if isinstance(group, int):
        return (int(group), )
    # `blockIdx`/`get_group_id(0)` and a plain list of axes are all acceptable: what matters
    # is the three numbers, not the container.
    # The index-vector class lives in the package facade, which imports this module:
    # the import is made where it is used, so the two do not import each other at
    # module level.
    from . import _IndexVector
    if isinstance(group, _IndexVector):
        from ..host_runtime import get_host_frame

        frame: object = get_host_frame()
        axes: int = len(frame.global_size) if frame is not None else 3
        return tuple(group[axis] for axis in range(max(1, axes)))
    try:
        return tuple(int(v) for v in group)
    except TypeError:
        raise TypeError(
            f"a {what}() buffer is indexed by a work group, not by "
            f"{type(group).__name__}: use `buf[blockIdx]`") from None
