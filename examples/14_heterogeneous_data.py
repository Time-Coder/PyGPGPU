"""14_heterogeneous_data.py -- every kind of value a kernel takes, builds and returns.

A kernel boundary carries the same kinds of data python works with, and this file walks that
matrix one row at a time. Each row is tried in all three roles: as an argument, as an object
the body builds, and as a return value.

    scalars               int / float / bool / char
    1/2/3-D arrays        of scalars
    vectors               float3, by value
    1/2/3-D arrays        of vectors
    structures            members may be scalars, vectors, arrays, other structures,
                          and arrays of structures
    1/2/3-D arrays        of structures

A structure is a numpy record dtype -- ``np.dtype([("i", "<i4"), ("f", "<f4")])`` -- so the
same dtype describes the host array and the device type. Member access keeps numpy's
spelling: ``s["i"]``, ``s["inner"]["a"]``, and ``s["v"]["x"]`` for a vector member.

The functions here are deliberately trivial; only the shapes of the data are interesting.

Run:
    python examples/14_heterogeneous_data.py
"""
import numpy as np

from pygpgpu import kernel, float3, get_default_device, set_print_info

# A structure with one member of each kind it may hold.
PAIR = np.dtype([("i", "<i4"), ("f", "<f4")])
INNER = np.dtype([("a", "<i4"), ("b", "<i4")])
RICH = np.dtype([
    ("tag", "<i4"),                     # a scalar
    ("v", float3.dtype),                # a vector
    ("xs", "<f4", (3,)),                # a fixed-size array
    ("inner", INNER),                   # another structure
    ("rows", PAIR, (2,)),               # an array of structures
])


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


@kernel
def scalar_arguments(count: int, scale: float, flag: bool, letter: np.int8, out: np.ndarray):
    """`letter` is a byte: a kernel has no strings, and a char is what a byte-valued
    argument binds as."""
    for i in range(out.shape[0]):
        out[i] = count * scale + (1.0 if flag else 0.0) + (1.0 if letter == 121 else 0.0)


# ---------------------------------------------------------------------------
# Arrays of scalars, one to three dimensions
# ---------------------------------------------------------------------------


@kernel
def array_1d(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * 2.0


@kernel
def array_2d(x: np.ndarray, out: np.ndarray):
    for i, j in np.ndindex(x.shape[0], x.shape[1]):
        out[i, j] = x[i, j] + 1.0


@kernel
def array_3d(x: np.ndarray, out: np.ndarray):
    for i, j, k in np.ndindex(x.shape[0], x.shape[1], x.shape[2]):
        out[i, j, k] = x[i, j, k] - 1.0


# ---------------------------------------------------------------------------
# Vectors, by value and in arrays
# ---------------------------------------------------------------------------


@kernel
def vector_argument(offset: float3, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = offset.x + offset.y + offset.z


@kernel
def vector_array_1d(points: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = points[i].z


@kernel
def vector_array_2d(points: np.ndarray, out: np.ndarray):
    for i, j in np.ndindex(points.shape[0], points.shape[1]):
        out[i, j] = points[i, j].y


@kernel
def vector_array_3d(points: np.ndarray, out: np.ndarray):
    for i, j, k in np.ndindex(points.shape[0], points.shape[1], points.shape[2]):
        out[i, j, k] = points[i, j, k].x


# ---------------------------------------------------------------------------
# Structures: by value, and in 1/2/3-D arrays
# ---------------------------------------------------------------------------


@kernel
def structure_argument(record: np.void, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = record["tag"] + record["v"]["x"] + record["xs"][1]
        out[i] = out[i] + record["inner"]["a"] + record["rows"][0]["i"]


@kernel
def structure_array_1d(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["i"]


@kernel
def structure_array_2d(rows: np.ndarray, out: np.ndarray):
    for i, j in np.ndindex(rows.shape[0], rows.shape[1]):
        out[i, j] = rows[i, j]["i"]


@kernel
def structure_array_3d(rows: np.ndarray, out: np.ndarray):
    for i, j, k in np.ndindex(rows.shape[0], rows.shape[1], rows.shape[2]):
        out[i, j, k] = rows[i, j, k]["i"]


# ---------------------------------------------------------------------------
# Building each kind inside the body
# ---------------------------------------------------------------------------


@kernel
def builds_arrays(x: np.ndarray, out: np.ndarray):
    """Above the loop the host builds the object; below it, each work item gets its own.

    `private` is not a usable name here: CUDA and HIP compile as C++, where it is a
    keyword.
    """
    shared = np.zeros(4, dtype=np.float32)          # built on the host, one per launch
    for i in range(out.shape[0]):
        scratch = np.zeros((2, 2), dtype=np.float32)  # built on the device, one per item
        shared[i % 4] = shared[i % 4] + x[i] * 0.5
        scratch[1, 1] = x[i] + 3.0
        out[i] = shared[i % 4] + scratch[1, 1]


@kernel
def builds_structures(x: np.ndarray, out: np.ndarray):
    host_rows = np.zeros(2, dtype=PAIR)              # host-built array of structures
    host_rows[0]["i"] = 10
    for i in range(out.shape[0]):
        local_rows = np.zeros(2, dtype=PAIR)         # device-private array of structures
        local_rows[1]["i"] = int(x[i])
        out[i] = host_rows[0]["i"] + local_rows[1]["i"]


@kernel
def builds_vectors(x: np.ndarray, out: np.ndarray):
    points = np.zeros(2, dtype=float3.dtype)         # host-built array of vectors
    points[0] = float3(1.0, 2.0, 3.0)
    for i in range(out.shape[0]):
        scratch = np.zeros(2, dtype=float3.dtype)    # device-private array of vectors
        scratch[0] = float3(x[i], x[i] + 1.0, x[i] + 2.0)
        out[i] = points[0].z + scratch[0].z


# ---------------------------------------------------------------------------
# Returning each kind
# ---------------------------------------------------------------------------


@kernel
def returns_scalar(x: np.ndarray):
    for i in range(x.shape[0]):
        return x[i] * 10.0


@kernel
def returns_array(x: np.ndarray):
    doubled = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(doubled.shape[0]):
        doubled[i] = x[i] * 2.0
    return doubled


@kernel
def returns_vector(x: np.ndarray):
    for i in range(x.shape[0]):
        return float3(x[i], x[i] + 1.0, x[i] + 2.0)


@kernel
def returns_structure(x: np.ndarray, row: np.void):
    for i in range(x.shape[0]):
        return row


# ---------------------------------------------------------------------------


def _show(label: str, value: object):
    print(f"{label:<34} {value}")


def main() -> None:
    # The backends log every buffer copy and every launch. Picking a device is what imports
    # a backend, so the log is switched off after that rather than at module level -- a flag
    # set before the import would not reach the backend that just started up.
    get_default_device()
    set_print_info(False)

    # --- scalars -----------------------------------------------------------
    out = np.zeros(2, dtype=np.float32)
    scalar_arguments(3, 2.5, True, np.int8(121), out)
    _show("scalars", out.tolist())

    # --- arrays of scalars -------------------------------------------------
    out = np.zeros(3, dtype=np.float32)
    array_1d(np.arange(3, dtype=np.float32), out)
    _show("1-D scalar array", out.tolist())

    square = np.arange(4, dtype=np.float32).reshape(2, 2)
    out = np.zeros((2, 2), dtype=np.float32)
    array_2d(square, out)
    _show("2-D scalar array", out.tolist())

    cube = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = np.zeros((2, 2, 2), dtype=np.float32)
    array_3d(cube, out)
    _show("3-D scalar array", out[0, 0].tolist())

    # --- vectors -----------------------------------------------------------
    out = np.zeros(1, dtype=np.float32)
    vector_argument(float3(1.0, 2.0, 3.0), out)
    _show("vector by value", float(out[0]))

    points = np.array([float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0)])
    out = np.zeros(2, dtype=np.float32)
    vector_array_1d(points, out)
    _show("1-D vector array", out.tolist())

    grid = np.array([[float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0)],
                     [float3(7.0, 8.0, 9.0), float3(1.0, 1.0, 1.0)]])
    out = np.zeros((2, 2), dtype=np.float32)
    vector_array_2d(grid, out)
    _show("2-D vector array", out.tolist())

    cube_points = np.zeros((2, 2, 2), dtype=float3.dtype)
    cube_points[...] = (7.0, 8.0, 9.0)
    out = np.zeros((2, 2, 2), dtype=np.float32)
    vector_array_3d(cube_points, out)
    _show("3-D vector array", out[1, 1].tolist())

    # --- structures --------------------------------------------------------
    record = np.zeros(1, dtype=RICH)[0]
    record["tag"] = 100
    record["v"] = (1.0, 2.0, 3.0)
    record["xs"][1] = 20.0
    record["inner"]["a"] = 300
    record["rows"][0]["i"] = 4000
    out = np.zeros(1, dtype=np.float32)
    structure_argument(record, out)
    _show("structure by value", float(out[0]))

    rows = np.zeros(2, dtype=PAIR)
    rows["i"] = [1, 2]
    out = np.zeros(2, dtype=np.float32)
    structure_array_1d(rows, out)
    _show("1-D structure array", out.tolist())

    rows = np.zeros((2, 2), dtype=PAIR)
    rows["i"] = [[1, 2], [3, 4]]
    out = np.zeros((2, 2), dtype=np.float32)
    structure_array_2d(rows, out)
    _show("2-D structure array", out.tolist())

    rows = np.zeros((2, 2, 2), dtype=PAIR)
    rows["i"] = np.arange(1, 9).reshape(2, 2, 2)
    out = np.zeros((2, 2, 2), dtype=np.float32)
    structure_array_3d(rows, out)
    _show("3-D structure array", out[1, 1].tolist())

    # --- built inside the body --------------------------------------------
    out = np.zeros(4, dtype=np.float32)
    builds_arrays(np.arange(4, dtype=np.float32), out)
    _show("arrays built in the body", out.tolist())

    out = np.zeros(2, dtype=np.float32)
    builds_structures(np.array([5.0, 6.0], np.float32), out)
    _show("structures built in the body", out.tolist())

    out = np.zeros(2, dtype=np.float32)
    builds_vectors(np.array([1.0, 2.0], np.float32), out)
    _show("vectors built in the body", out.tolist())

    # --- returned ----------------------------------------------------------
    _show("returned scalar", returns_scalar(np.array([1.5], np.float32)))

    returned = returns_array(np.array([1.0, 2.0, 3.0], np.float32))
    _show("returned array", returned.tolist())

    vector = returns_vector(np.array([1.0], np.float32))
    _show("returned vector", (vector.x, vector.y, vector.z))

    structure = returns_structure(np.zeros(1, np.float32), record)
    _show("returned structure", (structure["tag"], structure["inner"]["a"]))


if __name__ == "__main__":
    main()
