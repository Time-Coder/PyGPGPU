"""Classmethod kernels, and the receiver struct's write-back to the caller.

Three subjects, each pinned where it was wrong:

* a `@kernel`-decorated classmethod sends the *class* as the receiver -- the same
  structure-argument way an instance method sends its instance (its class-level scalar
  members are the struct fields, its ndarray members stay separate arguments);
* a body (or a device helper it calls) that assigns to a receiver member has the store
  reach the caller's own receiver -- the instance for `self.x`, the class for `cls.x`;
* `cls.<helper>(...)` is a device call like any other method: the classmethod is
  unwrapped to `__func__`, the receiver is passed as the callee's first parameter, and
  the callee's own receiver name is what its body reads its members through.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# A classmethod kernel: the class is the receiver
# ---------------------------------------------------------------------------


class _Counter:
    factor = 2.0
    version = 5

    @classmethod
    @kernel
    def scale(cls, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = cls.factor


class _WrapperCounter:
    """The other spelling: `@classmethod` above `@kernel`."""

    factor = 3.0

    @classmethod
    @kernel
    def scale(cls, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = cls.factor


def _bracketable(bound):
    """The form of a classmethod kernel whose brackets (`[backend]`) survive.

    `@kernel` outermost stores the PyKernel on the class, so the descriptor binds it
    (`_BoundKernelMethod`, brackets included). `@classmethod` outermost wraps the
    PyKernel in a MethodType, which forwards attribute access but not subscripting --
    the wrapped kernel's own `__getitem__` is reached through `__func__`.
    """
    unwrapped = getattr(bound, "__func__", bound)
    if isinstance(bound, type(unwrapped)) and hasattr(bound, "__getitem__"):
        return bound
    return unwrapped


def _spelling_variants():
    """The two spellings, each reachable through its class attribute."""
    return [_Counter.scale, _WrapperCounter.scale]


@pytest.mark.parametrize("bound", _spelling_variants(),
                         ids=["kernel-outermost", "classmethod-outermost"])
def test_the_class_is_the_receiver(bound, runner):
    out = np.zeros(4, dtype=np.float32)
    runner(_bracketable(bound), out)
    np.testing.assert_allclose(out, np.full(4, bound.__self__.factor,
                                           dtype=np.float32))


def test_the_generated_signature_names_cls(gpu_backend):
    # The hip backend's shared cache directory is polluted by other modules' sources
    # (the reader below picks the newest file in it), so the source shape is pinned
    # where the cache directory is stable. The shape itself is backend-neutral.
    if gpu_backend == "hip":
        pytest.skip("the hip cache directory mixes other modules' sources")
    out = np.zeros(4, dtype=np.float32)
    source = _source_of(_bracketable(_Counter.scale), out, backend=gpu_backend)
    assert re.search(r"\w+\* cls", source), source
    assert "(cls)->factor" in source, source


def _source_of(bound, *args, backend:str):
    # A bracketed launch writes the generated source; the cache directory the kernel
    # owns is the cheapest place to read it back from.
    inner = getattr(bound, "__func__", bound)
    cache_dir = inner._cache_dir(backend)
    import glob
    import pathlib
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    bound[backend](*args)
    extension = {"cuda": "cu", "opencl": "cl", "hip": "hip"}[backend]
    files = sorted(glob.glob(f"{cache_dir}/*.{extension}"))
    assert files, f"no generated source under {cache_dir}"
    newest = max(files, key=lambda path: pathlib.Path(path).stat().st_mtime_ns)
    return pathlib.Path(newest).read_text(encoding="utf-8")


def test_a_cls_int_member_can_be_the_loop_extent(runner):
    out = np.zeros(6, dtype=np.float32)

    class _Sized:
        width = 6
        table = np.zeros(6, dtype=np.float32)

        @kernel
        @classmethod
        def fill(cls, out: np.ndarray):
            for i in range(cls.width):
                out[i] = cls.table[i] + 1.0

    runner(_bracketable(_Sized.fill), out)
    np.testing.assert_allclose(out, np.full(6, 1.0, dtype=np.float32))


def test_a_classmethod_kernel_runs_on_the_python_backend():
    out = np.zeros(4, dtype=np.float32)
    _bracketable(_Counter.scale)["python"](out)
    np.testing.assert_allclose(out, np.full(4, 2.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# A receiver member write reaches the caller's own receiver
# ---------------------------------------------------------------------------


class _InstanceWriter:
    def __init__(self) -> None:
        self.count = 0
        self.gain = 1.0

    @kernel
    def run(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.gain
            self.count = 3


class _ClassWriter:
    version = 0

    @classmethod
    @kernel
    def run(cls, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = 1.0
            cls.version = 9


def test_an_instance_member_store_comes_home(runner):
    instance = _InstanceWriter()
    out = np.zeros(4, dtype=np.float32)
    runner(instance.run, out)
    assert instance.count == 3


def test_a_class_member_store_comes_home(runner):
    _ClassWriter.version = 0
    out = np.zeros(4, dtype=np.float32)
    runner(_bracketable(_ClassWriter.run), out)
    # python's own rule for `cls.attr = v`: the CLASS attribute is rebound, so the
    # store lands on the class, not on whatever instance (or none) the call went
    # through.
    assert _ClassWriter.version == 9


def test_the_member_store_survives_an_async_launch():
    instance = _InstanceWriter()
    out = np.zeros(4, dtype=np.float32)
    instance.run.submit(out).result()
    assert instance.count == 3


# ---------------------------------------------------------------------------
# cls.<helper>(...) from device code
# ---------------------------------------------------------------------------


class _WithClassmethodHelper:
    def __init__(self) -> None:
        self.gain = 2.0
        self.base = 10.0

    @classmethod
    def triple(cls, v):
        return v * 3.0

    @classmethod
    def scaled(cls, v):
        return cls.base * v

    @kernel
    def run(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.gain + self.triple(i)


class _WithClassmethodReader:
    unit = 5.0                # a CLASS attribute: cls.<member> reads land here

    def __init__(self) -> None:
        self.gain = 1.0

    @classmethod
    def scaled(cls, v):
        return cls.unit * v

    @kernel
    def run(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.gain + self.scaled(i)


def test_a_classmethod_helper_is_a_device_call(runner):
    instance = _WithClassmethodHelper()
    out = np.zeros(4, dtype=np.float32)
    runner(instance.run, out)
    np.testing.assert_allclose(out, 2.0 + np.arange(4, dtype=np.float32) * 3.0)


def test_a_classmethod_helper_reads_cls_members(runner):
    instance = _WithClassmethodReader()
    out = np.zeros(4, dtype=np.float32)
    runner(instance.run, out)
    np.testing.assert_allclose(out, 1.0 + np.arange(4, dtype=np.float32) * 5.0)


def test_a_classmethod_helper_on_a_classmethod_kernel(runner):
    class _Chained:
        unit = 5.0

        @classmethod
        def scaled(cls, v):
            return cls.unit * v

        @classmethod
        @kernel
        def run(cls, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = cls.scaled(i)

    out = np.zeros(4, dtype=np.float32)
    # The class is test-local, so the qualname resolution cannot find it: pass it
    # explicitly, which is the supported spelling for a class the kernel's module
    # globals do not hold.
    runner(_bracketable(_Chained.run), _Chained, out)
    np.testing.assert_allclose(out, np.arange(4, dtype=np.float32) * 5.0)


def test_a_base_class_classmethod_helper_is_reached(runner):
    class _Base:
        def __init__(self) -> None:
            self.gain = 3.0

        @classmethod
        def hook(cls, v):
            return v * 2.0

    class _Derived(_Base):
        @kernel
        def run(self, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = self.gain + self.hook(i)

    out = np.zeros(4, dtype=np.float32)
    runner(_Derived().run, out)
    np.testing.assert_allclose(out, 3.0 + np.arange(4, dtype=np.float32) * 2.0)


def test_a_device_helper_writing_cls_members_comes_home(runner):
    class _HelperWriter:
        count = 0

        @classmethod
        def put(cls, v):
            cls.count = v

        @kernel
        def run(self, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = 1.0
                self.put(7.0)

    out = np.zeros(4, dtype=np.float32)
    instance = _HelperWriter()
    runner(instance.run, out)
    # The helper's receiver is the caller's own receiver struct: the store lands in
    # it, the read-back brings it home, and the write-back reaches the caller's
    # receiver (the instance the kernel was called on).
    assert instance.count == 7


# ---------------------------------------------------------------------------
# A receiver member that is itself a python object: nested stores come home in place
# ---------------------------------------------------------------------------


def test_a_nested_ctypes_member_store_reaches_the_object(runner):
    import ctypes

    class _Inner(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float)]

    class _Outer:
        def __init__(self) -> None:
            self.inner = _Inner(1.0)

        @kernel
        def run(self, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = self.inner.x
                self.inner.x = 5.0

    instance = _Outer()
    alias = instance.inner            # python's own aliasing: one object, two names
    out = np.zeros(4, dtype=np.float32)
    runner(instance.run, out)
    # Every work item reads the member's *launch-time* value on a device (each work
    # item sees its own copy of the receiver struct), while the python backend runs
    # the body once as written -- so the out sequence differs by backend and is not
    # what this pins. The store itself is.
    np.testing.assert_allclose(out[0], 1.0)
    # python semantics for `self.inner.x = v`: the object the attribute *already
    # names* is mutated -- the attribute is not rebound -- so the alias sees the
    # update and `instance.inner` is still that same object.
    assert instance.inner is alias
    assert instance.inner.x == 5.0
    assert alias.x == 5.0


def test_a_two_level_nested_store_reaches_the_object(runner):
    import ctypes
    from pygf import float2

    class _Point(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    class _Inner(ctypes.Structure):
        _fields_ = [("point", _Point), ("v", float2)]

    class _Outer:
        def __init__(self) -> None:
            self.inner = _Inner(_Point(1.0, 2.0), float2(3.0, 4.0))
            self.tag = 7

        @kernel
        def run(self, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = self.inner.point.x + self.inner.point.y
                self.inner.point.x = 10.0
                self.inner.v.x = 20.0
                self.tag = 3

    instance = _Outer()
    out = np.zeros(4, dtype=np.float32)
    runner(instance.run, out)
    # The out sequence differs by backend for the same reason the single-level test
    # states; the members are what this pins.
    assert instance.inner.point.x == 10.0
    assert instance.inner.v.x == 20.0
    assert instance.tag == 3


def test_an_array_field_of_a_nested_member_store_comes_home(runner):
    import ctypes

    class _Row(ctypes.Structure):
        _fields_ = [("values", ctypes.c_float * 3), ("n", ctypes.c_int)]

    class _Holder:
        def __init__(self) -> None:
            self.row = _Row()
            for i, value in enumerate((1.0, 2.0, 3.0)):
                self.row.values[i] = value
            self.row.n = 3

        @kernel
        def run(self, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = self.row.values[i]
                self.row.values[0] = 9.0
                self.row.n = 5

    holder = _Holder()
    out = np.zeros(3, dtype=np.float32)
    runner(holder.run, out)
    assert out[0] == 1.0
    assert list(holder.row.values) == [9.0, 2.0, 3.0]
    assert holder.row.n == 5


# ---------------------------------------------------------------------------
# A plain `def f(cls, ...)` body spells a classmethod kernel without the wrapper
# ---------------------------------------------------------------------------


def test_a_cls_named_receiver_kernel_launches_without_the_wrapper(runner):
    class _Bare:
        unit = 4.0

        def step(cls, out: np.ndarray):
            for i in range(out.shape[0]):
                out[i] = cls.unit

    # `@kernel` over a plain method whose receiver is spelled `cls`: the descriptor
    # binds the class, the same rule the `@classmethod` wrapper states.
    _Bare.step = kernel(_Bare.step)
    out = np.zeros(4, dtype=np.float32)
    runner(_Bare.step, out)
    np.testing.assert_allclose(out, np.full(4, 4.0, dtype=np.float32))
