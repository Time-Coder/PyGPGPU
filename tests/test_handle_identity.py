"""Bugs in the handle wrappers' identity: hashing a NULL handle, and the instance registry.

A backend handle is a `c_void_p`, and a failed creation leaves it NULL. `__hash__` returned
`.value`, which is `None` for that case, and python then refuses the object outright:

    TypeError: __hash__ method should return an integer

The failure is not hypothetical -- `CUObject.__init__` itself hashes the new object to record
it in the instance registry, so a NULL handle could not even be constructed.
"""
from __future__ import annotations

from ctypes import c_void_p

import pytest

from pygpgpu.cuda.oop.cuobject import CUObject as CudaObject
from pygpgpu.hip.oop.hipobject import HIPObject
from pygpgpu.opencl.oop.clobject import CLObject


class _CudaHandle(CudaObject):
    """A concrete CUObject: the abstract protocol has no release to run."""

    @staticmethod
    def _prefix() -> str:
        return "CU_DEVICE_ATTRIBUTE"

    @staticmethod
    def _get_info_func() -> object:
        return None

    @staticmethod
    def _info_enum() -> object:
        return None

    @staticmethod
    def _release_func() -> object:
        return None


class _HipHandle(HIPObject):
    @staticmethod
    def _prefix() -> str:
        return "hipDeviceAttribute"

    @staticmethod
    def _get_info_func() -> object:
        return None

    @staticmethod
    def _info_enum() -> object:
        return None

    @staticmethod
    def _release_func() -> object:
        return None


class _ClHandle(CLObject):
    @staticmethod
    def _prefix() -> str:
        return "CL_DEVICE"

    @staticmethod
    def _get_info_func() -> object:
        return None

    @staticmethod
    def _info_types_map() -> object:
        return None

    @staticmethod
    def _info_enum() -> object:
        return None

    @staticmethod
    def _release_func() -> object:
        return None


@pytest.mark.parametrize("wrapper", [_CudaHandle, _HipHandle, _ClHandle],
                         ids=["cuda", "hip", "opencl"])
def test_hashing_a_null_handle_is_an_integer(wrapper):
    """`c_void_p(None)` must hash to a number, not to `None`.

    Every backend wrapper shares the shape: `isinstance(self._id, c_void_p)` then
    `return self._id.value`, and `.value` of a NULL pointer is `None`.
    """
    handle = wrapper.__new__(wrapper)
    handle._id = c_void_p(None)

    value = hash(handle)

    assert isinstance(value, int), f"hash answered {value!r} for a NULL handle"


@pytest.mark.parametrize("wrapper", [_CudaHandle, _HipHandle, _ClHandle],
                         ids=["cuda", "hip", "opencl"])
def test_a_null_handle_can_be_put_in_a_set(wrapper):
    """The practical consequence: `{handle}` raised TypeError before."""
    handle = wrapper.__new__(wrapper)
    handle._id = c_void_p(None)
    assert handle in {handle}


def test_a_real_handle_still_hashes_to_its_address():
    """The control: a non-NULL handle keeps hashing to its numeric address."""
    handle = _CudaHandle.__new__(_CudaHandle)
    handle._id = c_void_p(0x1234)
    assert hash(handle) == 0x1234


def test_constructing_with_a_null_id_does_not_raise():
    """`CUObject.__init__` hashes the object to register it, so the NULL case hit it there.

    A driver that answers a NULL handle on a creation the library does not consider fatal
    (a stale binary's missing kernel, an argument that could not be bound) would otherwise
    crash with a message about `__hash__` -- nowhere near the real cause.
    """
    handle = _CudaHandle(c_void_p(None))
    assert hash(handle) == 0
