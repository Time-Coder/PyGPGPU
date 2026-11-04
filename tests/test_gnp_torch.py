"""PyTorch interop for gnp arrays.

torch is an optional dependency, so the whole module skips when it is missing. What is
pinned here is the contract the bridge promises:

* a **CPU** tensor and a host-resident gnp array can be *one* buffer, so a write through
  either side is visible in the other;
* a **device** tensor cannot share its storage with a host array, so the request raises
  instead of silently copying;
* ``to_torch`` pulls a device-resident array home first, because its host bytes are stale
  by design.
"""
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu.numpy as gnp

torch = pytest.importorskip("torch")


def test_from_torch_copies_by_default():
    tensor = torch.arange(8, dtype=torch.float32)
    array = gnp.ndarray.from_torch(tensor)
    assert isinstance(array, gnp.ndarray)
    np.testing.assert_array_equal(array.tolist(), tensor.tolist())
    # A copy, not a view: changing the tensor must not move the array.
    tensor[0] = 100.0
    assert float(array[0]) == 0.0


def test_from_torch_zero_copy_shares_the_tensors_storage():
    tensor = torch.arange(8, dtype=torch.float32)
    array = gnp.ndarray.from_torch(tensor, zero_copy=True)
    assert array.view(np.ndarray).__array_interface__["data"][0] == tensor.data_ptr()
    tensor[1] = 42.0
    assert float(array[1]) == 42.0
    array[2] = 7.0
    assert float(tensor[2]) == 7.0


def test_from_torch_rejects_a_strided_tensor():
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4).T
    assert not tensor.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        gnp.ndarray.from_torch(tensor)


def test_from_torch_rejects_zero_copy_for_a_device_tensor():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA-capable torch")
    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    with pytest.raises(NotImplementedError, match="zero-copy"):
        gnp.ndarray.from_torch(tensor, zero_copy=True)


def test_from_torch_stages_a_device_tensor_through_the_host():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA-capable torch")
    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    array = gnp.ndarray.from_torch(tensor)
    np.testing.assert_array_equal(array.tolist(), tensor.cpu().tolist())


def test_to_torch_copies_by_default():
    array = gnp.arange(8, dtype=np.float32)
    tensor = array.to_torch()
    np.testing.assert_array_equal(tensor.tolist(), array.tolist())
    tensor[0] = 5.0
    assert float(array[0]) == 0.0, "the default is a copy, not a shared buffer"


def test_to_torch_zero_copy_shares_the_arrays_storage():
    array = gnp.arange(8, dtype=np.float32)
    tensor = array.to_torch(zero_copy=True)
    assert tensor.data_ptr() == array.view(np.ndarray).__array_interface__["data"][0]
    tensor[0] = 11.0
    assert float(array[0]) == 11.0


def test_to_torch_pulls_a_device_resident_array_home(gpu_backend):
    """A device result has stale host bytes, so the tensor must not be built from them."""
    from pygpgpu import kernel

    @kernel
    def bump(x: np.ndarray) -> None:
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0

    array = gnp.arange(8, dtype=np.float32)
    bump[gpu_backend](array)
    tensor = array.to_torch()
    np.testing.assert_allclose(tensor.tolist(), (np.arange(8) + 1).tolist())


def test_to_torch_zero_copy_refuses_a_device_bound_array(gpu_backend):
    array = gnp.arange(8, dtype=np.float32)
    array.to_device(gpu_backend, wait=True)
    with pytest.raises(NotImplementedError, match="zero_copy=False"):
        array.to_torch(zero_copy=True)


def test_torch_round_trip_survives_a_kernel(gpu_backend):
    """tensor -> array -> kernel -> array -> tensor is the whole point of the bridge."""
    from pygpgpu import kernel

    @kernel
    def bump(x: np.ndarray) -> None:
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0

    source = torch.arange(8, dtype=torch.float32)
    array = gnp.ndarray.from_torch(source, zero_copy=True)
    bump[gpu_backend](array)
    result = array.to_torch()
    np.testing.assert_allclose(result.tolist(), (np.arange(8) + 1).tolist())