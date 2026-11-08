"""The device tests, split out of `test_metal_launch_path.py`.

The subject is device.
"""
from __future__ import annotations



def test_a_device_key_distinguishes_two_identical_gpus() -> None:
    """`unique_key` is a cache key, so the name alone cannot be it.

    `unique_key` is what a compiled kernel is cached under per device. A Mac with two
    of the same GPU reports the same `name` for both, so a name-only key would let
    one device reuse the other's build. Every other backend folds an identity fact
    into the key for this reason; Metal's is `registryID`.
    """
    from pygpgpu.utils import sanitize_filename

    class FakeDevice:
        def __init__(self, name: str, registry_id: int) -> None:
            self.name = name
            self.registry_id = registry_id

    def key(device: FakeDevice) -> str:
        return sanitize_filename(f"{device.name} {device.registry_id}")

    first = FakeDevice("Apple M2", 0x1234)
    second = FakeDevice("Apple M2", 0x5678)
    again = FakeDevice("Apple M2", 0x1234)

    assert key(first) != key(second), "two identical GPUs must not share a key"
    assert key(first) == key(again), "one GPU must answer the same key every time"
    # The name stays readable, which is what the key is compared by a human.
    assert "Apple M2" in key(first)


def test_the_device_reports_a_registry_id() -> None:
    """`registryID` is the identity fact the key is built from.

    It is an `MTLDevice` selector, so the Device class has to expose it next to the
    other device queries rather than inventing an identifier of its own.
    """
    import inspect

    import pygpgpu.metal.oop.devices as metal_devices

    assert hasattr(metal_devices.Device, "registry_id")
    # `inspect.getsource` needs the getter: the attribute is a `property`, not a
    # function, so passing the descriptor itself raises TypeError.
    assert "registryID" in inspect.getsource(
        metal_devices.Device.registry_id.fget)
    # And `unique_key` folds it in rather than using the bare name.
    key_source = inspect.getsource(metal_devices.Device.unique_key.fget)
    assert "registry_id" in key_source
