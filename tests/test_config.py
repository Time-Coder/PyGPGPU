"""`pygpgpu.config`: the one object every process-wide setting hangs off.

Each setting is a property that validates its value, so the failure a caller sees is the one
that names the setting rather than a device error three layers down. These tests pin the
round trip (what is written is what is read), the validation (which exception, with which
key/backend named), and the two shapes that are easy to get wrong: a returned list is a
copy, and assigning a mapping replaces it rather than merging into it.

No device is needed -- every assertion here is about the settings object itself.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import Config, config


@pytest.fixture(autouse=True)
def restore_settings():
    """Every test starts and ends at the built-in values, whatever it changes."""
    saved_type_map = config.type_map
    saved_device = config.default_device
    saved_print_info = config.print_info
    saved_import = config.kernel_import.options
    saved_type_checked = config.kernel_import.type_checked
    saved_generate_pyi = config.kernel_import.generate_pyi
    config.type_map = None
    config.default_device = None
    config.print_info = True
    config.kernel_import.type_checked = False
    config.kernel_import.generate_pyi = True
    config.kernel_import.options = None
    yield
    config.type_map = saved_type_map
    config.default_device = saved_device
    config.print_info = saved_print_info
    config.kernel_import.type_checked = saved_type_checked
    config.kernel_import.generate_pyi = saved_generate_pyi
    config.kernel_import.options = saved_import


# ---------------------------------------------------------------------------
# The singleton
# ---------------------------------------------------------------------------


def test_pygpgpu_publishes_one_config_instance():
    assert isinstance(config, Config)
    assert type(config) is Config


def test_config_is_not_a_module():
    """`config` is an attribute of the package, and there is no `pygpgpu.config` module."""
    import pygpgpu

    assert pygpgpu.config is config
    assert not hasattr(config, "__file__")


def test_repr_names_every_setting():
    text = repr(config)
    for name in ("print_info", "type_map", "default_device", "kernel_import"):
        assert name in text


# ---------------------------------------------------------------------------
# print_info
# ---------------------------------------------------------------------------


def test_print_info_round_trips():
    config.print_info = False
    assert config.print_info is False
    config.print_info = True
    assert config.print_info is True


def test_print_info_defaults_to_on():
    assert config.print_info is True


def test_print_info_refuses_a_non_bool():
    with pytest.raises(TypeError) as caught:
        config.print_info = 0
    assert "config.print_info" in str(caught.value)


# ---------------------------------------------------------------------------
# type_map
# ---------------------------------------------------------------------------


def test_type_map_reports_resolved_device_type_names():
    config.type_map = {float: np.float64}
    assert config.type_map == {float: "double"}


def test_type_map_defaults_to_empty():
    """Empty means "the built-in widths", not "nothing is mapped"."""
    assert config.type_map == {}


def test_type_map_takes_both_builtins():
    config.type_map = {int: np.int64, float: np.float64}
    assert config.type_map == {int: "long", float: "double"}


def test_type_map_accepts_a_ctypes_class_and_a_dtype():
    config.type_map = {int: ctypes.c_int64, float: np.dtype("float64")}
    assert config.type_map == {int: "long", float: "double"}


def test_type_map_accepts_a_device_type_name():
    config.type_map = {float: "double"}
    assert config.type_map == {float: "double"}


def test_type_map_replaces_the_whole_mapping():
    config.type_map = {int: np.int64, float: np.float64}
    config.type_map = {float: np.float32}
    assert config.type_map == {float: "float"}


@pytest.mark.parametrize("reset", [None, {}])
def test_type_map_none_and_empty_restore_the_defaults(reset):
    config.type_map = {float: np.float64}
    config.type_map = reset
    assert config.type_map == {}


def test_type_map_refuses_a_key_that_is_not_a_python_builtin():
    with pytest.raises(ValueError) as caught:
        config.type_map = {np.float32: np.float64}
    assert "int and float" in str(caught.value)


def test_type_map_refuses_a_width_the_key_cannot_take():
    with pytest.raises(ValueError) as caught:
        config.type_map = {int: np.float64}
    assert "int" in str(caught.value)


def test_type_map_refuses_a_non_mapping():
    with pytest.raises(TypeError) as caught:
        config.type_map = [float, np.float64]
    assert "config.type_map" in str(caught.value)


# ---------------------------------------------------------------------------
# default_device
# ---------------------------------------------------------------------------


def test_default_device_reports_the_probe_order_until_it_is_set():
    assert config.default_device == ["cuda", "hip", "metal", "opencl"]


def test_default_device_accepts_a_single_selector():
    config.default_device = "cuda:0"
    assert config.default_device == ["cuda:0"]


def test_default_device_accepts_a_list_in_order():
    config.default_device = ["opencl:cpu", "cuda"]
    assert config.default_device == ["opencl:cpu", "cuda"]


def test_default_device_accepts_a_tuple():
    config.default_device = ("cuda", "hip:0")
    assert config.default_device == ["cuda", "hip:0"]


def test_default_device_none_restores_the_probe_order():
    config.default_device = ["cuda:0"]
    config.default_device = None
    assert config.default_device == ["cuda", "hip", "metal", "opencl"]


def test_default_device_reports_a_copy():
    """Mutating what the getter returned must not change the setting."""
    config.default_device = ["cuda:0"]
    reported = config.default_device
    reported.append("opencl:0")
    assert config.default_device == ["cuda:0"]


def test_default_device_refuses_a_non_selector():
    with pytest.raises(TypeError) as caught:
        config.default_device = 3
    assert "config.default_device" in str(caught.value)


def test_default_device_refuses_an_entry_that_is_not_a_string():
    with pytest.raises(TypeError) as caught:
        config.default_device = ["cuda:0", 3]
    assert "selector" in str(caught.value)


def test_default_device_refuses_an_empty_selector():
    with pytest.raises(ValueError) as caught:
        config.default_device = ""
    assert "empty" in str(caught.value)


# ---------------------------------------------------------------------------
# kernel_import
# ---------------------------------------------------------------------------


def test_kernel_import_defaults():
    settings = config.kernel_import
    assert settings.type_checked is False
    assert settings.generate_pyi is True
    assert settings.options == {"cuda": None, "hip": None, "metal": None, "opencl": None}


def test_kernel_import_flags_round_trip():
    config.kernel_import.type_checked = True
    config.kernel_import.generate_pyi = False
    assert config.kernel_import.type_checked is True
    assert config.kernel_import.generate_pyi is False


@pytest.mark.parametrize("name", ["type_checked", "generate_pyi"])
def test_kernel_import_flags_refuse_a_non_bool(name):
    with pytest.raises(TypeError) as caught:
        setattr(config.kernel_import, name, "yes")
    assert "config.kernel_import.%s" % name in str(caught.value)


def test_kernel_import_options_replace_wholesale():
    config.kernel_import.options = {"cuda": "build-options"}
    assert config.kernel_import.options == {
        "cuda": "build-options", "hip": None, "metal": None, "opencl": None}


def test_kernel_import_options_none_clears_them():
    config.kernel_import.options = {"cuda": "build-options"}
    config.kernel_import.options = None
    assert config.kernel_import.options == {
        "cuda": None, "hip": None, "metal": None, "opencl": None}


def test_kernel_import_options_report_a_copy():
    config.kernel_import.options = {"cuda": "build-options"}
    reported = config.kernel_import.options
    reported["cuda"] = "something else"
    assert config.kernel_import.options["cuda"] == "build-options"


def test_kernel_import_options_refuse_an_unknown_backend():
    with pytest.raises(ValueError) as caught:
        config.kernel_import.options = {"tpu": "build-options"}
    assert "tpu" in str(caught.value)


def test_kernel_import_options_refuse_a_non_mapping():
    with pytest.raises(TypeError) as caught:
        config.kernel_import.options = ["cuda"]
    assert "config.kernel_import.options" in str(caught.value)


def test_kernel_import_object_has_a_stable_identity():
    """``config.kernel_import`` is one nested object, not a fresh wrapper per read."""
    assert config.kernel_import is config.kernel_import
