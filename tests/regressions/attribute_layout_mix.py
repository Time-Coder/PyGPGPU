"""A receiver struct whose members mix widths: the layout the launch packs must be the
layout the C compiler computes.

Found by the promo renderer's uniform set -- ``i_resolution`` float2, ``i_time`` float,
``i_frame`` int, ``i_mouse`` float4 as instance attributes of the viewer class. The
receiver struct is synthesized on the host with `ctypes` alignment (a vector member sits
at any 4-byte offset) while the compiler lays the same declaration out by the *device*
rules (``float4`` is 16-byte aligned on CUDA, so it moved to offset 16): every member
after the first vector read the wrong bytes, and the int member's own slot was the only
one that answered. The same scramble hit any struct where a narrower member precedes a
wider-aligned one -- class attributes and instance attributes read by one kernel
misbound against each other the same way.

The synthesized structure is now placed with `device_member_extent`'s table and explicit
padding members, and the rendered C declaration carries the same padding, so one layout
computation serves the packer, the parser and the compiler.
"""
from __future__ import annotations


from pygpgpu import kernel
from pygf import float2, float4


class ViewerLike:
    """The shadertoy viewer's uniform set: float2, float, int, float4, in that order."""

    def __init__(self) -> None:
        self.i_resolution = float2(1280.0, 720.0)
        self.i_time = 4.0
        self.i_frame = 7
        self.i_mouse = float4(1.0, 2.0, 3.0, 4.0)

    @kernel
    def read(self, out):
        out[0] = self.i_resolution.x + self.i_resolution.y
        out[1] = self.i_time
        out[2] = float(self.i_frame)
        out[3] = self.i_mouse.x + self.i_mouse.y + self.i_mouse.z + self.i_mouse.w


class ClassAndInstance:
    """Class attributes and instance attributes read by the same kernel."""

    class_center = float2(100.0, 50.0)
    class_scale = 3.0

    def __init__(self) -> None:
        self.inst_center = float2(7.0, 9.0)
        self.inst_scale = 2.0

    @kernel
    def read(self, out):
        out[0] = self.class_center.x + self.class_center.y + self.class_scale
        out[1] = self.inst_center.x + self.inst_center.y + self.inst_scale


class Float2BeforeInt:
    """A float2 member whose offset the int member's placement shifts."""

    def __init__(self) -> None:
        self.center = float2(6.0, 8.0)
        self.count = 5
        self.gain = 1.5

    @kernel
    def read(self, out):
        out[0] = self.center.x + self.center.y + self.count + self.gain
