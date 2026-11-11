"""Bug: an override with more parameters than the base call passes loses the self-attr scan.

Two shapes, one root. `ExtraArg.sample` takes a third parameter the base's call does not
pass; the launch surfaces an arity error (`sample() takes 5 device arguments but 4 types were
given`) -- annoying but honest. `FlipCall.draw` calls `self.sample(i, j, <expression>)` with
all three, where the expression reads ``self.image.shape[0]`` -- and then the scan returns
*empty*: the override's own ``self.origin`` and ``self.phase`` reads are gone
(``discovered attributes: image``), the launch builds without them, and the refusal names the
first missing attribute.

The base call site is scanned at decoration time, when ``owner`` is None and the scanned
callee is the base's own ``sample``; the launch-time scan follows the call into the owner's
override. When the override's signature does not match the base call's arity -- an extra
parameter the base does not pass, the exact shape a faithful GLSL ``mainImage`` port draws
(``float(height) - float(i) - 0.5`` built from the receiver's own array) -- the follower loses
the override instead of collecting against the owner's signature, and every uniform read in
the subclass disappears with it.

The controls bound the defect: an override matching the base's arity scans fine, a base call
whose expression reads ``self`` with a two-argument hook scans fine, and the arity error for
the extra-parameter shape at least names the problem. What is owed here: the launch-time scan
resolves ``self.<hook>(...)`` against the *owner's* override (whose signature it has), and an
override the base call cannot satisfy is refused with an error that says so -- never a silent
empty scan.

Found by `examples/shadertoy/shadertoy_viewer.py`: its ``draw`` passes the GLSL flip
(``float(height) - float(i) - 0.5``) as a third argument to a hook the base declares with
two, and every shader subclass that reads a uniform through ``math.sin(self.i_time)`` or
``min(x, self.limit)`` refused to launch. The colour-wheel example escaped by accident: its
uniform reads sit inside module-helper calls the follower does not walk into.
"""
from __future__ import annotations

import math

import numpy as np

from pygpgpu import kernel
from pygf import float2


class Base:
    """The kernel calls the hook with two arguments; the base hook takes two."""

    def __init__(self) -> None:
        self.image = np.zeros((2, 3), dtype=np.float32)
        self.phase = np.float64(0.5)

    def sample(self, i: int, j: int) -> np.float64:
        """The hook the base declares and calls with two arguments."""
        return np.float64(0)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)


class ExtraArg(Base):
    """The override takes one argument MORE than the base's call passes."""

    def sample(self, i: int, j: int, y: float) -> np.float64:
        return math.sin(self.phase) * float(i + j)


class FlipCall(Base):
    """A subclass whose own kernel passes three arguments, the third built from ``self``."""

    def __init__(self) -> None:
        super().__init__()
        self.origin = float2(3.0, 2.0)

    @kernel
    def draw(self):
        height = self.image.shape[0]
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j, float(height) - float(i) - 0.5)


class FlipSin(FlipCall):
    """The override reads two members; its subclass's own kernel calls it with three."""

    def sample(self, i: int, j: int, y: float) -> np.float64:
        uv = float2(float(i), float(j)) / self.origin
        return math.sin(self.phase) * uv.x
