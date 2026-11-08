# Binding an array to a launch: which buffer a kernel argument reads, and which
# writes have to travel back.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ImageDescriptor

from .binding import (
    _Binding,
    _IMAGE_ALWAYS_UPLOAD,
    _queue_backend,
)
from typing import Optional, Tuple
import numpy as np
from ...constants import KernelFlags
from .._backends import (allocate, allocate_image, event_callback,
                        upload)
from .._devices import Target, target_of
from ..transfer import Transfer


class _LaunchBindingMixin:

    def _layout_key(self, backend: str) -> tuple:
        """What a device copy's bytes depend on, besides the contents."""
        view = self.view(np.ndarray)
        return (backend, view.dtype.str, view.shape, view.strides)


    def _to_device(self, cmd_queue: object, arg_name: str,
                   flags: KernelFlags = KernelFlags(1 << 0),
                   kernel_flags: KernelFlags = KernelFlags(1 << 0),
                   image_info: Optional["ImageDescriptor"] = None,
                   *, backend: Optional[str] = None,
                   device: object = None) -> Tuple[object, object]:
        """Bind this argument to the device a launch runs on; returns ``(mem_obj, event)``.

        This is the interface the backends' ``use_buffer``/``use_image`` call. ``event`` is
        the backend's copy event (None when nothing had to travel); the caller keeps it
        alive for the launch and for the read-back.
        """
        used_backend = backend if backend is not None else _queue_backend(cmd_queue)
        used_device = device
        if used_device is None:
            used_device = getattr(cmd_queue, "device", None)
        if used_device is None:
            raise ValueError(
                f"cannot tell which device the {type(cmd_queue).__name__} runs on; pass "
                f"device= explicitly")
        target = target_of((used_backend, used_device))
        self._arg_name = arg_name
        if image_info is not None:
            return self._bind_image(target, cmd_queue, arg_name, image_info, kernel_flags)
        return self._bind_for_launch(target, cmd_queue, kernel_flags)


    def _bind_for_launch(self, target: Target, cmd_queue: object,
                         kernel_flags: KernelFlags) -> Tuple[object, object]:
        """Bind this element's bytes to ``target`` on the caller's own channel."""
        family = self._state
        key = (target.key, self._layout_key(target.backend))
        binding = self._bindings.get(key)

        if binding is not None and binding.epoch == family.epoch:
            # The device already holds exactly these bytes: the launch is free of copies.
            self._wait_pending([binding])
            return binding.mem_obj, None

        if binding is None:
            host_array, _offset = self._member_bytes()
            mem_obj = allocate(target, int(host_array.nbytes))
            binding = _Binding(mem_obj, 0, target, False)
            self._bindings[key] = binding

        self._wait_pending([binding])
        host_array, _offset = self._member_bytes()
        if family.current_epoch < family.epoch:
            newest = family.newest_binding()
            if newest is not None and newest[1] is not binding:
                self._hydrate_from(family, newest)

        event = upload(target, cmd_queue, binding.mem_obj, host_array)
        family.note_writer(self)
        if event is None:
            with family.lock:
                binding.epoch = family.epoch
            return binding.mem_obj, None

        # The launch reads these bytes; keep anything from changing them until it has.
        family.begin_read()
        event_callback(target, cmd_queue, event, lambda _code: family.end_read())
        with family.lock:
            binding.epoch = family.epoch
        if self._print_copy_info():
            print(f"copy data to device for argument '{self._arg_name}'")
        return binding.mem_obj, event


    def _bind_image(self, target: Target, cmd_queue: object, arg_name: str,
                    image_info: "ImageDescriptor",
                    kernel_flags: KernelFlags) -> Tuple[object, object]:
        """Bind an image argument: images are re-uploaded per launch, as they always were."""
        host_array, _offset = self._member_bytes()
        view = self.view(np.ndarray)
        if view.shape != tuple(image_info.shape):
            raise ValueError(
                f"the image argument '{arg_name}' has host shape {view.shape} but its "
                f"image is declared with shape {tuple(image_info.shape)}")
        flags = image_info.flags
        key = (target.key, self._layout_key(target.backend) + ("image", id(image_info)),
               image_info.shape, image_info.dtype, flags)
        binding = self._bindings.get(key)
        if binding is None:
            mem_obj = allocate_image(target, cmd_queue, image_info)
            binding = _Binding(mem_obj, 0, target, False, image_info)
            self._bindings[key] = binding
            always_upload = True
        else:
            always_upload = _IMAGE_ALWAYS_UPLOAD

        if binding.epoch == self._state.epoch and not always_upload:
            return binding.mem_obj, None
        if always_upload:
            event = binding.mem_obj.set_data(cmd_queue, view)
        else:
            event = upload(target, cmd_queue, binding.mem_obj, host_array)
        binding.epoch = self._state.epoch
        self._state.note_writer(self)
        if self._print_copy_info() and event is not None:
            print(f"copy data to device for argument '{arg_name}'")
        return binding.mem_obj, event


    def _mark_device_written(self, mem_obj: object = None,
                             target: Optional[Target] = None) -> None:
        """Record that a kernel wrote this argument on the device.

        The binding just launched on is the freshest device copy of this element, so it
        takes the family's next epoch. That makes the host bytes stale (the next host
        access downloads) while leaving that binding current (the next launch on the same
        device reuses the buffer untouched).

        ``mem_obj`` names the buffer the launch actually wrote, which is what distinguishes
        "this device" from another one the array was also copied to: an array resident on
        two devices has a binding for each, and only the one the kernel ran on holds the
        new content.

        The binding may belong to a *different* view object than this one. Two views of
        one buffer (`k(a[0], a[0], a[0])` naming three separately-created views) share a
        single device buffer -- the launch binds the family's buffer once (`seen_arrays`) --
        so the binding lives wherever that first view put it, and searching only this
        object's own `_bindings` would find nothing and silently drop the write.
        """
        family = self._state
        self._wait_pending()
        candidates = [binding for binding in self._bindings.values()
                      if (mem_obj is None or binding.mem_obj is mem_obj)
                      and (target is None or binding.target.key == target.key)]
        if not candidates and mem_obj is not None:
            # Same buffer, held by another member of the family.
            root = family.root()
            for reference in list(family.writers):
                member = reference()
                if member is None or (root is not None and member is root and member is self):
                    continue
                for binding in member.__dict__.get("_bindings", {}).values():
                    if binding.mem_obj is mem_obj and (
                            target is None or binding.target.key == target.key):
                        candidates.append(binding)
        if not candidates:
            return
        with family.lock:
            epoch = family.epoch + 1
            family.epoch = epoch
            for binding in candidates:
                binding.epoch = epoch
            family.note_writer(self)


    def to_device_zero_copy(self, target: object = None) -> Transfer:
        """``to_device(..., zero_copy=True)`` as its own spelling.

        Equivalent to ``to_device(target, zero_copy=True)``: the device reads and writes
        the array's own host memory. Raises ``NotImplementedError`` naming the backend when
        its API cannot guarantee that.
        """
        return self._to_device_target(target_of(target), zero_copy=True)


    def is_zero_copy(self, target: object = None) -> bool:
        """Whether the array's bytes are bound to that device's own memory.

        ``target=None`` asks about any device: True when at least one binding is zero-copy.
        """
        if target is None:
            return any(binding.mapped for binding in self._bindings.values())
        resolved = target_of(target)
        return any(binding.mapped for key, binding in self._bindings.items()
                   if key[0] == resolved.key)
