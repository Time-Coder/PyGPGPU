# Moving an array between the host and a device.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .binding import (
    _Binding,
    _Family,
)
from typing import Optional
from .._backends import (allocate, download, error_code_ok, event_callback,
                        release, upload, zero_copy_support)
from .._devices import Target, copy_queue, release_copy_queue, target_of
from ..transfer import Transfer


class _TransferMixin:

    def to_host(self, stream: object = None, *, backend: Optional[str] = None,
                device: object = None, wait: bool = False) -> Transfer:
        """Start copying this array's contents back to host memory.

        The copy is asynchronous: the returned :class:`Transfer` is already valid, and
        anything that later *reads* the array waits for it automatically. Pass
        ``wait=True`` to block now, or use ``transfer.wait()`` / ``await transfer`` to
        block explicitly.

        The host bytes the array already has are what the copy lands in, so a device that
        wrote the array does not disturb the rest of the family.
        """
        family = self._state
        newest = family.newest_binding()
        if newest is None or newest[1].epoch <= family.current_epoch:
            transfer = Transfer.completed("to_host")
            if wait:
                transfer.wait()
            return transfer

        member, binding = newest
        target = binding.target
        host_array, offset = member._member_bytes()
        if binding.mapped:
            # The binding *is* the host bytes, so the device's writes are already here.
            with family.lock:
                family.current_epoch = binding.epoch
            transfer = Transfer.completed(f"to_host {target}")
            if wait:
                transfer.wait()
            return transfer

        queue = copy_queue(target)
        transfer = Transfer(f"to_host {family.data.shape} from {target}")
        try:
            event = download(target, queue, binding.mem_obj, host_array)
        except BaseException:
            release_copy_queue(target, queue)
            raise

        def complete(error_code: Optional[int] = None) -> None:
            try:
                if not error_code_ok(target.backend, error_code):
                    transfer._complete(RuntimeError(
                        f"the copy from {target} failed ({error_code})"))
                    return
                with family.lock:
                    family.current_epoch = binding.epoch
                member._absorb(host_array, offset)
                if self._print_copy_info():
                    print(f"copy data to host for argument '{self._arg_name}'")
                transfer._complete()
            finally:
                release_copy_queue(target, queue)

        if event is None:
            complete()
        else:
            binding.add_pending(transfer)
            event_callback(target, queue, event, complete)
        if wait:
            transfer.wait()
        return transfer


    def to_device(self, target: object = None, stream: object = None, *,
                  backend: Optional[str] = None, device: object = None,
                  zero_copy: Optional[bool] = None, wait: bool = False) -> Transfer:
        """Start copying this array's contents to a device.

        ``target`` is a selector string (``"cuda"``, ``"opencl:1"``), a
        ``(backend, device)`` pair, or a Device object; ``None`` means the default device,
        which is the one a plain ``kernel(...)`` call would run on. The ``backend=`` /
        ``device=`` keywords are the same thing spelled in two arguments.

        When the array's bytes are already current on that device nothing is copied and the
        transfer is complete at once -- that is what makes a second kernel call on the same
        device free.

        ``zero_copy=True`` asks the device to read and write the array's own host memory
        instead of a copy of it. That is honoured only where the backend's API guarantees
        it (see :func:`pygpgpu.numpy.zero_copy_support`); elsewhere it raises
        ``NotImplementedError`` naming the backend. ``None`` (the default) keeps whatever
        choice the array already made for that device.
        """
        if target is not None and (backend is not None or device is not None):
            raise TypeError(
                "to_device() takes either a positional target selector or the "
                "backend=/device= keywords, not both")
        if target is None and (backend is not None or device is not None):
            target = (backend, device)
        return self._to_device_target(target_of(target), zero_copy=zero_copy, wait=wait)


    def _to_device_target(self, target: Target, zero_copy: Optional[bool] = None,
                          wait: bool = False) -> Transfer:
        """The engine's device copy: reuse a current binding, or upload."""
        family = self._state
        key = (target.key, self._layout_key(target.backend))
        binding = self._bindings.get(key)
        if binding is not None and binding.epoch == family.epoch and \
                (zero_copy is None or binding.zero_copy == bool(zero_copy)):
            transfer = Transfer.completed(f"to_device {target}")
            if wait:
                transfer.wait()
            return transfer

        if binding is not None and zero_copy is not None and \
                binding.zero_copy != bool(zero_copy):
            # Changing the memory model for this device: a mapped buffer and a copy do not
            # hold the same bytes at the same address, so the old binding is dropped.
            self._release_binding(key, binding)
            binding = None

        if binding is None:
            use_zero_copy = bool(zero_copy)
            if use_zero_copy:
                supported, reason = zero_copy_support(target)
                if not supported:
                    raise NotImplementedError(reason)
            host_array, offset = self._member_bytes()
            host_ptr = (int(host_array.__array_interface__["data"][0]) if use_zero_copy
                        else None)
            mem_obj = allocate(target, int(host_array.nbytes), use_zero_copy, host_ptr)
            binding = _Binding(mem_obj, 0, target, use_zero_copy)
            self._bindings[key] = binding
        return self._upload_into(target, binding, family, wait)


    def _upload_into(self, target: Target, binding: _Binding, family: "_Family",
                     wait: bool) -> Transfer:
        """Bring a stale binding up to date on the device."""
        # A queued copy on this binding has to land before a new one starts on it.
        self._wait_pending([binding])

        host_array, offset = self._member_bytes()
        if binding.mapped:
            # The binding is the host bytes, so it is current the moment the host is.
            self._ensure_host_bytes()
            binding.epoch = family.epoch
            transfer = Transfer.completed(f"to_device {target} (zero-copy)")
            if wait:
                transfer.wait()
            return transfer

        if family.current_epoch < family.epoch:
            newest = family.newest_binding()
            if newest is not None and newest[1] is not binding:
                # Another device holds newer content: stage through the host, which is the
                # only route between two devices and between two backends.
                self._hydrate_from(family, newest)

        queue = copy_queue(target)
        transfer = Transfer(f"to_device {target} for {family.data.shape}")
        family.begin_read()
        try:
            event = upload(target, queue, binding.mem_obj, host_array)
        except BaseException:
            family.end_read()
            release_copy_queue(target, queue)
            raise

        def complete(error_code: Optional[int] = None) -> None:
            try:
                family.end_read()
                if not error_code_ok(target.backend, error_code):
                    transfer._complete(RuntimeError(
                        f"the copy to {target} failed ({error_code})"))
                    return
                with family.lock:
                    binding.epoch = family.epoch
                if self._print_copy_info():
                    print(f"copy data to device for argument '{self._arg_name}'")
                transfer._complete()
            finally:
                release_copy_queue(target, queue)

        if event is None:
            complete()
        else:
            binding.add_pending(transfer)
            family.note_writer(self)
            event_callback(target, queue, event, complete)
        if wait:
            transfer.wait()
        return transfer


    def _release_binding(self, key: tuple, binding: _Binding) -> None:
        """Drop one device binding, waiting for what it still has queued."""
        self._wait_pending([binding])
        self._bindings.pop(key, None)
        release(binding.mem_obj)
        binding.mem_obj = None


    def release(self, target: object = None) -> None:
        """Release the device copies of this array (the host bytes stay untouched).

        Without a target it releases every copy; with one it releases that device's copy.
        Any content the device holds that the host does not have is downloaded first --
        releasing a copy is not a way to lose the result -- and a copy still in flight is
        waited for, because the device is writing into it.
        """
        family = self._state
        if target is None:
            self._wait_pending()
            with family.lock:
                if family.current_epoch < family.epoch:
                    newest = family.newest_binding()
                    if newest is not None:
                        self._hydrate_from(family, newest)
            for key in list(self._bindings.keys()):
                self._release_binding(key, self._bindings[key])
            return
        resolved = target_of(target)
        keys = [key for key in self._bindings if key[0] == resolved.key]
        # Only downloading from this device is pointless if another one holds newer bytes;
        # the family-wide pull happens first so nothing is lost either way.
        self._wait_pending([self._bindings[key] for key in keys])
        with family.lock:
            if family.current_epoch < family.epoch:
                newest = family.newest_binding()
                if newest is not None:
                    self._hydrate_from(family, newest)
        for key in keys:
            self._release_binding(key, self._bindings[key])
