# The host-side state of an array: the bytes it holds, who owns them, and when
# a pending device write has to land before the host may read.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ndarray

from .binding import (
    _Binding,
    _Family,
)
from typing import List, Optional, Tuple
import numpy as np
from ...utils import initial_print_info
from .._backends import (download, sync_queue, wait_event)
from .._devices import copy_queue, release_copy_queue


class _HostStateMixin:

    def _root(self) -> "ndarray":
        current: "ndarray" = self
        for _ in range(64):
            parent = getattr(current, "_parent", None)
            if parent is None:
                return current
            current = parent
        return current


    def _make_family(self) -> "_Family":
        """Create this family's state, choosing the layout its root can express."""
        root = self._root()
        root_array = root.view(np.ndarray)
        if root.flags["C_CONTIGUOUS"]:
            family = _Family(root, root_array, live=True)
        else:
            # A strided root cannot be handed to a device as one contiguous buffer, so the
            # family keeps a contiguous copy and refreshes it on every host write.
            family = _Family(root, np.ascontiguousarray(root_array), live=False)
        root._family = family
        if root is not self:
            self._family = family
        return family


    @property
    def _state(self) -> "_Family":
        """The family this element belongs to, created on first use."""
        family = self.__dict__.get("_family")
        if family is not None:
            return family
        root = self._root()
        family = root.__dict__.get("_family")
        if family is None:
            return self._make_family()
        self._family = family
        return family


    def _member_bytes(self) -> Tuple[np.ndarray, int]:
        """``(host_array, byte_offset)``: the bytes this element occupies in the family.

        For the root (and a view whose layout is the family's own) the array is
        ``family.data`` itself; for any other view it is a contiguous copy kept on the
        element, rebuilt whenever the family moves on. The offset is where those bytes
        start inside ``family.data``, which is how a device write to a view is folded back
        into the whole buffer.
        """
        family = self._state
        root = family.root()
        if root is None or self is root:
            return family.data, 0
        view = self.view(np.ndarray)
        if view.flags["C_CONTIGUOUS"] and root is not None:
            offset = int(view.__array_interface__["data"][0]) - int(
                root.view(np.ndarray).__array_interface__["data"][0])
            if 0 <= offset <= family.data.nbytes - view.nbytes:
                return family.data[offset:offset + view.nbytes], offset
        # A non-contiguous or unrepresentable view: keep its own contiguous copy.
        if self._staging is None or self._staging_epoch < family.current_epoch:
            self._staging = np.ascontiguousarray(view)
            self._staging_epoch = family.current_epoch
        return self._staging, -1


    def _absorb(self, host_array: np.ndarray, offset: int) -> None:
        """Fold the bytes just downloaded for one element back into the family."""
        if offset >= 0:
            # The bytes came down straight into the family's buffer: nothing to fold.
            return
        # This element owns a private contiguous copy (its view is not contiguous, so a
        # device buffer for it is a copy rather than a window into the family). The
        # downloaded values are the *view's* values, so they have to be scattered back
        # through the view's own layout -- writing them contiguously into the family
        # would shift every row and lose the assignment entirely.
        view = self.view(np.ndarray)
        try:
            np.copyto(view, np.ascontiguousarray(host_array).reshape(view.shape))
        except (ValueError, TypeError):
            # A shape/dtype the copy cannot express: the host keeps the staged copy,
            # which is already current, so there is nothing else to do here.
            return
        finally:
            self._staging = None
            self._staging_epoch = 0


    def _print_copy_info(self) -> bool:
        """Whether the copy log is on: ``config.print_info``, read without a device.

        This asks the process-wide default rather than scanning the loaded backends' own
        flags, so a copy line is decided by the setting the caller wrote. Silencing one
        backend directly with its own class attribute does not move this, the same way it
        does not move what a backend imported later starts from.
        """
        return initial_print_info()


    def _wait_pending(self, bindings: Optional[List[_Binding]] = None) -> None:
        """Wait for the copies these bindings still have in flight."""
        entries = (list(self._bindings.values()) if bindings is None else bindings)
        for binding in entries:
            if binding.pending:
                pending, binding.pending = list(binding.pending), []
                for transfer in pending:
                    transfer.wait()


    def _ensure_host_bytes(self) -> None:
        """Make the host bytes current, waiting for the copy that carries them.

        Every host-side operation goes through here, which is what lets ``to_host()`` be
        asynchronous: the next actual read waits for the copy.
        """
        if not self.__dict__.get("_should_copy", True):
            return
        family = self._state
        self._wait_pending()
        with family.lock:
            if family.current_epoch < family.epoch:
                newest = family.newest_binding()
                if newest is not None:
                    self._hydrate_from(family, newest)


    def _hydrate_from(self, family: "_Family", newest: Tuple["ndarray", _Binding]) -> None:
        """Download the newest device content into the family's host bytes, now."""
        member, binding = newest
        target = binding.target
        host_array, offset = member._member_bytes()
        queue = copy_queue(target)
        try:
            event = download(target, queue, binding.mem_obj, host_array)
            if event is not None:
                wait_event(target, event)
            else:
                sync_queue(target, queue)
        finally:
            release_copy_queue(target, queue)
        member._absorb(host_array, offset)
        family.current_epoch = binding.epoch


    def _after_host_write(self) -> None:
        """Record that this element's bytes changed on the host."""
        family = self.__dict__.get("_family")
        if family is None:
            return
        family.note_writer(self)
        family.refresh_data()
        with family.lock:
            family.epoch += 1
            family.current_epoch = family.epoch
            self._staging = None
            self._staging_epoch = 0


    def _set_host_dirty(self) -> None:
        """Mark the whole family as changed on the host."""
        self._root()._after_host_write()


    def _set_tree_host_dirty(self) -> None:
        """Kept for the launch path: marking the family covers every view of it."""
        self._set_host_dirty()
