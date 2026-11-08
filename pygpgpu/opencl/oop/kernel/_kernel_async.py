# The async forms: the worker pool, the futures, the queue release.
from __future__ import annotations
from ._kernel_support import _async_workers, _bridge, _set_error
from ..command_queue import CommandQueue
from ..event import Event


class _KernelAsyncMixin:

    def _launch_async(self, args, kwargs, future_type:type):
        """Launch without waiting, drawing from the asynchronous queue pool.

        When every pooled queue is busy this returns a pending future at once and
        finishes the launch on a worker thread once a queue frees up, so the
        caller never blocks.
        """
        future = future_type()
        context = self._program.context
        device = self._pending_device() or context.devices[0]
        queue = context.get_async_queue(device, blocking=False)

        if queue is None:
            _async_workers().submit(self._launch_when_free, future, args, kwargs, future_type)
            return future

        try:
            events = self._call(args, kwargs, queue)
        except BaseException as error:
            self._release_queue(queue, async_launch=True)
            future.set_exception(error)
            return future

        _bridge(Event.create_future(events, future_type), future,
                lambda: self._release_queue(queue, async_launch=True))
        return future


    def _launch_when_free(self, future, args, kwargs, future_type:type)->None:
        """Worker-thread half of an asynchronous launch that had to wait."""
        context = self._program.context
        device = context.devices[0]
        try:
            queue = context.get_async_queue(device, blocking=True)
        except BaseException as error:
            _set_error(future, error)
            return

        try:
            events = self._call(args, kwargs, queue)
        except BaseException as error:
            self._release_queue(queue, async_launch=True)
            _set_error(future, error)
            return

        _bridge(Event.create_future(events, future_type), future,
                lambda: self._release_queue(queue, async_launch=True))


    def _release_queue(self, queue:CommandQueue, async_launch:bool)->None:
        context = self._program.context
        if async_launch:
            context.release_async_queue(queue)
        else:
            context.release_sync_queue(queue)
