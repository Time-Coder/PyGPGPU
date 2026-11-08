# The async forms: the worker pool, the stream callbacks, the release.
from __future__ import annotations
from ._kernel_support import _async_workers, _deliver, _set_future_error
import concurrent.futures
from ...driver import (
    HIP
)
from ..stream import Stream


class _KernelAsyncMixin:

    def _launch_async(self, args, kwargs, future_type:type):
        """Launch without waiting, drawing from the asynchronous stream pool.

        When every pooled stream is busy this returns a pending future at once and
        performs the launch on a worker thread once a stream frees up, so the
        caller never blocks.
        """
        future = future_type()
        context = self._program.context
        stream = context.get_async_stream(blocking=False)

        if stream is None:
            _async_workers().submit(self._launch_when_free, future, args, kwargs, future_type)
            return future

        try:
            self._launch_on(stream, args, kwargs)
        except BaseException as error:
            self._release_stream(stream, async_launch=True)
            future.set_exception(error)
            return future

        self._attach_completion(stream, future, future_type)
        return future


    def _launch_when_free(self, future, args, kwargs, future_type:type)->None:
        """Worker-thread half of an asynchronous launch that had to wait."""
        context = self._program.context
        try:
            stream = context.get_async_stream(blocking=True)
        except BaseException as error:
            _set_future_error(future, error)
            return

        try:
            self._launch_on(stream, args, kwargs)
        except BaseException as error:
            self._release_stream(stream, async_launch=True)
            _set_future_error(future, error)
            return

        self._attach_completion(stream, future, future_type)


    def _attach_completion(self, stream:Stream, future:object, future_type:type)->None:
        """Mirror the stream's completion onto ``future`` and return the stream."""
        def relay(pending:concurrent.futures.Future)->None:
            _deliver(future, pending)
            self._release_stream(stream, async_launch=True)

        stream.create_future(future_type).add_done_callback(relay)


    def _launch_on(self, stream:Stream, args, kwargs)->None:
        """Bind the arguments and enqueue the launch on an already-taken stream."""
        grid_dims, block_dims, value_ptrs, bound = self._prepare_call(args, kwargs, stream)
        HIP.hipModuleLaunchKernel(
            self.id,
            grid_dims[0], grid_dims[1], grid_dims[2],
            block_dims[0], block_dims[1], block_dims[2],
            0, stream.id, value_ptrs, None
        )
        if HIP.print_info:
            print(f"launch HIP {self} on {stream.device} with grid_dim={tuple(grid_dims)}, block_dim={tuple(block_dims)}", flush=True)

        self._after_call(stream, bound)


    def _release_stream(self, stream:Stream, async_launch:bool)->None:
        context = self._program.context
        if async_launch:
            context.release_async_stream(stream)
        else:
            context.release_sync_stream(stream)
