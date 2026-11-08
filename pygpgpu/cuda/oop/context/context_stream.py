from __future__ import annotations
import threading
from typing import Optional, TYPE_CHECKING

from ...driver import CUstream_flags

if TYPE_CHECKING:
    # numpy is imported inside the methods that use it; the annotation needs the name.


    from ..stream import Stream


# Upper bound on the streams a Context keeps alive. Synchronous calls share one
# cached stream per thread and asynchronous calls draw from a separate pool of
# non-blocking streams; capping both keeps a process that creates and drops
# threads (or keeps launching asynchronously) from growing streams forever.
STREAM_POOL_SIZE:int = 16

# How many times create_texture rebuilds an image before giving up. The driver
# rejects texture creation intermittently, and a refused array stays refused, so
# each attempt allocates a replacement rather than retrying the same handle.
TEXTURE_RETRY_ATTEMPTS:int = 40


# Upper bound on the streams a Context keeps alive. Synchronous calls share one
# cached stream per thread and asynchronous calls draw from a separate pool of
# non-blocking streams; capping both keeps a process that creates and drops
# threads (or keeps launching asynchronously) from growing streams forever.
STREAM_POOL_SIZE:int = 16

if TYPE_CHECKING:
    from ..stream import Stream

class _ContextStreamMixin:

    def _take_idle_sync(self) -> Optional[Stream]:
        """Pop the least-recently-used idle stream from the synchronous cache.

        Caller must hold _stream_lock. The entry is returned without being marked
        busy; whoever takes it does that.
        """
        for ident, stream in self._sync_streams.items():
            if not stream.busy:
                del self._sync_streams[ident]
                return stream
        return None


    def _take_idle_async(self) -> Optional[Stream]:
        """Pop an idle stream from the asynchronous pool (caller holds the lock)."""
        for index, stream in enumerate(self._async_streams):
            if not stream.busy:
                del self._async_streams[index]
                return stream
        return None


    def get_sync_stream(self)->Stream:
        """The stream a blocking call should use.

        The main thread runs on CUDA's legacy NULL stream: it is never created and
        never destroyed, and its implicit ordering with every other blocking stream
        is exactly what sequential code wants, so the common case allocates nothing.

        Other threads share a least-recently-used cache capped at STREAM_POOL_SIZE,
        keyed by thread id. A stream belongs to the context that was current when it
        was created, not to the thread, so it can be handed to another thread as
        long as this context is made current there first (which the pool always
        does). Three things are done to avoid needless create/destroy cycles:

        - A new thread takes over the least-recently-used idle entry rather than
          creating a stream, then evicting it and creating another.
        - When the cache is full, the victim entry is re-keyed onto the current
          thread instead of being destroyed, so no stream is created at all.
        - Only when the cache and the asynchronous pool hold no idle stream is a
          new stream created -- and even then the asynchronous pool is drained
          first, so the two pools share one budget.
        """
        if threading.current_thread() is threading.main_thread():
            with self._stream_lock:
                self.make_current()
                if self._legacy_stream is None:
                    from ..stream import Stream
                    self._legacy_stream = Stream.legacy()
                return self._legacy_stream

        ident = threading.get_ident()
        with self._stream_lock:
            self.make_current()

            cached = self._sync_streams.get(ident)
            if cached is not None and not cached.busy:
                self._sync_streams.move_to_end(ident)
                cached.busy = True
                return cached

            # A new thread (or one whose cached stream is still in flight): reuse
            # whatever is already idle before creating anything.
            stream = self._take_idle_sync()
            if stream is None:
                stream = self._take_idle_async()
                if stream is not None:
                    self._async_slots.release()
            if stream is not None:
                stream.busy = True
                self._sync_streams[ident] = stream
                return stream

            from ..stream import Stream
            stream = Stream()
            stream.busy = True
            self._sync_streams[ident] = stream

            if len(self._sync_streams) > STREAM_POOL_SIZE:
                self._shrink_sync_cache(ident)

            return stream


    def _shrink_sync_cache(self, keep_ident:int)->None:
        """Bring the synchronous cache back to STREAM_POOL_SIZE (caller holds lock).

        The least-recently-used entry is re-keyed onto ``keep_ident`` rather than
        destroyed, so the cache stays full but nothing is allocated or released.
        Only if every entry is busy -- real concurrency above the pool size -- is
        the new entry left out of the cache.
        """
        while len(self._sync_streams) > STREAM_POOL_SIZE:
            for victim_ident, victim in self._sync_streams.items():
                if victim_ident == keep_ident:
                    continue
                if victim.busy:
                    continue
                # Re-key instead of destroy: the victim becomes the entry for the
                # thread that just asked, so no stream is created or destroyed.
                del self._sync_streams[victim_ident]
                del self._sync_streams[keep_ident]
                self._sync_streams[keep_ident] = victim
                return
            # Everything else is in flight; drop the newest entry from the cache
            # and let its finalizer reclaim it when it goes idle.
            del self._sync_streams[keep_ident]
            return


    def release_sync_stream(self, stream:Stream)->None:
        """Return a synchronous stream to the cache (the NULL stream is kept)."""
        if stream is self._legacy_stream:
            return
        with self._stream_lock:
            stream.busy = False


    def get_async_stream(self, blocking:bool=True)->Optional[Stream]:
        """Take a non-blocking stream from the asynchronous pool.

        ``blocking`` selects what a caller does when all STREAM_POOL_SIZE streams
        are in flight: a thread that is willing to wait blocks here, while an
        asynchronous launch asks with ``blocking=False`` and waits on its own
        worker instead of stalling the caller. Returns None when
        ``blocking=False`` and no stream is free.

        An idle stream from the synchronous cache is taken over before creating a
        new one, so the two pools share a single budget of live streams.
        """
        if blocking:
            self._async_slots.acquire()
        elif not self._async_slots.acquire(blocking=False):
            return None

        with self._stream_lock:
            self.make_current()

            stream = self._take_idle_async()
            if stream is None:
                stream = self._take_idle_sync()
            if stream is not None:
                stream.busy = True
                self._async_streams.append(stream)
                return stream

            from ..stream import Stream
            stream = Stream(CUstream_flags.CU_STREAM_NON_BLOCKING)
            stream.busy = True
            self._async_streams.append(stream)
            return stream


    def release_async_stream(self, stream:Stream)->None:
        """Return a non-blocking stream to the pool and free its slot."""
        with self._stream_lock:
            stream.busy = False
        self._async_slots.release()
