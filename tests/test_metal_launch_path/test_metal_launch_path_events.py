"""The events tests, split out of `test_metal_launch_path.py`.

The subject is events.
"""
from __future__ import annotations



def test_an_event_runs_every_registered_callback() -> None:
    """Both callback spellings run immediately, because the work is already done.

    Metal reports completion through the command buffer's status and the launch waits
    on it, so an `Event` that exists describes finished work. `add_callback` has to
    answer to that; `on_completed_callbacks_add` used to append to a list nothing
    drained, so a caller that registered one was told nothing and heard nothing --
    a silent drop rather than a refusal.
    """
    from pygpgpu.metal.oop.event import Event

    called = []
    event = Event()
    assert event.finished
    assert event.status is None
    # `wait()` is a no-op rather than an error: the state is already final.
    event.wait()

    event.add_callback(lambda ev, error: called.append(("add", error)))
    event.on_completed_callbacks_add(lambda ev, error: called.append(("deferred", error)))
    assert [name for name, _error in called] == ["add", "deferred"]
    assert event.status is None


def test_an_failed_event_carries_its_error_to_the_callback() -> None:
    """A non-None `status` is an exception, and the callback receives it."""
    from pygpgpu.metal.oop.event import Event

    failure = ValueError("boom")
    event = Event(failure)
    assert event.status is failure
    seen = []
    event.add_callback(lambda ev, error: seen.append(error))
    assert seen == [failure]


def test_struct_result_sets_its_signature_under_the_send_lock() -> None:
    """`struct_result` touches the same shared `objc_msgSend` as `send` does.

    Setting `restype`/`argtypes` outside the lock would race with a concurrent `send`
    -- and the return convention differs (a struct comes back in registers on arm64,
    through `stret` on x86_64), so a mixed-up signature corrupts the value rather than
    raising.
    """
    import inspect

    import pygpgpu.metal.driver.objc as objc

    body = inspect.getsource(objc.ObjC.struct_result)
    assert "with ObjC._send_lock:" in body
    locked = body[body.index("with ObjC._send_lock:"):]
    assert "msg.restype" in locked
    assert "result = msg(" in locked


def test_a_message_send_is_serialized_across_threads() -> None:
    """`objc_msgSend`'s signature is shared state, so the send has to be serialized.

    `objc_msgSend` is variadic in C: ctypes needs the called method's own signature
    set on the SHARED function object before each call. Two threads readying
    different signatures at the same time can hand the wrong one to the call, which
    is undefined behaviour at the ABI level rather than a python exception -- and the
    launch path takes no lock of its own, because a Metal command queue is thread-safe
    at the framework level.
    """
    import inspect

    import pygpgpu.metal.driver.objc as objc

    assert hasattr(objc.ObjC, "_send_lock")
    source = inspect.getsource(objc.ObjC.send)
    assert "with ObjC._send_lock" in source
    # The protected region covers both the signature set-up and the call itself:
    # releasing between them would defeat the point.
    body = source[source.index("with ObjC._send_lock:"):]
    assert "msg.argtypes" in body
    assert "return msg(" in body
