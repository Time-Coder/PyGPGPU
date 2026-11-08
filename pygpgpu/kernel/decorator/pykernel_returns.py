from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..translator import ReturnSiteLayout
import asyncio
import ast
import concurrent.futures
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
import numpy as np
from ...utils import _current_loop
from .slots import (
    _return_value,
)
from ._typing import (
    OuterState,
    ReturnSlots,
    ReturnState,
)


class _PykernelReturnsMixin:

    def _alloc_return_slots(self, returns:ReturnSlots)->ReturnState:
        """One device buffer per value slot, plus the shared claim and site words.

        Every buffer is zero-initialised: the claim word then starts unclaimed, so a
        launch where no work item returned reads back as None rather than as garbage.
        """
        buffers:List[np.ndarray] = [np.zeros(1, dtype=dtype) for dtype in returns["slot_types"]]
        return {"buffers": buffers,
                "claim": np.zeros(1, dtype=np.int32),
                "site": np.zeros(1, dtype=np.int32)}


    def _finish_returns(self, return_state:ReturnState,
                        returns:ReturnSlots, full:List[object],
                        created:Optional[Dict[str, object]]=None)->object:
        """Rebuild the python result of a finished launch.

        The winning return site decides the shape: `return a, b` gives a tuple,
        `return value` gives the bare value, and a bare `return` gives None. A launch in
        which no work item reached any `return` gives None too, exactly as a python
        function does when no returning branch was taken.

        An array position resolves to the host's own object: a parameter comes from the
        launch arguments, and an object the body created comes from `created`. Either way
        the caller receives the same object, never a copy.
        """
        if not int(return_state["claim"][0]):
            return None
        site:int = int(return_state["site"][0]) - 1
        layout:ReturnSiteLayout = returns["sites"][site]
        if not layout["positions"]:
            # The winning site was a bare `return` (or `return None`).
            return None
        values:List[object] = []
        for entry in layout["positions"]:
            if entry["kind"] == "array":
                name:str = entry["name"]
                if created is not None and name in created:
                    values.append(created[name])
                else:
                    values.append(full[returns["arg_indices"][name]])
            elif entry.get("reused"):
                # The value already lives in the argument the body returned, so the caller
                # gets it back from there -- no slot was allocated for it.
                # The parameter travels as a one-element buffer, so the value it holds is
                # what the caller passed -- and what the body may have updated. A numpy
                # record (`np.void`) is a parameter too, but indexing a record reads its
                # first *field*, so only a real 0-d/1-element array is unwrapped.
                shared_slot:object = full[returns["arg_indices"][entry["reused"]]]
                if isinstance(shared_slot, np.void):
                    values.append(shared_slot)
                else:
                    try:
                        values.append(shared_slot[0])
                    except (IndexError, TypeError):
                        values.append(shared_slot)
            else:
                dtype:object = returns["slot_types"][entry["slot"]]
                kind:str = returns["slot_kinds"][entry["slot"]]
                values.append(_return_value(return_state["buffers"][entry["slot"]][0], dtype, kind))
        return tuple(values) if layout["is_tuple"] else values[0]


    def _with_returns(self, future:object,
                      return_state:Dict[str, object],
                      returns:Optional[Dict[str, object]],
                      full:List[object],
                      outer_state:Optional[Dict[str, object]]=None,
                      created:Optional[Dict[str, object]]=None)->object:
        """Wrap an asynchronous launch so its future resolves to the returned value.

        The backend future resolves once every read-back has completed (see the kernel
        classes), so by the time this callback runs the slots hold the winner's values.
        A backend future may be bound to a loop other than the caller's (the python
        backend installs one when none is running), so an asyncio result is delivered
        through a fresh future on the caller's running loop when there is one, and is
        otherwise completed in place on whatever loop the backend future uses.
        """
        is_asyncio:bool = isinstance(future, asyncio.Future)
        deliver:object = concurrent.futures.Future()
        try:
            caller_loop:object = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if is_asyncio:
            deliver = caller_loop.create_future() if caller_loop is not None else future

        def settle(*_args:object)->None:
            error:Optional[BaseException] = None
            try:
                error = future.exception()
            except BaseException as caught:  # cancelled, or never resolved
                error = caught

            def apply() -> None:
                if deliver.done():
                    return
                if error is not None:
                    deliver.set_exception(error)
                else:
                    # Write-back happens here, after the launch has completed and its
                    # buffers have been read back, so a declared outer scalar holds the
                    # value the kernel left in it by the time the caller sees the result.
                    if outer_state is not None:
                        self._write_back_outer(outer_state)
                    deliver.set_result(self._finish_returns(return_state, returns, full, created)
                                       if return_state else None)

            if is_asyncio:
                try:
                    future.get_loop().call_soon_threadsafe(apply)
                except RuntimeError:
                    pass  # the loop was closed before the launch finished
                return
            apply()

        if future.done():
            settle()
        else:
            future.add_done_callback(settle)
        return deliver


    def _with_tail(self, future:object,
                   tail_statements:List[ast.stmt],
                   args:Tuple[object, ...], instance:object,
                   carried_slots:Dict[str, object],
                   gap_slots:Dict[str, object],
                   created_objects:Dict[str, object],
                   outer_state:OuterState,
                   outer_writable:List[Dict[str, object]],
                   return_state:Dict[str, object],
                   returns:Optional[Dict[str, object]],
                   full:List[object],
                   created:Optional[Dict[str, object]]=None)->object:
        """Wrap an asynchronous launch so its future resolves the way a call would.

        The synchronous launch runs the host tail (the statements after the last
        loop, whose `return` is often the answer) and writes the outer scalars back
        after the read-backs land. The future must hand back the same thing: the
        backend future resolves once every read-back has completed, so the settle
        callback runs the tail there and finishes exactly as the inline path does.
        A body with neither a tail nor a write-back (and no device claim) never
        reaches this method. The backend future resolves from the driver's callback,
        and the tail must not race it: the value is delivered through a fresh future
        (a concurrent one, or one on the caller's loop for asyncio), settled only
        after the tail has run.
        """
        tail_writable:Dict[str, np.ndarray] = {
            entry["name"]: entry["buffer"] for entry in outer_writable}
        is_asyncio:bool = isinstance(future, asyncio.Future)
        try:
            caller_loop:object = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        deliver:object = (caller_loop.create_future()
                          if is_asyncio and caller_loop is not None
                          else (asyncio.Future(loop=_current_loop())
                                if is_asyncio else concurrent.futures.Future()))

        def settle(*_args:object)->None:
            error:Optional[BaseException] = None
            try:
                error = future.exception()
            except BaseException as caught:  # cancelled, or never resolved
                error = caught

            def settle_error() -> None:
                if not deliver.done():
                    deliver.set_exception(error)

            def apply() -> None:
                if error is not None:
                    if is_asyncio:
                        try:
                            future.get_loop().call_soon_threadsafe(settle_error)
                        except RuntimeError:
                            pass  # the loop was closed before the launch finished
                    else:
                        settle_error()
                    return
                try:
                    tail_result:object = self._run_host_region(
                        tail_statements, args, instance,
                        carried_slots, gap_slots, created_objects=created_objects,
                        outer_state=outer_state, writable_outer=tail_writable)
                    self._write_back_outer(outer_state)
                    if return_state and int(return_state["claim"][0]):
                        # A work item reached an in-loop return first: python's
                        # "first return wins".
                        result:object = self._finish_returns(
                            return_state, returns, full, created)
                    elif tail_result is not None:
                        result = tail_result
                    else:
                        result = (self._finish_returns(return_state, returns, full, created)
                                  if return_state else None)
                except BaseException as tail_error:
                    if is_asyncio:
                        try:
                            future.get_loop().call_soon_threadsafe(
                                self._deliver_error, deliver, tail_error)
                        except RuntimeError:
                            pass
                    else:
                        self._deliver_error(deliver, tail_error)
                    return
                if is_asyncio:
                    try:
                        future.get_loop().call_soon_threadsafe(
                            self._deliver_result, deliver, result)
                    except RuntimeError:
                        pass  # the loop was closed before the launch finished
                else:
                    self._deliver_result(deliver, result)

            apply()

        if future.done():
            settle()
        else:
            future.add_done_callback(settle)
        return deliver
