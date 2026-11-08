"""Chaining the passes of a multi-loop body for submit()/async_call().

The passes are dependent launches: each one's host region reads the previous pass's
read-backs and the next launch cannot start before it. The chain is driven by the
backend's own completion events -- a future per pass -- and never by a thread that waits
for the GPU. A completion callback only hands the continuation to a shared worker: a CUDA
stream callback may not call the driver API, and the next pass is a driver call.
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import ast
from typing import Dict, List, Optional, Tuple

from ._phase_readback import flush_deferred


# Shared by every multi-pass submission of the process: the continuation work is small
# (a host region and one enqueue), so a few workers serve every chain, and no thread is
# created per call.
_CHAIN_WORKERS: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="pygpgpu-phases")


class _PhaseAsyncMixin:

    def _launch_phases_async(self, plan: object, args: Tuple[object, ...],
                             instance: object, launch: str) -> object:
        """Run every pass of a multi-loop body asynchronously, in order.

        Returns the same future shape the single-pass asynchronous path returns: a
        ``concurrent.futures.Future`` for ``submit`` and an ``asyncio.Future`` for
        ``async_call``, settled with the call's result once the last pass and its tail
        have finished.
        """
        is_asyncio: bool = launch == "async"
        total: int = len(plan.active_kernels)
        if is_asyncio:
            try:
                caller_loop: Optional[object] = asyncio.get_running_loop()
            except RuntimeError:
                caller_loop = None
            deliver: object = caller_loop.create_future() if caller_loop is not None \
                else asyncio.Future()
        else:
            caller_loop = None
            deliver: object = concurrent.futures.Future()

        def settle_result(result: object) -> None:
            if is_asyncio and caller_loop is not None:
                caller_loop.call_soon_threadsafe(self._deliver_result, deliver, result)
            else:
                self._deliver_result(deliver, result)

        def settle_error(error: BaseException) -> None:
            if is_asyncio and caller_loop is not None:
                caller_loop.call_soon_threadsafe(self._deliver_error, deliver, error)
            else:
                self._deliver_error(deliver, error)

        def finish() -> None:
            """The tail: it runs once the last pass's read-backs have landed."""
            try:
                # A deferred array's bytes are still on the device: bring them home before
                # the tail (or the caller) reads the array.
                flush_deferred(getattr(plan, "deferred_arrays", None) or {})
                tail_result: object = self._run_host_region(
                    list(plan.launch_meta.get("body_tail")
                         or plan.launch_meta.get("phase_tail") or []),
                    args, instance, plan.carried_slots, plan.gap_slots,
                    created_objects=plan.created_objects, outer_state=plan.outer_state,
                    writable_outer={entry["name"]: entry["buffer"]
                                    for entry in plan.outer_writable})
                self._write_back_outer(plan.outer_state)
                if plan.object_binding.written:
                    plan.object_binding.write_back()
                if plan.return_state and int(plan.return_state["claim"][0]):
                    result: object = self._finish_returns(
                        plan.return_state, plan.returns, plan.full, plan.created)
                elif tail_result is not None:
                    result = tail_result
                else:
                    result = (self._finish_returns(
                        plan.return_state, plan.returns, plan.full, plan.created)
                        if plan.return_state else None)
            except BaseException as error:  # noqa: BLE001
                settle_error(error)
                return
            settle_result(result)

        def run_phase(position: int) -> None:
            try:
                # The chain runs on a worker thread, and a CUDA/HIP driver call needs the
                # device's context bound to the calling thread.
                context: object = getattr(plan.device_obj, "default_context", None)
                if context is not None and hasattr(context, "make_current"):
                    context.make_current()
                if position:
                    # The previous pass wrote the shared slots on the device; the region
                    # between them reads those on the host, so the copies must land first.
                    # This runs on a worker, never inside a stream callback.
                    self._sync_device(plan.backend, plan.device_obj)
                    self._run_phase_region(plan, position, args, instance)
                phase_args = self._phase_arguments(plan, position)
                phase_kernel: object = plan.active_kernels[position]
                bound = (phase_kernel[plan.phase_geometry]
                         if plan.phase_geometry is not None else phase_kernel)
                inner: object = bound.submit(*phase_args)
            except BaseException as error:  # noqa: BLE001
                settle_error(error)
                return

            def advance(done: object) -> None:
                try:
                    done.result()
                except BaseException as error:  # noqa: BLE001
                    _CHAIN_WORKERS.submit(settle_error, error)
                    return
                if position + 1 < total:
                    _CHAIN_WORKERS.submit(run_phase, position + 1)
                else:
                    _CHAIN_WORKERS.submit(finish)

            inner.add_done_callback(advance)

        _CHAIN_WORKERS.submit(run_phase, 0)
        return deliver

    def _run_phase_region(self, plan: object, position: int,
                          args: Tuple[object, ...], instance: object) -> None:
        """The host statements between pass ``position - 1`` and ``position``."""
        regions: List[List[ast.stmt]] = list(plan.launch_meta.get("phase_gaps") or [])
        if position - 1 >= len(regions):
            return
        region_writable: Dict[str, object] = {
            entry["name"]: entry["buffer"] for entry in plan.outer_writable}
        self._run_host_region(regions[position - 1], args, instance,
                              plan.carried_slots, plan.gap_slots,
                              plan.gap_namespaces[position - 1][0],
                              plan.gap_namespaces[position - 1][1],
                              plan.created_objects, plan.outer_state, region_writable,
                              parameter_slots=plan.parameter_slots,
                              parameter_values=plan.full,
                              written_parameters=plan.written_parameters,
                              region_values=plan.host_region_values)

    def _phase_arguments(self, plan: object, position: int) -> List[object]:
        """One pass's argument list, aligned to that pass's own signature."""
        phase_shapes: List[Tuple[int, int]] = (
            plan.kernel_shapes[position] if position < len(plan.kernel_shapes) else [])
        phase_created: List[str] = (
            plan.kernel_created[position] if position < len(plan.kernel_created) else [])
        claims: bool = (plan.phase_claims[position]
                        if position < len(plan.phase_claims) else True)
        return self._build_phase_args(plan, phase_shapes, phase_created, position, claims)
