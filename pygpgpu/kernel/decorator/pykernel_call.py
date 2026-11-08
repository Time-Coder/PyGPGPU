# The call surface around the launch: entry, host regions, the python backend.
from __future__ import annotations
import ast
import copy
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
import numpy as np
from ..python_backend import PythonKernel
from ...numpy import ndarray as gnp
from ..translator import (
    _assigned_name_of,
)
from .frames import (
    _caller_traceback,
    _user_traceback,
)
from .call_proxies import (
    _TAIL_RETURN_NAME,
    _TailReturn,
    _TailReturnRewriter,
)
from .host_regions import (
    _positional_argument_names,
    _receiver_name,
)
from ._typing import (
    OuterState,
)


class _PykernelCallMixin:

    def _invoke(self, instance:object, args:Tuple[object, ...],
                backend_sel:Optional[str], device_sel:Optional[str],
                global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                launch:str = "call")->object:
        try:
            return self._invoke_inner(instance, args, backend_sel, device_sel,
                                      global_size, local_size, launch)
        except SyntaxError as caught:
            # Trim to the caller's frame and re-raise with that traceback. CPython prepends one
            # frame for the re-raise itself, which is why the exception carries the caller's
            # line rather than this one: the syntax rendering uses the fields on the
            # exception, not the chain.
            caller = _caller_traceback(caught.__traceback__)
            raise caught.with_traceback(caller) from None
        except ValueError as caught:
            # Two different things arrive as a ValueError, and python's own reading decides
            # which treatment each takes.
            #
            # An *argument* refusal -- an integer too wide for the type map (`_check_integer_fits`)
            # -- is about the call, not the body: it carries no source position, and it is
            # trimmed to the caller's frame the way a SyntaxError is. Falling through to the
            # body handler would dress it up as a failure inside the function.
            #
            # One the translator *positioned* (see `_position_error`) is a failure of the
            # body's own line -- an unpacking count that cannot match, a subscript that is
            # too wide -- and reads exactly as the interpreter reads it, so it gets the body
            # treatment below: the caller's line and then the line that failed.
            if not str(caught):
                raise
            if getattr(caught, "lineno", None):
                caught.__traceback__ = _user_traceback(caught, self.raw_func)
                raise caught from None
            caller = _caller_traceback(caught.__traceback__)
            raise caught.with_traceback(caller) from None
        except Exception as caught:
            # A failure of the *body* -- a wrong unpack, an out-of-range index -- has to read
            # exactly as it does without the decorator: the caller's line, then the line in
            # the function that failed. The body's own frame belongs to the execution
            # machinery, and the device backends never run the body at all, so it is
            # replaced by a stand-in carrying the same file, line and function name; the
            # library frames between the two are spliced out here.
            import traceback as _tb
            _tb.print_exception(type(caught), caught, caught.__traceback__)
            caught.__traceback__ = _user_traceback(caught, self.raw_func)
            raise caught from None


    def _run_host_region(self, statements:List[ast.stmt],
                         args:Tuple[object, ...], instance:object,
                         carried_slots:Dict[str, object], gap_slots:Dict[str, object],
                         globalns:Optional[Dict[str, object]]=None,
                         localns:Optional[Dict[str, object]]=None,
                         created_objects:Optional[Dict[str, object]]=None,
                         outer_state:Optional[OuterState]=None,
                         writable_outer:Optional[Dict[str, np.ndarray]]=None)->object:
        """Run one region of python statements on the host, and return a tail `return`.

        A region is the statements between two passes, or after the last one. A statement
        reads the launch arguments and whatever earlier statements and slots hold, exactly
        as python would. An array it touches is pulled back to the host first (the previous
        pass wrote it on the device) and marked dirty again afterwards, so the next launch
        pushes the changed values back. ``globalns``/``localns`` may pass a namespace in so
        consecutive regions share one scope, the way python's function body does.
        """
        if not statements:
            return None

        if globalns is None:
            globalns = dict(self.raw_func.__globals__)
        if localns is None:
            localns = {}
            for name, value in zip(_positional_argument_names(self.raw_func), args):
                localns[name] = value
            receiver:str = _receiver_name(self.raw_func)
            if receiver and instance is not None:
                localns[receiver] = instance
        # Closure cells the body reads (a factory returning a @kernel is the common shape)
        # and the outer-scope values the launch carries. A writable outer slot holds the
        # value the device may have updated, and a host write back into it travels to the
        # scope through the same buffer _write_back_outer copies from.
        for name, cell in zip(self.raw_func.__code__.co_freevars,
                              self.raw_func.__closure__ or ()):
            try:
                localns.setdefault(name, cell.cell_contents)
            except ValueError:
                continue                  # an empty cell: python would raise too
        for slot in (outer_state["writable"] if outer_state else []):
            localns[slot["name"]] = slot["buffer"][0]
        for name, buffer in (writable_outer or {}).items():
            localns[name] = buffer[0]
        # The slots the launch shares with the device: a carried scalar or a gap value that
        # an earlier phase updated is the current value, not the initial one. A created
        # array is the object itself -- indexing it would hand the statements its first row.
        for name, slot in list(carried_slots.items()) + list(gap_slots.items()):
            localns[name] = slot[0]
        # A created array is the object itself: unwrapping it would hand the statements its
        # first element (or row), which is not what python's name means.
        for name, value in (created_objects or {}).items():
            localns[name] = value

        # Every `return` in the region is rewritten into a raise before the statements are
        # compiled (see _TailReturnRewriter): a module body cannot hold a `return` at all,
        # and a region's `return` -- including one nested in an `if` -- is how a tail
        # answers a value. The rewriter runs on a copy, because these statements come from
        # the compiled cache and are shared by every call of the kernel.
        localns[_TAIL_RETURN_NAME] = _TailReturn
        rewriter:_TailReturnRewriter = _TailReturnRewriter()
        exec_statements:List[ast.stmt] = [
            rewriter.visit(copy.deepcopy(statement)) for statement in statements]

        touched:List[object] = []
        try:
            for statement, exec_statement in zip(statements, exec_statements):
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and node.id in localns:
                        value:object = localns[node.id]
                        if isinstance(value, gnp):
                            value.to_host()
                            if value not in touched:
                                touched.append(value)
                if isinstance(statement, ast.AugAssign):
                    # `count += 1` in a region updates the slot it names, the same way an
                    # assignment does; the shared helper does not answer for this spelling.
                    target_node:ast.expr = statement.target
                    target:Optional[str] = (target_node.id
                                            if isinstance(target_node, ast.Name) else None)
                else:
                    target:Optional[str] = _assigned_name_of(statement)
                # compile() needs a Module node, and the statement is one entry of the body.
                module:ast.Module = ast.Module(body=[exec_statement], type_ignores=[])
                exec(compile(module, "<kernel gap>", "exec"), globalns, localns)  # noqa: S102 - the kernel source itself
                if target is not None and target in localns:
                    value:object = localns[target]
                    if isinstance(value, gnp) and value not in touched:
                        touched.append(value)
                    elif isinstance(value, np.ndarray) and value.ndim > 0:
                        # A creation the gap statement built: the object itself is what the next
                        # pass receives, exactly as it does for a creation above the loops.
                        (created_objects if created_objects is not None else {})[target] = value
                    elif not isinstance(value, gnp):
                        # `or` cannot pick the slot: bool(np.array([0])) is False, which made a
                        # zero-valued slot read as missing and silently skipped the write-back.
                        slot:Optional[np.ndarray] = (gap_slots.get(target)
                                                     if gap_slots.get(target) is not None
                                                     else carried_slots.get(target))
                        if slot is None:
                            slot = (writable_outer or {}).get(target)
                        if slot is not None:
                            slot[0] = int(value)
                            localns[target] = slot[0]
        except _TailReturn as answered:
            return answered.value
        finally:
            # An array the region touched was pulled back to the host, so the next launch
            # must push it again -- on the `return` path too.
            for value in touched:
                value._set_host_dirty()
        return None


    def _sync_device(self, backend:str, device_obj:object)->None:
        """Wait until every launch this backend has queued has finished, callbacks included.

        The statements between two passes read shared slots on the host; the copies that
        fill them are stream callbacks, so a device-level sync is the one wait that is
        correct regardless of which stream each launch used.
        """
        if backend == "cuda":
            from ...cuda.driver import CUDA
            CUDA.cuCtxSynchronize()
        elif backend == "hip":
            from ...hip.driver import HIP
            HIP.hipDeviceSynchronize()
        elif backend == "metal":
            # Metal's launch waits on its own command buffer, so every launch that
            # returned has already completed; the queue is serial, which makes that
            # wait cover the launches before it too.
            device_obj.default_context.sync()


    def _invoke_python(self, instance:object, args:Tuple[object, ...],
                       device_sel:Optional[str],
                       global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                       launch:str)->object:
        """Run the body in the Python interpreter (no translation, no device).

        The backend is device-free, so a device selector in the bracket is rejected
        rather than ignored; the work size and launch style mean what they mean on the
        GPU backends. The runner decides what a work item is (see python_backend.py): a
        magic-loop kernel runs once as written, a gid-style kernel runs once per work
        item of its domain.
        """
        if device_sel is not None:
            raise ValueError(
                f"the python backend has no devices, so '{self.__name__}[\"python:{device_sel}\"]' "
                f"has no meaning; use {self.__name__}[\"python\"]")
        if self._is_method and instance is None:
            raise TypeError(f"{self.__name__}() requires an instance")
        runner:object = self._python_runner
        if runner is None:
            runner = PythonKernel(self.raw_func, self._is_method, self._self_param,
                                  domain=self._domain or "")
            self._python_runner = runner
        return runner.call(instance, args, global_size, local_size, launch=launch)


    @staticmethod
    def _deliver_result(future:object, result:object)->None:
        """Complete a future with a value, unless something already settled it."""
        if not future.done():
            future.set_result(result)


    @staticmethod
    def _deliver_error(future:object, error:BaseException)->None:
        """Complete a future with an exception, unless something already settled it."""
        if not future.done():
            future.set_exception(error)
