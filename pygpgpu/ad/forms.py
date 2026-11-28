# The forms: the objects `k.grad()`, `k.value_and_grad()`, `k.vjp()` and `k.jvp()` return.
#
# The transform hangs on the kernel object as the same two-call shape the bracket already
# uses -- `k[(4, 256)](args)` names a launch, `k.grad(loss="loss")(args)` names a
# derivative -- and each form is a small object holding the configuration, callable with
# the kernel's own parameters. `PyKernel` carries the four properties that build these;
# nothing here is on the import path of a plain launch.
from __future__ import annotations

import ast
import inspect
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..kernel.translator.source_text import _source_first_line, kernel_source_text
from .build import (
    RETURN_SPELL,
    _build_grad_kernel,
    _check_tracked_members,
    _gradient_buffers,
    _returned_carried_scalar,
    _scan_of,
)
from .body_scan import BodyScan, parameter_names_of
from .errors import ad_error
from .multi_pass import MultiPassPlan
from .reverse import OutCotangent, SEED_COT, SEED_ONE, SEED_PARAM
from . import rules as rule_table


class _ADForm:
    """The configuring call of one derivative spelling, and the call it builds."""

    def __init__(self, kernel:object, instance:object = None,
                 backend:Optional[str] = None, wrt:Optional[Sequence[str]] = None):
        self.kernel:object = kernel
        self.instance:object = instance
        self.backend:Optional[str] = backend
        self.wrt:Optional[Sequence[str]] = wrt
        # A form straight from the property (or the facade) has not been configured yet:
        # its first call must name the derivative, which is what keeps `k.grad(x, out)`
        # -- a positional configuring call -- a refusal rather than a surprising launch.
        self.fresh:bool = (wrt is None and not self._has_config())

    def _has_config(self)->bool:
        """True when the subclass constructor carried configuration keywords."""
        return False

    # ------------------------------------------------------------------ bracket

    def __getitem__(self, selector:object)->"_ADForm":
        """`k.grad()["cuda"](...)` -- the same selector spelling the kernel's bracket takes."""
        if not isinstance(selector, str):
            raise TypeError("a derivative form names a backend with a string selector")
        form:_ADForm = self._copy()
        form.backend = selector
        return form

    def _copy(self)->"_ADForm":
        form:_ADForm = type(self)(self.kernel, self.instance, self.backend, self.wrt)
        form.fresh = self.fresh
        return form

    # ------------------------------------------------------------------ plumbing

    def _params(self)->List[str]:
        """The kernel's positional parameter names, receiver excluded."""
        return parameter_names_of(self.kernel.raw_func)

    def _bind(self, args:Tuple[object, ...],
              kwargs:Dict[str, object])->List[Tuple[str, object]]:
        """The call's arguments as (parameter name, value) pairs, receiver first."""
        signature = inspect.signature(self.kernel.raw_func)
        call_args:Tuple[object, ...] = args
        if self.instance is not None:
            call_args = (self.instance,) + tuple(args)
        bound = signature.bind(*call_args, **kwargs)
        return list(bound.arguments.items())

    def _positional(self, arguments:List[Tuple[str, object]])->List[object]:
        """The argument values in the kernel's own parameter order."""
        return [value for _name, value in arguments]

    def _resolve_wrt(self, arguments:List[Tuple[str, object]])->List[str]:
        """The parameters the derivative is taken of, inferred or spelled.

        The array flag (`requires_grad`) is the declaration; an explicit `wrt=` overrides
        it. An empty set is refused -- a derivative with respect to nothing is a mistake,
        not an empty answer.
        """
        if self.wrt is not None:
            names:List[str] = list(self.wrt)
            known:Set[str] = set(name for name, _ in arguments)
            for wrt_name in names:
                if wrt_name not in known:
                    raise self._error(f"`wrt` names `{wrt_name}`, which is not one of "
                                      f"the kernel's parameters")
            if not names:
                raise self._error("`wrt=()` is refused: a derivative with respect to "
                                  "nothing is a mistake, not an empty answer")
            return names
        inferred:List[str] = []
        for name_, value in arguments:
            if isinstance(value, np.ndarray) \
                    and bool(getattr(value, "requires_grad", False)):
                inferred.append(name_)
        if not inferred:
            raise self._error(
                "no argument carries `requires_grad`, and no `wrt` was named: a "
                "derivative with respect to nothing is refused -- flag the arrays "
                "(`x.requires_grad = True`) or spell `wrt=(...)`")
        return inferred

    def _adjoint(self, key:Tuple[object, ...], builder:Callable[[], object])->object:
        """The adjoint kernel for one configuration, cached on the parent kernel.

        A rule fix moves the transformed body while the user's `.py` has not moved, so the
        rules' version is part of the key beside the configuration itself.
        """
        cache:Dict[Tuple[object, ...], object] = getattr(self.kernel, "_ad_forms", None)
        if cache is None:
            cache = {}
            self.kernel._ad_forms = cache
        if key not in cache:
            cache[key] = builder()
        return cache[key]

    def _error(self, message:str, node:Optional[ast.AST] = None)->SyntaxError:
        code:object = getattr(self.kernel.raw_func, "__code__", None)
        filename:str = getattr(code, "co_filename", "<kernel>") or "<kernel>"
        return ad_error(message, node, filename,
                        kernel_source_text(self.kernel.raw_func),
                        _source_first_line(self.kernel.raw_func))

    def __call__(self, *args:object, **kwargs:object)->object:
        """The two-call shape: the first call names the derivative, the second launches.

        A call with no positional arguments is the configuring call -- keywords only
        (`loss=`, `out=`, `tangent=`, `wrt=`) -- and answers a configured form. A call
        with positional arguments is the kernel's own call, on a form a configuring call
        produced. `k.grad()(x, t)` therefore reads as it is written: configure with
        nothing, then launch.
        """
        if not args:
            return self._configured(kwargs)
        if self.fresh:
            raise self._error(
                "the configuring call takes keywords only: the first call names the "
                "derivative and the second is the kernel's own call -- spell "
                f"`{self._spell()}(...)(args)`")
        return self._launch_form(args, kwargs)

    def submit(self, *args:object, **kwargs:object)->object:
        """Launch without waiting, returning the adjoint launch's future.

        The fused forms run ONE adjoint kernel per call -- forward recomputed inside it,
        reverse walked after -- so the future resolving IS "settled after the last adjoint
        pass": the gradient buffers are only valid once `.result()` has landed, and the
        buffers themselves travel as the launch arguments, not as the future's value.
        """
        return self._deferred(args, kwargs, "submit")

    def async_call(self, *args:object, **kwargs:object)->object:
        """The asyncio counterpart of `submit`; await the returned future."""
        return self._deferred(args, kwargs, "async")

    def _deferred(self, args:Tuple[object, ...], kwargs:Dict[str, object],
                  launch:str)->object:
        if self.fresh:
            raise self._error(
                "the configuring call takes keywords only: the first call names the "
                "derivative and the second is the kernel's own call -- spell "
                f"`{self._spell()}(...)(args)`")
        return self._launch_form(args, kwargs, launch=launch)

    def _spell(self)->str:
        return "k.grad"

    def _config_keys(self)->Set[str]:
        raise NotImplementedError

    def _configured(self, kwargs:Dict[str, object])->"_ADForm":
        """The configuring call: keywords name the derivative, and the answer is a form."""
        unknown:Set[str] = set(kwargs) - self._config_keys()
        if unknown:
            raise self._error(
                f"the configuring call takes {sorted(self._config_keys())}; it was "
                f"given {sorted(unknown)}")
        form:_ADForm = self._copy()
        for key, value in kwargs.items():
            setattr(form, key, value)
        form.fresh = False
        return form

    def _launch_form(self, args:Tuple[object, ...], kwargs:Dict[str, object],
                     launch:str = "call")->object:
        raise NotImplementedError

    def _multi_pass(self, key:Tuple[object, ...], wrt:List[str],
                    loss_name:Optional[str], seed_mode:str,
                    cot_of_out:Optional[OutCotangent])->object:
        """The multi-pass plan for one configuration, cached like an adjoint kernel."""
        plan_key:Tuple[object, ...] = ("plan",) + key
        plan:object = self._adjoint(plan_key, lambda: MultiPassPlan(
            self.kernel, wrt, loss_name, seed_mode, cot_of_out))
        return plan

    def _launch(self, adjoint:object, positional:List[object], extra:List[object],
                launch:str = "call")->object:
        """Launch the adjoint kernel with the caller's values and the form's arrays."""
        call_args:List[object] = list(positional)
        if self.instance is not None:
            call_args = [self.instance] + call_args
        call_args.extend(extra)
        bracket:object = adjoint[self.backend] if self.backend else adjoint
        if launch == "submit":
            return bracket.submit(*call_args)
        if launch == "async":
            return bracket.async_call(*call_args)
        return bracket(*call_args)

    def _forward(self, args:Tuple[object, ...], kwargs:Dict[str, object])->object:
        """The plain forward call, as the kernel itself would run it."""
        call_args:Tuple[object, ...] = args
        if self.instance is not None:
            call_args = (self.instance,) + tuple(args)
        if self.backend:
            return self.kernel[self.backend](*call_args, **kwargs)
        return self.kernel(*call_args, **kwargs)


class GradForm(_ADForm):
    """`k.grad(...)` -- the fused reverse form: one kernel, gradients only."""

    def __init__(self, kernel:object, instance:object = None,
                 backend:Optional[str] = None, wrt:Optional[Sequence[str]] = None,
                 loss:object = RETURN_SPELL):
        self.loss:object = loss
        super().__init__(kernel, instance, backend, wrt)

    def _has_config(self)->bool:
        return self.loss != RETURN_SPELL

    def _spell(self)->str:
        return "k.grad"

    def _config_keys(self)->Set[str]:
        return {"loss", "wrt"}

    def _copy(self)->"_ADForm":
        form:GradForm = GradForm(self.kernel, self.instance, self.backend, self.wrt,
                                 self.loss)
        form.fresh = self.fresh
        return form

    def _launch_form(self, args:Tuple[object, ...], kwargs:Dict[str, object],
                     launch:str = "call")->object:
        arguments:List[Tuple[str, object]] = self._bind(args, kwargs)
        _check_tracked_members(self, arguments)
        wrt:List[str] = self._resolve_wrt(arguments)
        loss_name:Optional[str] = self._resolve_loss()
        scan:BodyScan = _scan_of(self.kernel)
        if len(scan.dispatches) > 1:
            plan:object = self._multi_pass(self._key(wrt), wrt, loss_name,
                                           SEED_ONE, None)
            gradients:Dict[str, object] = plan.launch(self, arguments, None,
                                                      forward=True, raw_args=args,
                                                      raw_kwargs=kwargs)
            return tuple(gradients[name] for name in wrt)
        gradients:Dict[str, object] = _gradient_buffers(self, arguments, wrt,
                                                        mode="reverse")
        adjoint:object = self._adjoint(self._key(wrt), lambda: self._build(
            wrt, loss_name, seed_mode=SEED_ONE, cot_param=None, ret_cot_param=None,
            cot_of_out=None))
        settled:object = self._launch(adjoint, self._positional(arguments),
                                      [gradients[wrt_name] for wrt_name in wrt],
                                      launch=launch)
        if launch != "call":
            # A deferred launch answers its future: the gradient buffers travel as the
            # launch arguments and are only valid once the future has landed.
            return settled
        return tuple(gradients[wrt_name] for wrt_name in wrt)

    def _key(self, wrt:List[str])->Tuple[object, ...]:
        return ("grad", tuple(wrt), self.loss, rule_table.RULES_VERSION,
                tuple(sorted(self.kernel._type_map.items())))

    def _resolve_loss(self)->Optional[str]:
        """The single number the gradient is of: the body's return, or a named parameter."""
        if self.loss == RETURN_SPELL:
            return _returned_carried_scalar(self._scan(), self)
        if not isinstance(self.loss, str):
            raise self._error("`loss=` names a parameter, or \"return\" for the value "
                              "the body returned")
        if self.loss not in self._params():
            raise self._error(f"`loss={self.loss!r}` does not name a parameter of this "
                              f"kernel")
        return self.loss

    def _scan(self)->BodyScan:
        return _scan_of(self.kernel)

    def _build(self, wrt:List[str], loss_name:Optional[str], seed_mode:str,
               cot_param:Optional[str], ret_cot_param:Optional[str],
               cot_of_out:Optional[OutCotangent])->object:
        return _build_grad_kernel(
            self.kernel, wrt, loss_name, seed_mode,
            {wrt_name: f"_g_{wrt_name}" for wrt_name in wrt},
            cot_param, ret_cot_param, cot_of_out,
            allow_scatter=False, drop_launch_wide=False,
            suffix="_grad" if seed_mode == SEED_ONE else "_pullback")


class ValueAndGradForm(GradForm):
    """`k.value_and_grad(...)` -- the same fused kernel, also answering the loss value."""

    def _copy(self)->"_ADForm":
        form:ValueAndGradForm = ValueAndGradForm(self.kernel, self.instance,
                                                 self.backend, self.wrt, self.loss)
        form.fresh = self.fresh
        return form

    def _launch_form(self, args:Tuple[object, ...], kwargs:Dict[str, object],
                     launch:str = "call")->object:
        if launch != "call":
            raise self._error(
                "value_and_grad answers the loss VALUE the launch computes, which a "
                "deferred launch cannot carry: use grad().submit(...) and read the loss "
                "separately, or the synchronous spelling")
        arguments:List[Tuple[str, object]] = self._bind(args, kwargs)
        wrt:List[str] = self._resolve_wrt(arguments)
        values:Dict[str, object] = dict(arguments)
        loss_name:Optional[str] = self._resolve_loss()
        scan:BodyScan = _scan_of(self.kernel)
        if len(scan.dispatches) > 1:
            plan = self._multi_pass(self._key(wrt), wrt, loss_name, SEED_ONE, None)
            gradients = plan.launch(self, arguments, None, forward=True,
                                    raw_args=args, raw_kwargs=kwargs)
            loss_value:object = values[self.loss]
            return loss_value, tuple(gradients[name] for name in wrt)
        gradients:Dict[str, object] = _gradient_buffers(self, arguments, wrt,
                                                        mode="reverse")
        adjoint:object = self._adjoint(self._key(wrt), lambda: self._build(
            wrt, loss_name, seed_mode=SEED_ONE, cot_param=None, ret_cot_param=None,
            cot_of_out=None))
        result:object = self._launch(adjoint, self._positional(arguments),
                                     [gradients[wrt_name] for wrt_name in wrt])
        loss_value:object = result if self.loss == RETURN_SPELL else values[self.loss]
        return loss_value, tuple(gradients[wrt_name] for wrt_name in wrt)


class VJPForm(_ADForm):
    """`k.vjp(...)` -- forward as usual; the pullback runs the adjoint with a weight."""

    def __init__(self, kernel:object, instance:object = None,
                 backend:Optional[str] = None, wrt:Optional[Sequence[str]] = None,
                 out:object = RETURN_SPELL):
        self.out:object = out
        super().__init__(kernel, instance, backend, wrt)

    def _has_config(self)->bool:
        return self.out != RETURN_SPELL

    def _spell(self)->str:
        return "k.vjp"

    def _config_keys(self)->Set[str]:
        return {"out", "wrt"}

    def _copy(self)->"_ADForm":
        form:VJPForm = VJPForm(self.kernel, self.instance, self.backend, self.wrt,
                               self.out)
        form.fresh = self.fresh
        return form

    def _launch_form(self, args:Tuple[object, ...], kwargs:Dict[str, object],
                     launch:str = "call")->object:
        if launch != "call":
            raise self._error(
                "vjp's forward runs synchronously and its pullback is a closure over the "
                "bound arguments: a deferred spelling would settle the forward while the "
                "cotangent is still unknown -- use the synchronous spelling")
        arguments:List[Tuple[str, object]] = self._bind(args, kwargs)
        _check_tracked_members(self, arguments)
        wrt:List[str] = self._resolve_wrt(arguments)
        out_name:Optional[str] = self._resolve_out()
        form:VJPForm = self
        scan:BodyScan = _scan_of(self.kernel)
        if len(scan.dispatches) > 1:
            forward_result:object = self._forward(args, kwargs)
            if self.out != RETURN_SPELL:
                forward_result = dict(arguments)[self.out]

            def multi_pullback(weight:object)->Tuple[object, ...]:
                if not isinstance(weight, np.ndarray):
                    raise form._error("this kernel answers an array; the pullback takes "
                                      "a weight array of the output's shape")
                plan:object = form._multi_pass(
                    form._key(wrt, f"array:{out_name}"), wrt, None, SEED_COT,
                    OutCotangent(out_name, "_c_out"))
                gradients:Dict[str, object] = plan.launch(form, arguments, weight,
                                                          forward=False,
                                                          raw_args=args,
                                                          raw_kwargs={})
                return tuple(gradients[name] for name in wrt)

            return forward_result, multi_pullback

        def pullback(weight:object)->Tuple[object, ...]:
            """`J^T g` for any weight `g`: one fused launch, forward recomputed."""
            gradients:Dict[str, object] = _gradient_buffers(form, arguments, wrt,
                                                            mode="reverse")
            if out_name is None:
                if not isinstance(weight, (int, float, np.floating, np.integer)):
                    raise form._error("this kernel answers a scalar; the pullback takes "
                                      "the scalar weight `d loss / d return`")
                adjoint:object = form._adjoint(form._key(wrt, "scalar"), lambda:
                                               form._build(
                                                   wrt, None, seed_mode=SEED_PARAM,
                                                   cot_param=None, ret_cot_param="_c_ret",
                                                   cot_of_out=None))
                form._launch(adjoint, form._positional(arguments),
                             [gradients[w] for w in wrt] + [weight])
            else:
                if not isinstance(weight, np.ndarray):
                    raise form._error(
                        "this kernel answers an array; the pullback takes a weight array "
                        "of the output's shape (`gnp.ones_like(result)` seeds sum(result))")
                adjoint = form._adjoint(form._key(wrt, f"array:{out_name}"), lambda:
                                        form._build(
                                            wrt, None, seed_mode=SEED_COT,
                                            cot_param="_c_out", ret_cot_param=None,
                                            cot_of_out=OutCotangent(out_name, "_c_out")))
                form._launch(adjoint, form._positional(arguments),
                             [gradients[w] for w in wrt] + [weight])
            return tuple(gradients[wrt_name] for wrt_name in wrt)

        # The forward runs once, as the kernel itself: the result is the object it
        # returns -- or, when the body answers through a named parameter, the caller's own
        # object holding it (symmetry with a functional call, the plan's 2.4).
        forward_result:object = self._forward(args, kwargs)
        if self.out != RETURN_SPELL:
            forward_result = dict(self._bind(args, kwargs))[self.out]
        return forward_result, pullback

    def _key(self, wrt:List[str], kind:str)->Tuple[object, ...]:
        return ("vjp", tuple(wrt), kind, self.out, rule_table.RULES_VERSION,
                tuple(sorted(self.kernel._type_map.items())))

    def _resolve_out(self)->Optional[str]:
        """The output the pullback weights: the body's return, or a named parameter."""
        if self.out == RETURN_SPELL:
            scan:BodyScan = _scan_of(self.kernel)
            returns:List[ast.Return] = scan.returns()
            if not returns:
                raise self._error("the body returns nothing, and no `out=` was named: "
                                  "spell `out=\"name\"` for the parameter that carries "
                                  "the result")
            names:Set[str] = set()
            for ret in returns:
                if not isinstance(ret.value, ast.Name):
                    raise self._error("the body's `return` carries an expression; the "
                                      "pullback weights one named result -- spell "
                                      "`out=\"name\"`")
                names.add(ret.value.id)
            if len(names) != 1:
                raise self._error("the body's returns do not agree on one named result, "
                                  "which the pullback needs; spell `out=\"name\"`")
            out_name:str = names.pop()
            if out_name in self._params():
                return out_name
            if out_name in scan.carried_scalars:
                return None            # a scalar result: the pullback takes a scalar weight
            raise self._error(
                f"the body returns `{out_name}`, which is neither a parameter nor a "
                f"launch-wide scalar; spell `out=\"name\"` for the parameter that "
                f"carries the result")
        if not isinstance(self.out, str) or self.out not in self._params():
            raise self._error(f"`out={self.out!r}` does not name a parameter of this "
                              f"kernel")
        return self.out

    def _build(self, wrt:List[str], loss_name:Optional[str], seed_mode:str,
               cot_param:Optional[str], ret_cot_param:Optional[str],
               cot_of_out:Optional[OutCotangent])->object:
        return _build_grad_kernel(
            self.kernel, wrt, None, seed_mode,
            {wrt_name: f"_g_{wrt_name}" for wrt_name in wrt},
            cot_param, ret_cot_param, cot_of_out,
            allow_scatter=True, drop_launch_wide=False, suffix="_pullback")


