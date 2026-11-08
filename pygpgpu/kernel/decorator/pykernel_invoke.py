from __future__ import annotations
from typing import (
    Optional,
    Tuple,
)

from .launch_plan import ANSWERED, LaunchPlan
from ._phase1 import _PrepareLaunchMixin
from ._phase2 import _BuildLaunchMixin
from ._phase3 import _RunLaunchMixin
from ._phase_async import _PhaseAsyncMixin
from ._build_phase_args import _BuildPhaseArgsMixin


class _PykernelInvokeMixin(
        _PrepareLaunchMixin, _BuildLaunchMixin, _RunLaunchMixin,
        _PhaseAsyncMixin, _BuildPhaseArgsMixin):


    def _invoke_inner(self, instance:object, args:Tuple[object, ...],
                      backend_sel:Optional[str], device_sel:Optional[str],
                      global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                      launch:str = "call")->object:
        """One launch, in three phases.

        The phases share their state through `plan`, a plain namespace: the first
        decides which kernel runs, the second packs the values the launch carries, and
        the third runs the phases and settles the result.
        """
        plan:object = LaunchPlan()
        answered:object = self._prepare_launch(plan, instance, args, backend_sel,
                                            device_sel, global_size, local_size, launch)
        if answered is ANSWERED:
            # the interpreter backend answers the call from the first phase
            return plan.result
        self._build_launch(plan, instance, args, backend_sel, device_sel,
                          global_size, local_size, launch)
        return self._run_launch(plan, instance, args, backend_sel, device_sel,
                            global_size, local_size, launch)
