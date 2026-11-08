"""The state one launch shares between its phases.

A plain namespace, not a class with declared fields: the phases bind about a hundred names
between them, and a declared field for each of them would be a second place to keep in step
with the code that writes it.

The attributes are whatever the phases set. `__getattr__` answers `None` for a name that has
not been set yet, so a phase may read a value the phase before it did not produce -- the
same as a local that was never assigned would NOT be, which is why the reading is explicit
about what it expects.
"""


class LaunchPlan:
    """The state one launch carries from phase to phase."""

    def __getattr__(self, name: str) -> object:
        # an unset attribute reads as None: the phases check the ones they depend on, and
        # a name they did not expect is a bug the check reports, not a crash here.
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def __repr__(self) -> str:
        return f"<LaunchPlan backend={getattr(self, 'backend', '')!r}>"


#: The answer the interpreter backend gives the first phase, so the facade knows the call
#: is already finished. A kernel that returns nothing answers None, so None cannot be the
#: signal.
ANSWERED = object()
