"""The state one compilation shares between its phases.

A plain namespace, like the launch's: the parts bind about seventy names between them, and
a declared field for each of them would be a second place to keep in step with the code
that writes it.

The attributes are whatever the parts set. `__getattr__` answers `None` for a name that has
not been set yet.
"""


class CompilePlan:
    """The state one compilation carries from part to part."""

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def __repr__(self) -> str:
        return f"<CompilePlan backend={getattr(self, 'backend', '')!r}>"
