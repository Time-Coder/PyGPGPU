from __future__ import annotations
import inspect
import os
from typing import (
    Optional,
)
from ..translator import (
    short_hash,
)
from .helper_closure import (
    helper_closure_digest,
)


class _PykernelIdentityMixin:

    def _type_map_key(self)->str:
        """A stable spelling of this kernel's `int=`/`float=` mapping, for cache keys."""
        if not self._type_map:
            return ""
        return ",".join(f"{key}={self._type_map[key]}"
                        for key in sorted(self._type_map))


    def _helper_closure_digest(self)->str:
        """The source digest of the helper closure this kernel inlines ("" for none).

        Read live on every launch rather than cached on the instance: the answer follows
        the helper function objects the module globals hold *now*, so an edit to a helper
        module changes it on the next call without the kernel's own source moving. The
        walk is the same shape the launch already pays for per call
        (`baked_shared_shapes`): a cached parse and a plain AST pass.
        """
        return helper_closure_digest(self.raw_func, self._resolve_global)


    def _source_signature(self)->str:
        """Python source signature: source changes -> regenerate (recompile when cached content differs).

        `@kernel(domain=...)` is deliberately **not** part of this hash, and that is safe only
        because the domain never reaches the generated code: it picks which array supplies the
        launch geometry, and geometry is a host-side, per-call fact. Two decorators over one
        body with different domains therefore share one source (they even compute one C name),
        and each instance keeps its own `_cache` entry so neither reuses the other's launch.
        If the domain ever starts to shape the device code, it has to be added here -- otherwise
        the two would share a source they no longer agree on.

        The `int=`/`float=` mapping **is** in this hash, because it does shape the device
        code: it decides which C type a python `int`/`float` parameter is declared as, so two
        mappings are two different translation units. Without it here, a kernel declaring
        `int=` would share the cache file (and the C name) of the default one, and the second
        launch would compile against a signature the caller did not ask for.
        """
        try:
            source:str = inspect.getsource(self.raw_func)
        except OSError:
            # An AD-adjoint function has no file to read: its identity is the TRANSFORMED
            # TREE itself, so any change in the adjoint's body -- a rule fix, a transform
            # fix -- moves the signature, and with it the generated file and the binary
            # cache, even though the user's own source has not moved. The rules' version
            # travels with the dump, because the rules are part of what produced the tree.
            import ast as _ast  # noqa: PLC0415
            from ...ad.rules import RULES_VERSION  # noqa: PLC0415
            from ..translator.source_text import adjoint_tree_of  # noqa: PLC0415
            entry:object = adjoint_tree_of(self.raw_func)
            tree_dump:str = (_ast.dump(entry[0]) if entry is not None else "")
            source = f"ad={RULES_VERSION}\n{tree_dump}"
        return short_hash(f"{source}\nbackend={self.backend}\ntypes={self._type_map_key()}")


    def _resolve_global(self, name:str)->object:
        """Resolve a name the kernel body reads from outside the function.

        A closure cell takes precedence over a module global, exactly as python resolves
        it, so a nested kernel sees its enclosing function's variable (which is what the
        outer-scope feature parameterizes).
        """
        for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                   self.raw_func.__closure__ or ()):
            if cell_name == name:
                try:
                    return cell.cell_contents
                except ValueError:  # the cell is still empty
                    return None
        return self.raw_func.__globals__.get(name)


    def _c_kernel_name(self, sig:str, backend:str, declaration:str="",
                     owner:Optional[type]=None, helpers:str="")->str:
        """The kernel's C name, unique per module and per what it declares.

        Two modules can define a kernel with the same name and the same argument types --
        two test modules both declaring calls_a_method, say -- and one generated file
        #includes the other. Without the module in the hash the two kernels share one C
        name, and the source defines it twice.

        `declaration` is the rendered parameter list, when the caller already has one. The
        name has to cover it as well as `sig`: `sig` is built from the *argument types* the
        caller passed, while the generated signature also carries the extents derived from
        the body, the outer-scope values and the shared buffers. Without them in the hash a
        body whose derived signature changed kept its old name, and the file then held two
        declarations of that name. It is folded into the *same* hash rather than appended
        as a second one, so a kernel name stays `name_<hash>` (see docs/generated-source.md).

        `owner` is the class the *instance* belongs to, for a method kernel. Such a kernel's
        device body is not the decorated body alone: a body that calls `self.<hook>(...)`
        compiles against the hook the receiver's class supplies, and two subclasses of one
        `@kernel` supply different hooks over the identical signature. The class is
        therefore part of what the kernel *is* -- its absence is what let a second subclass
        reuse the first one's compiled binary, and silently answer the first one's numbers
        (see tests/regressions/subclass_kernel_identity.py). The in-memory cache key
        carries the same mark, so the two halves cannot disagree.

        `helpers` is the source digest of the helper closure the body inlines (see
        `helper_closure.helper_closure_digest`). A helper is monomorphized inside this
        kernel's own translation unit, so its text is this kernel's device code: a literal
        edited inside a helper has to change the name, or the generated file keeps its old
        bytes, the binary cache keeps its old key (it is keyed on those bytes), and a fresh
        process silently loads a binary built from the old helper. The in-process cache
        key folds the same digest (see _prepare_launch), so the two halves cannot disagree.
        """
        source_file:str = getattr(getattr(self.raw_func, "__code__", None),
                                  "co_filename", "") or ""
        module_mark:str = os.path.basename(source_file)
        # A method reads `Class_method`, exactly as a device helper does: the receiver is
        # the first parameter, so the class is part of what the kernel is. A plain function
        # keeps its own name.
        base:str = self.__name__
        qualname:str = getattr(self.raw_func, "__qualname__", "")
        if "." in qualname:
            owner_name:str = qualname.rsplit(".", 1)[0]
            if "<" not in owner_name:
                base = f"{owner_name}_{self.__name__}"
        payload:str = module_mark + "|" + sig + "_" + backend
        if owner is not None:
            payload += "|owner=" + f"{owner.__module__}.{owner.__qualname__}"
        if declaration:
            payload += "|" + declaration
        if helpers:
            payload += "|helpers=" + helpers
        return f"{base}_{short_hash(payload)}"
