from __future__ import annotations
import inspect
import os
from ..translator import (
    short_hash,
)


class _PykernelIdentityMixin:

    def _type_map_key(self)->str:
        """A stable spelling of this kernel's `int=`/`float=` mapping, for cache keys."""
        if not self._type_map:
            return ""
        return ",".join(f"{key}={self._type_map[key]}"
                        for key in sorted(self._type_map))


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
            source = ""
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


    def _c_kernel_name(self, sig:str, backend:str, declaration:str="")->str:
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
        if declaration:
            payload += "|" + declaration
        return f"{base}_{short_hash(payload)}"
