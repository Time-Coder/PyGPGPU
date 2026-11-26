# One kernel per block site: build it once, launch it there every time.
"""A block usually sits inside a hot loop -- that is the whole point of writing it inline.

`PyKernel` keeps its compiled form in a per-instance cache, so building a new `PyKernel` on
every execution of the `with` statement would re-translate the block every time. The rebuilt
function and its `PyKernel` are therefore cached per **site** (the block's own identity in
the file), and one site always launches through one kernel.

The cache key is the file, the `with` line, the block's text and the options digest. The
text matters because an edit to a block (a new statement, a changed body) must not serve the
old translation; the options matter because `int=`/`float=`/`default_device=` change what the
kernel *is*.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from ..decorator import PyKernel
from ..decorator.devices import _snapshot_backend
from .rebuild import build_function
from .site import InlineBlock, text_of


class _SiteEntry:
    """What one block site keeps between launches: its kernel and its launch shape."""

    def __init__(self, kernel:PyKernel, names:List[str],
                 global_size:Optional[Tuple[int, ...]],
                 local_size:Optional[Tuple[int, ...]])->None:
        self.kernel:PyKernel = kernel
        self.names:List[str] = names
        self.global_size:Optional[Tuple[int, ...]] = global_size
        self.local_size:Optional[Tuple[int, ...]] = local_size


# Keyed by (file, with line, block text, options). The file and line say which site; the two
# digests say whether what lives there is still the same kernel.
_SITES:Dict[Tuple[str, int, str, str], _SiteEntry] = {}


def _digest(text:str)->str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _options_digest(spec:object)->str:
    """A stable spelling of the options that shape the generated kernel."""
    return "|".join([
        f"device={getattr(spec, 'default_devices', None)!r}",
        f"domain={getattr(spec, 'domain', None)!r}",
        f"types={sorted((getattr(spec, 'type_map', None) or {}).items())!r}",
        f"size={getattr(spec, 'global_size', None)!r}/{getattr(spec, 'local_size', None)!r}",
    ])


def _site_key(site:InlineBlock, text:str)->Tuple[str, int, str, str]:
    return (site.filename, site.node.lineno,
            _digest(text), _digest(_options_digest(site.spec)))


def _build(site:InlineBlock, text:str)->PyKernel:
    """The block's function, bound to the ordinary `@kernel` pipeline."""
    globals_dict:Dict[str, object] = site.frame.f_globals  # type: ignore[union-attr]
    func:object = build_function(text, site.filename, site.node.lineno, globals_dict)
    spec:object = site.spec
    # The rebuilt function's name is `inline_block_<line>`, which is what the generated C
    # name and the module-level cache file are keyed on. The options are the head's, so what
    # comes back is the kernel `@kernel(...)` would have produced over that function.
    return PyKernel(func, _snapshot_backend(spec.default_devices),
                    default_devices=spec.default_devices, domain=spec.domain,
                    type_map=spec.type_map)


def launch_block(site:InlineBlock)->object:
    """Launch one captured block: build its kernel once per site, then call it."""
    text:str = text_of(site)
    key:Tuple[str, int, str, str] = _site_key(site, text)
    entry:Optional[_SiteEntry] = _SITES.get(key)
    if entry is None:
        spec:object = site.spec
        entry = _SiteEntry(_build(site, text), list(site.names),
                           spec.global_size, spec.local_size)
        _SITES[key] = entry
    assert site.frame is not None
    values:Tuple[object, ...] = tuple(site.frame.f_locals[name] for name in entry.names)
    # A block written inside a method reads `self`, and that receiver travels the way a
    # method kernel's does: separately from the arguments, as the structure its attributes
    # describe. `captured_names` hoists it to the front, which is where the launch looks.
    instance:Optional[object] = None
    args:Tuple[object, ...] = values
    if entry.names and entry.names[0] in ("self", "cls"):
        instance, args = values[0], values[1:]
    return entry.kernel._invoke(instance, args, None, None,
                                entry.global_size, entry.local_size)
