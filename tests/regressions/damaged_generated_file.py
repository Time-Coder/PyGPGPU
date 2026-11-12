"""Bug: a damaged generated file is never repaired, because its digest still matches.

The user contract is that caches invalidate themselves: a user never deletes `__*cache__`.
The freshness test (`source_paths/writing.py`) decides a generated file is current by
comparing the digest sidecar against the digest of the python module that wrote it. That
answers "does this file describe today's python source?" -- and nothing else. Every other
way a file on disk can lose text is then permanent for the life of the cache:

* a file from a *previous package version* whose generated shape changed (the forward-
  declaration block did not exist before the owner-identity fix; a file written then keeps
  its old digest, which still matches, and the missing block is never re-added);
* a crashed or interrupted write, a truncated file, an editor or an indexer that saved a
  partial copy -- any loss the digest cannot see, because the sidecar describes the python,
  not the generated text.

Both were observed live this session, cured only by hand-deleting the cache -- which is the
one thing the contract says a user never does. `repro_damaged_file.py` plants the exact
damage: it strips the forward-declaration block out of a generated file and leaves the
digest sidecar untouched. The next build refuses the module

    opencl: unresolved extern / implicit declaration of the sample helper
    cuda:   `error: identifier "_AnswersOne_sample_<hash>" is undefined`

and after the failed build the file is *still* damaged (the block was not re-added), so
every later launch fails the same way until something deletes the cache by hand.

The right answer is that the digest has to describe the generated text as well as the
python: stamp the digest of the assembled source beside it (or hash the file on read and
compare against what this render would write), so "the file on disk is not what this
package version writes" restarts the file. A cheaper first step that fixes the observed
shape: whenever a build fails, rewrite the failed file from the current render before
reporting, so the next launch self-heals.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class Base:
    """One kernel, two subclasses; the sample helpers live in the callers' module."""

    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.image = np.zeros((height, width), dtype=np.float32)

    def sample(self, i: int, j: int) -> np.float32:
        """The hook a subclass overrides."""
        return np.float32(0)

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.image.shape[0], self.image.shape[1]):
            self.image[i, j] = self.sample(i, j)
