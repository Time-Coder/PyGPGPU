# -*- coding: utf-8 -*-
"""Repo-wide unified check: ruff, ty, the file-size budget, the `Any` budget, then the
duplicate-binding check.

Usage:
    python check.py
"""

from __future__ import annotations

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "ruff", "check", "--fix"])
# Warnings are the ctypes-layer inference noise documented in pyproject.toml; only error-level
# diagnostics should fail the run.
subprocess.check_call([sys.executable, "-m", "ty", "check", "--exit-zero-on-warning"])
# The budgets are next because they are cheap: ruff and ty should have their say about a file
# before it is judged on its length or its annotations. The policies live in AGENTS.md and in
# the scripts themselves.
subprocess.check_call([sys.executable, "line_budget.py"])
subprocess.check_call([sys.executable, "any_budget.py"])
# Last, because it is a property of a tree ruff and ty have already read: one module-level name
# bound twice to one value is a copy-paste artifact neither of them reports.
subprocess.check_call([sys.executable, "dup_defs.py"])