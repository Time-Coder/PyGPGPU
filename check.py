# -*- coding: utf-8 -*-
"""Repo-wide unified check: ruff, ty, the file-size budget, then the `Any` budget.

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
# The two budgets are last because they are the cheapest: ruff and ty should have their say
# about a file before it is judged on its length or its annotations. See
# docs/file-split-plan.md and the type-annotation policy in AGENTS.md.
subprocess.check_call([sys.executable, "line_budget.py"])
subprocess.check_call([sys.executable, "any_budget.py"])