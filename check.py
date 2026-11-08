# -*- coding: utf-8 -*-
"""Repo-wide unified check: runs ruff check, ty check, then the file-size budget.

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
# The size budget is last because it is the cheapest: ruff and ty should have their say about a
# file before it is judged on its length. See docs/file-split-plan.md.
subprocess.check_call([sys.executable, "line_budget.py"])