# -*- coding: utf-8 -*-
"""Repo-wide unified check: runs ruff check then ty check.

Usage:
    python check.py
"""

from __future__ import annotations

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "ruff", "check", "--fix"])
subprocess.check_call([sys.executable, "-m", "ty", "check"])