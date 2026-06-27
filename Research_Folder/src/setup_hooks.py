#!/usr/bin/env python3
"""
CodeRefactor AI — Git Hooks Setup
==================================
Installs the pre-commit hook into the local .git/hooks/ directory.

Usage:
  python src/setup_hooks.py          # Install hooks
  python src/setup_hooks.py --remove # Remove hooks
"""

import os
import sys
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HOOKS_SOURCE = PROJECT_ROOT / "hooks" / "pre-commit"
HOOKS_TARGET = PROJECT_ROOT / ".git" / "hooks" / "pre-commit"


def install():
    """Install the pre-commit hook."""
    if not HOOKS_SOURCE.exists():
        print(f"  {red('✗')} Hook source not found: {HOOKS_SOURCE}")
        print(f"  Run this script from the project root directory.")
        sys.exit(1)

    if not HOOKS_TARGET.parent.exists():
        print(f"  {red('✗')} Not a git repository: {PROJECT_ROOT}")
        print(f"  Run 'git init' first, or run this from within a git repo.")
        sys.exit(1)

    # Copy the hook
    shutil.copy2(str(HOOKS_SOURCE), str(HOOKS_TARGET))
    HOOKS_TARGET.chmod(0o755)

    print(f"  {green('✓')} Pre-commit hook installed:")
    print(f"    {HOOKS_TARGET}")
    print()
    print(f"  The hook will now automatically analyze and refactor")
    print(f"  staged Python files before each commit.")
    print()
    print(f"  {yellow('Environment variables:')}")
    print(f"    OPENROUTER_API_KEY   Required for refactoring")
    print(f"    CODEREFACTOR_DRY_RUN Set to 1 for analysis only")
    print(f"    CODEREFACTOR_SKIP    Set to 1 to skip the hook")
    print()
    print(f"  To test: git add app.py && git commit -m \"test\"")
    print()


def remove():
    """Remove the pre-commit hook."""
    if HOOKS_TARGET.exists():
        HOOKS_TARGET.unlink()
        print(f"  {green('✓')} Pre-commit hook removed: {HOOKS_TARGET}")
    else:
        print(f"  No pre-commit hook found at {HOOKS_TARGET}")
    print()


def green(text):
    return f"\033[32m{text}\033[0m" if sys.stdout.isatty() else text


def yellow(text):
    return f"\033[33m{text}\033[0m" if sys.stdout.isatty() else text


def red(text):
    return f"\033[31m{text}\033[0m" if sys.stdout.isatty() else text


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        install()