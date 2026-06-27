---
name: incremental-code-smell-refactor
description: Set up an automated pipeline that detects changed Python files, analyzes them for code smells, auto-refactors bad-smell functions, and shows detailed per-function results in logs and PRs.
source: auto-skill
extracted_at: '2026-06-27T01:35:00.000Z'
---

# Incremental Code Smell Analysis & Auto-Refactoring Pipeline

Build a system that auto-detects, analyzes, and refactors Python code smells when files are changed — both locally (via pre-commit hooks) and in CI/CD (GitHub Actions).

## Architecture Overview

```
File Change (git add / push / PR)
        │
        ▼
  Detect Changed .py Files  ◄── git diff --name-only --diff-filter=ACMR
        │
        ▼
  Extract Functions via AST  ◄── scan_files() (targeted, not whole repo)
        │
        ▼
  Detect Code Smells (ML + rule-based)  ◄── predict_smell_batch() + detect_code_smells()
        │
        ├── No smells found → log "✓ OK" per function, done
        │
        └── Bad smells found → for each:
                ├── Recommend best LLM → refactor via OpenRouter API
                ├── Write refactored code in-place
                └── Log "⚠ BAD" + detected smells per function
```

## Key Components

### 1. CLI `--files` Flag (file-targeted mode)
Instead of scanning a whole repo, accept specific file paths so analysis is incremental:
```python
# In repo_analyzer.py
def scan_files(file_paths: List[Path], root: Optional[Path] = None) -> List[Dict]:
    """Scan specific Python files (not whole repo) and extract functions."""
    for py_file in file_paths:
        functions = extract_functions_from_file(py_file)
        # Add relative paths for display
        ...

# In cli.py — argparse
parser.add_argument("--files", "-F", nargs="+",
                    help="Specific Python file(s) to analyze (instead of whole repo)")
```

### 2. Enhanced Logging with Per-Function Smell Details
Capture and display which specific smells were detected per function:
```python
# During smell detection, also run the rule-based detector
func["smell_details"] = detect_code_smells(func["source_code"])

# Print helper
def print_smell_details(func, verbose=False):
    """Print detailed code smell breakdown for a function."""
    detected_smells = {s: v for s, v in details.items() if v}
    if detected_smells:
        print(f"  ⚠ Smells detected:")
        for smell_name, _ in detected_smells.items():
            print(f"    • {smell_name.replace('_', ' ').title()}")
    if verbose:
        for smell_name in SMELL_NAMES:
            icon = "✗" if details.get(smell_name) else "✓"
            print(f"    {icon} {smell_name.replace('_', ' ').title()}")
```

### 3. Git Pre-Commit Hook (local auto-refactoring)
Create a `hooks/pre-commit` script that:
- Finds staged `.py` files via `git diff --cached --name-only --diff-filter=ACM`
- Runs the CLI with `--files` + `--in-place` + `--verbose`
- Re-stages refactored files with `git add`
- Controlled via env vars: `CODEREFACTOR_DRY_RUN=1`, `CODEREFACTOR_SKIP=1`

Install with `src/setup_hooks.py` (copies hook to `.git/hooks/pre-commit`).

### 4. CI/CD Incremental Detection (GitHub Actions)
In the workflow, detect only changed files to avoid full-repo scans on push/PR:
```yaml
- name: Detect changed Python files
  run: |
    # Push mode
    CHANGED=$(git diff --name-only --diff-filter=ACMR HEAD~1 HEAD | grep '\.py$' || true)
    # PR mode (fallback)
    if [ -z "$CHANGED" ]; then
      CHANGED=$(git diff --name-only --diff-filter=ACMR $BASE_SHA HEAD | grep '\.py$' || true)
    fi
    if [ -n "$CHANGED" ]; then
      echo "$CHANGED" > /tmp/changed_py_files.txt
    fi

# Then use --files from the file
FILES=$(paste -sd ' ' /tmp/changed_py_files.txt)
python cli.py --files $FILES --output . --dry-run --json --verbose
```

### 5. Detailed CI Logs & PR Summaries
Move complex Python output logic into separate helper scripts to avoid YAML indentation bugs:
- `src/print_smell_report.py` — prints per-function smell breakdown
- `src/print_refactor_results.py` — prints refactoring results with ✅/❌ per function
- `src/generate_pr_summary.py` — generates structured markdown for PR body

## Common Pitfalls

### YAML Indentation with Inline Python
**Problem:** Multi-line Python inside `run: |` in GitHub Actions YAML can start at column 0, breaking the YAML parser.
```yaml
# ❌ BROKEN — Python code at column 0
run: |
  python -c "
import json    # <-- column 0 inside run: | block — YAML error!
"
```

**Fix:** Use a separate `.py` script file and call it from the workflow, or use a heredoc:
```bash
# ✅ WORKS — call a script file
python src/print_smell_report.py "$REPORT"
```

### Job Output Not Set (Causes Downstream Job to Skip)
**Problem:** If a step with an `if:` condition is skipped, its outputs are never set. The downstream `needs.job.outputs.output_name == 'true'` condition evaluates to `false` (empty string ≠ `'true'`), causing dependent jobs to silently skip.

```yaml
# ❌ BUG: if smell_check is skipped, has_bad_smells is never set
steps:
  - name: Detect changed files
    id: changed_files
    run: echo "changed=false" >> $GITHUB_OUTPUT

  - name: Run analysis
    id: smell_check
    if: steps.changed_files.outputs.changed == 'true'  # SKIPPED!
    run: echo "has_bad_smells=true" >> $GITHUB_OUTPUT  # Never runs

  - name: Refactor
    if: needs.analyze.outputs.has_bad_smells == 'true'  # FALSE — silently skipped!
```

**Fix:** Remove the `if:` condition. Always run the analysis step — handle "no changed files" as a full repo scan inside the script:
```yaml
- name: Run code smell analysis
  id: smell_check
  # ❌ NO if: condition — always runs
  run: |
    if [ -f /tmp/changed_py_files.txt ]; then
      # incremental mode
      python cli.py --files $(cat /tmp/changed_py_files.txt) ...
    else
      # fallback: full repo scan
      python cli.py --repo "$SRC_DIR" ...
    fi
    echo "has_bad_smells=..." >> $GITHUB_OUTPUT  # ALWAYS set
```

### git fetch --depth=1 Breaks HEAD~1
**Problem:** Running `git fetch --depth=1 origin main` in a shallow checkout clobbers the local history, making `HEAD~1` unresolvable and `git diff` comparisons fail.

```yaml
# ❌ BUG: fetch with depth=1 removes parent commit references
- run: |
    git fetch --depth=1 origin main
    git diff HEAD~1 HEAD  # FAILS — HEAD~1 doesn't exist!
```

**Fix:** Don't fetch — rely on the checkout's `fetch-depth: N` (e.g., 50). Use direct base SHA for PRs:
```yaml
- run: |
    if [ -n "${{ github.event.pull_request.base.sha }}" ]; then
      CHANGED=$(git diff --name-only "$BASE" HEAD | grep '\.py$' || true)
    fi
    if [ -z "$CHANGED" ]; then
      CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null | grep '\.py$' || true)
    fi
```

### Node 20 Deprecation on GitHub Runners
**Problem:** GitHub Actions runners now default to Node 24. Actions pinned to `@v6` (which uses Node 20) emit a deprecation warning and may fail if Node 20 is removed.

**Fix:** Update to the latest major version:
```yaml
# ❌ Deprecated (Node 20)
- uses: peter-evans/create-pull-request@v6

# ✅ Current (Node 24)
- uses: peter-evans/create-pull-request@v8
```

### GITHUB_OUTPUT Multi-Line (Heredoc) Parsing
**Problem:** Using `<<EOF` heredoc syntax for multi-line `$GITHUB_OUTPUT` values can fail if:
- The delimiter (`EOF`) appears inside the value
- The runner doesn't support older heredoc parsing

```python
# ❌ Fragile heredoc approach
f.write(f"details_md<<EOF\n{long_markdown}\nEOF\n")
```

**Fix:** Use simple single-line values or separate `echo` calls. Pipe Python stdout directly to `$GITHUB_OUTPUT`:
```python
# ✅ Simple stdout, no heredoc needed
print(f"total={total}")
print(f"bad={bad}")
print(f"details={' | '.join(details_lines)}")
```

```yaml
# ✅ Pipe to GITHUB_OUTPUT line by line
- run: |
    python script.py | while read line; do
      echo "$line" >> $GITHUB_OUTPUT
    done
```

### CLI Exit Code Fails the Workflow Step
**Problem:** If `python cli.py` exits with non-zero (e.g., model training issue, API error), the workflow step fails immediately. Subsequent output-setting commands (like `echo "has_bad_smells=..." >> $GITHUB_OUTPUT`) never run.

**Fix:** Wrap the CLI call with `set +e` to continue after failure, and always write the output:
```yaml
- run: |
    set +e
    python cli.py --files ...  # May fail
    CLI_EXIT=$?
    set -e

    # Always write output (even if CLI failed)
    echo "has_bad_smells=..." >> $GITHUB_OUTPUT

    # Log the exit code for debugging
    if [ "$CLI_EXIT" -ne 0 ]; then
      echo "⚠ CLI exited with code $CLI_EXIT"
    fi
```

### Model Pickle Version Mismatch
When retraining models across Python/numpy versions, retrain with `python src/train_smell_model.py`.

### Pre-Commit Hook Must Re-Stage
After in-place refactoring, the hook must `git add` the refactored files so the commit includes the changes.

## Verification Test
```bash
# Test with dry-run (no API calls)
python cli.py --files app.py --dry-run --json --verbose

# Expected output:
#   ✓ OK   good_function              (file.py:10)
#   ⚠ BAD  bad_function               (file.py:50)  [prob: 100.0%]
#        ⚠ Smells detected:
#          • Long Method
#          • Deep Nesting