#!/usr/bin/env python3
"""Print a detailed refactoring results summary from a refactor-report.json file."""
import json
import sys

report_path = sys.argv[1] if len(sys.argv) > 1 else "refactor-report.json"

try:
    d = json.load(open(report_path))
    refactored = [r for r in d if r.get("refactored")]
    failed = [r for r in d if r.get("refactor_error")]
    skipped = [r for r in d if not r.get("is_bad_smell")]
    bad = [r for r in d if r.get("is_bad_smell")]

    print(f"Total: {len(d)} | Refactored: {len(refactored)} | Failed: {len(failed)} | Skipped (good): {len(skipped)}")
    print()

    for r in refactored:
        details = r.get("smell_details", {})
        detected = [k.replace("_", " ").title() for k, v in details.items() if v]
        smell_str = f' (smells: {", ".join(detected)})' if detected else ""
        print(f'  ✅ {r["full_name"]:<35} — refactored by {r.get("best_llm", "?")}{smell_str}')

    for r in failed:
        print(f'  ❌ {r["full_name"]:<35} — {r["refactor_error"][:80]}')

    for r in bad:
        if not r.get("refactored") and not r.get("refactor_error"):
            print(f'  ⚠️ {r["full_name"]:<35} — not refactored')

    print()
except Exception as e:
    print(f"Error reading report: {e}", file=sys.stderr)
    sys.exit(1)