#!/usr/bin/env python3
"""Generate PR summary outputs from a refactor-report.json file."""
import json
import sys

report_path = sys.argv[1] if len(sys.argv) > 1 else "refactor-report.json"

try:
    d = json.load(open(report_path))
    total = len(d)
    bad = sum(1 for r in d if r.get("is_bad_smell"))
    refactored = sum(1 for r in d if r.get("refactored"))
    failed = sum(1 for r in d if r.get("refactor_error"))
    llms = set(r.get("best_llm", "") for r in d if r.get("best_llm"))
    llms.discard("")
    llm_str = ", ".join(sorted(llms)) if llms else "N/A"

    # Generate short per-function summary (single line per function)
    details_lines = []
    for r in d:
        if r.get("refactored"):
            details_lines.append(f"- ✅ {r['full_name']} ({r['file']}:{r['start_line']})")
        elif r.get("refactor_error"):
            details_lines.append(f"- ❌ {r['full_name']} ({r['file']}:{r['start_line']})")

    # Write outputs to stdout for GitHub Actions
    print(f"total={total}")
    print(f"bad={bad}")
    print(f"refactored={refactored}")
    print(f"llms={llm_str}")
    print(f"failed={failed}")
    # Simple single-line format - no heredoc needed
    details_line = " | ".join(details_lines) if details_lines else "No functions processed."
    print(f"details={details_line}")

except Exception as e:
    print(f"::error::PR summary generation failed: {e}")
    print(f"total=0")
    print(f"bad=0")
    print(f"refactored=0")
    print(f"llms=N/A")
    print(f"failed=0")
    print(f"details=Error generating report")
    sys.exit(0)  # Don't fail the step