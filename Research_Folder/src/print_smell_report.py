#!/usr/bin/env python3
"""Print a detailed smell breakdown from a refactor-report.json file."""
import json
import sys

report_path = sys.argv[1] if len(sys.argv) > 1 else "refactor-report.json"

try:
    d = json.load(open(report_path))
    for r in d:
        if r.get("is_bad_smell"):
            details = r.get("smell_details", {})
            detected = [k.replace("_", " ").title() for k, v in details.items() if v]
            print(f"  ⚠ {r['full_name']:<35} ({r['file']}:{r['start_line']})")
            print(f"     Probability: {r['smell_probability']:.1%}")
            if detected:
                print(f"     Smells: {', '.join(detected)}")
            print()
    print(f"Total functions: {len(d)} | Bad-smell: {sum(1 for r in d if r.get('is_bad_smell'))}")
except Exception as e:
    print(f"Error reading report: {e}", file=sys.stderr)
    sys.exit(1)