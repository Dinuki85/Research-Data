#!/usr/bin/env python3
"""
CodeRefactor AI — CLI Entry Point
==================================
Analyzes a Python repository for code smells, recommends the best LLM
for refactoring each bad-smell function, and executes the refactoring.

Pipeline stages:
  1. Scan repository → extract all Python functions
  2. Code smell detection → identify bad-smell functions
  3. LLM recommendation → predict best model for each bad function
  4. Refactoring → call OpenRouter API to refactor
  5. Apply changes → write refactored code back

Usage:
  python cli.py --repo /path/to/repo
  python cli.py --repo https://github.com/user/repo.git
  python cli.py --repo /path/to/repo --dry-run
  python cli.py --repo /path/to/repo --api-key sk-or-v1-...
  python cli.py --repo /path/to/repo --function-only "my_function"
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Ensure the src directory is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src import __version__
from src.code_smell_detector import predict_smell, predict_smell_batch
from src.llm_recommender import get_ranking, get_best_llm, MODEL_ORDER, MODEL_DISPLAY, detect_code_smells, SMELL_NAMES
from src.refactoring_engine import refactor_function, verify_refactored_code
from src.repo_analyzer import scan_functions, scan_files, clone_repository

# ── Default config ───────────────────────────────────────────────────────────
DEFAULT_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def green(text):
    return f"\033[32m{text}\033[0m" if sys.stdout.isatty() else text


def yellow(text):
    return f"\033[33m{text}\033[0m" if sys.stdout.isatty() else text


def red(text):
    return f"\033[31m{text}\033[0m" if sys.stdout.isatty() else text


def cyan(text):
    return f"\033[36m{text}\033[0m" if sys.stdout.isatty() else text


def bold(text):
    return f"\033[1m{text}\033[0m" if sys.stdout.isatty() else text


def dim(text):
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


def _save_json_report(all_functions, repo_path, output_base):
    """Save analysis/refactoring results as JSON for CI/CD integration."""
    import json
    json_results = []
    for f in all_functions:
        entry = {
            "name": f["name"],
            "full_name": f["full_name"],
            "file": f.get("rel_path", f.get("file_path", "")),
            "start_line": f.get("start_line", 0),
            "num_lines": f.get("num_lines", 0),
            "is_bad_smell": f.get("is_bad_smell", False),
            "smell_probability": f.get("smell_probability", 0),
            "smell_details": f.get("smell_details", {}),
            "best_llm": f.get("best_llm_name"),
            "refactored": f.get("refactored", False),
            "refactor_error": f.get("refactor_error"),
        }
        if "llm_ranking" in f:
            entry["llm_ranking"] = [
                {"model": m["display_name"], "quality": m["quality_score"],
                 "time_ms": m["pred_time_ms"], "cost": m["pred_cost"],
                 "composite": m["composite"]}
                for m in f["llm_ranking"]
            ]
        json_results.append(entry)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    json_path = output_base / "refactor-report.json"
    json_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    return json_path


def print_smell_details(func, verbose=False):
    """Print detailed code smell breakdown for a function."""
    details = func.get("smell_details", {})
    if not details:
        return

    detected_smells = {s: v for s, v in details.items() if v}
    if detected_smells:
        print(f"         {yellow('⚠ Smells detected:')}")
        for smell_name, _ in detected_smells.items():
            label = smell_name.replace("_", " ").title()
            print(f"           • {label}")
    else:
        print(f"         {green('✓ No specific smells detected (ML model flagged it)')}")

    if verbose:
        print(f"         {dim('Full smell analysis:')}")
        for smell_name in SMELL_NAMES:
            label = smell_name.replace("_", " ").title()
            val = details.get(smell_name, 0)
            icon = red("✗") if val else green("✓")
            print(f"           {icon} {label}")


def print_file_header(file_path, index, total):
    """Print a header for a file being processed."""
    print()
    print(bold(cyan(f"  📄 [{index}/{total}] File: {file_path}")))
    print(f"  {'─' * 66}")


def print_banner():
    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║          CodeRefactor AI — Automated Python Refactoring         ║")))
    print(bold(cyan("║          v{:<73}║".format(__version__))))
    print(bold(cyan("╚══════════════════════════════════════════════════════════════════╝")))
    print()


def print_summary(results: list, output_dir: Path):
    """Print a nice summary of the pipeline results."""
    total = len(results)
    bad = sum(1 for r in results if r["is_bad_smell"])
    refactored = sum(1 for r in results if r.get("refactored"))
    failed = sum(1 for r in results if r.get("refactor_error"))
    skipped = total - bad

    print()
    print(bold(cyan("═" * 70)))
    print(bold("  Pipeline Summary"))
    print(bold(cyan("═" * 70)))
    print(f"  Total functions found:    {total}")
    print(f"  Good-smell (skipped):     {green(str(skipped))}")
    print(f"  Bad-smell (to refactor):  {yellow(str(bad))}")
    print(f"  Refactored successfully:  {green(str(refactored))}")
    print(f"  Refactoring failed:       {red(str(failed)) if failed else green('0')}")
    print()

    if refactored > 0:
        print(f"  Refactored code written to: {bold(str(output_dir))}")
        print()

    # Print per-function results
    print(bold("  Detail:"))
    print(f"  {'Function':<45} {'Smell':>8} {'Best LLM':<20} {'Status':<12}")
    print("  " + "─" * 85)

    for r in results:
        name = r.get("full_name", r.get("name", "?"))[:44]
        smell = red("BAD ") if r["is_bad_smell"] else green("GOOD")
        llm = r.get("best_llm_name", "-")[:19] if r["is_bad_smell"] else dim("-")
        if r.get("refactored"):
            status = green("REFACTORED")
        elif r.get("refactor_error"):
            status = red("FAILED")
        elif r["is_bad_smell"] and not r.get("refactored"):
            status = yellow("PENDING")
        else:
            status = dim("SKIPPED")

        print(f"  {name:<45} {smell:>8} {llm:<20} {status:<12}")

    print()


def run_pipeline(args):
    """Run the full refactoring pipeline."""
    print_banner()

    # ── 1. Resolve target (repo or specific files) ───────────────────────────
    temp_dir = None
    repo_path = None
    all_functions = []

    if args.files:
        # ── File-specific mode ──────────────────────────────────────────────
        print(f"📄  Analyzing specific files: {cyan(str(args.files))}")
        file_paths = [Path(f).resolve() for f in args.files]

        # Determine root for relative paths (use git root if available)
        try:
            import subprocess
            first_file_dir = str(Path(args.files[0]).resolve().parent) if args.files else "."
            git_root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
                cwd=first_file_dir,
            )
            root = Path(git_root.stdout.strip()) if git_root.returncode == 0 else None
        except Exception:
            root = None

        all_functions = scan_files(file_paths, root=root)
        if not all_functions:
            print(f"  {yellow('No Python functions found in specified files.')}")
            return

        print(f"  Found {len(all_functions)} functions in {len(file_paths)} file(s)")

    else:
        # ── Repository mode ────────────────────────────────────────────────
        repo_source = args.repo
        is_remote = repo_source.startswith("http://") or repo_source.startswith("https://") or repo_source.startswith("git@")

        if is_remote:
            print(f"📦  Cloning remote repository: {cyan(repo_source)}")
            temp_dir = Path(tempfile.mkdtemp(prefix="coderefactor-"))
            try:
                repo_path = clone_repository(repo_source, temp_dir)
            except Exception as e:
                print(f"  {red('❌')} Failed to clone: {e}")
                sys.exit(1)
        else:
            repo_path = Path(repo_source).resolve()
            if not repo_path.exists():
                print(f"  {red('❌')} Repository path does not exist: {repo_path}")
                sys.exit(1)
            print(f"📂  Analyzing local repository: {cyan(str(repo_path))}")

        # ── 2. Scan repository for functions ───────────────────────────────────
        print(f"\n{'─' * 70}")
        print(bold("  Stage 1: Scanning Repository for Python Functions"))
        print(f"{'─' * 70}")

        try:
            all_functions = scan_functions(repo_path)
        except Exception as e:
            print(f"  {red('❌')} Error scanning repository: {e}")
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)

        if not all_functions:
            print(f"  {yellow('No Python functions found in repository.')}")
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Filter by specific function if requested
        if args.function_only:
            all_functions = [f for f in all_functions if args.function_only in f["full_name"] or args.function_only in f["name"]]
            if not all_functions:
                msg = f'No function matching "{args.function_only}" found.'
                print(f"  {yellow(msg)}")
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                return

        print(f"  Found {len(all_functions)} functions in {len(set(f['file_path'] for f in all_functions))} files")

    # ── 3. Code smell detection ─────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(bold("  Stage 2: Code Smell Detection"))
    print(f"{'─' * 70}")

    codes = [f["source_code"] for f in all_functions]
    smell_results = predict_smell_batch(codes)

    # Enrich functions with smell results and detailed smell breakdown
    for i, func in enumerate(all_functions):
        func["is_bad_smell"] = smell_results[i]["is_bad_smell"]
        func["smell_probability"] = smell_results[i]["probability"]
        func["smell_num_lines"] = smell_results[i]["num_lines"]
        # Add detailed smell analysis
        func["smell_details"] = detect_code_smells(func["source_code"])

    bad_functions = [f for f in all_functions if f["is_bad_smell"]]
    good_functions = [f for f in all_functions if not f["is_bad_smell"]]

    # Print per-file summary
    files_analyzed = set(f.get("rel_path", f.get("file_path", "")) for f in all_functions)
    print(f"  {green(f'✓ Good-smell functions: {len(good_functions)}')}")
    print(f"  {yellow(f'⚠ Bad-smell functions (need refactoring): {len(bad_functions)}')}")
    print(f"  Files analyzed: {len(files_analyzed)}")

    # Print detailed results per function
    if args.verbose:
        print(f"\n  {bold('Per-function analysis:')}")
        for f in all_functions:
            file_info = f.get("rel_path", f.get("file_path", "?"))
            label = f['full_name']
            if f["is_bad_smell"]:
                prob = f["smell_probability"]
                print(f"    {red('⚠ BAD')}  {label:<40} ({file_info}:{f['start_line']})  "
                      f"[prob: {prob:.1%}]")
                print_smell_details(f, verbose=True)
            else:
                print(f"    {green('✓ OK')}   {label:<40} ({file_info}:{f['start_line']})")

    elif bad_functions:
        print(f"\n  {bold('Bad-smell functions:')}")
        for f in bad_functions:
            prob = f["smell_probability"]
            file_info = f.get("rel_path", f.get("file_path", "?"))
            print(f"    {yellow('⚠')} {f['full_name']:<40} "
                  f"({file_info}:{f['start_line']})  "
                  f"[prob: {prob:.1%}, lines: {f['num_lines']}]")
            print_smell_details(f, verbose=False)

    # Define output base path (used by dry-run and full pipeline)
    if args.output:
        output_base = Path(args.output)
    elif repo_path:
        output_base = repo_path.parent / f"{repo_path.name}-refactored"
    else:
        # --files mode: use current directory
        output_base = Path.cwd() / "refactored-output"

    if args.dry_run:
        print(f"\n  {yellow('DRY RUN: stopping before LLM recommendation and refactoring.')}")

        # Still save the analysis report
        if args.json:
            report_path = _save_json_report(all_functions, repo_path or Path.cwd(), output_base)
            if report_path:
                print(f"  Analysis report saved to: {bold(str(report_path))}")

        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # ── 4. LLM Recommendation ────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(bold("  Stage 3: LLM Recommendation for Bad-Smell Functions"))
    print(f"{'─' * 70}")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or DEFAULT_API_KEY

    for i, func in enumerate(bad_functions):
        print(f"\n  [{i+1}/{len(bad_functions)}] Analyzing: {cyan(func['full_name'])}")
        print(f"         File: {dim(func['rel_path'])}:{func['start_line']}")
        # Show detected smells
        print_smell_details(func, verbose=args.verbose)

        ranking = get_ranking(func["source_code"], func["name"])
        best = ranking[0] if ranking else None

        if best:
            func["best_llm_key"] = best["model_key"]
            func["best_llm_name"] = best["display_name"]
            func["llm_ranking"] = ranking
            func["llm_quality_score"] = best["quality_score"]
            func["llm_composite"] = best["composite"]

            print(f"         Best LLM: {green(best['display_name'])}  "
                  f"(quality: {best['quality_score']:.1f}/10, "
                  f"composite: {best['composite']:.1f}/100)")

            # Show top 3
            for j, m in enumerate(ranking[:3]):
                print(f"           {j+1}. {m['display_name']:<20}  "
                      f"Q:{m['quality_score']:.1f}  "
                      f"T:{m['pred_time_ms']:,}ms  "
                      f"${m['pred_cost']:.6f}  "
                      f"Score:{m['composite']:.1f}")
        else:
            func["best_llm_key"] = "gpt_oss"
            func["best_llm_name"] = "GPT-OSS 120B"
            print(f"         {yellow('Using default: GPT-OSS 120B')}")

    # ── 5. Refactoring ───────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(bold("  Stage 4: Refactoring Bad-Smell Functions via OpenRouter"))
    print(f"{'─' * 70}")

    for i, func in enumerate(bad_functions):
        print(f"\n  [{i+1}/{len(bad_functions)}] Refactoring: {cyan(func['full_name'])}")
        print(f"         Using: {green(func.get('best_llm_name', 'GPT-OSS 120B'))}")

        llm_key = func.get("best_llm_key", "gpt_oss")

        result = refactor_function(
            code=func["source_code"],
            model_key=llm_key,
            api_key=api_key,
            max_retries=args.retries,
            timeout=args.timeout,
        )

        if result["success"] and verify_refactored_code(func["source_code"], result["refactored_code"]):
            func["refactored_code"] = result["refactored_code"]
            func["refactored"] = True
            func["refactor_usage"] = result["usage"]
            print(f"         {green('✓ Refactored successfully')}")

            if result["usage"]:
                print(f"           Tokens: {result['usage'].get('total_tokens', '?')}")

            # Write the refactored file
            original_file = Path(func["file_path"])
            if args.in_place:
                # Write directly back to the original source file
                out_path = original_file
            else:
                rel_path = Path(func["rel_path"])
                out_path = output_base / rel_path

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Read existing file and replace the function
            try:
                if original_file.exists():
                    content = original_file.read_text(encoding="utf-8", errors="ignore")
                    # Replace the function source code
                    new_content = content.replace(func["source_code"], result["refactored_code"])
                    out_path.write_text(new_content, encoding="utf-8")
                else:
                    out_path.write_text(result["refactored_code"] + "\n", encoding="utf-8")
            except Exception as e:
                out_path.write_text(result["refactored_code"] + "\n", encoding="utf-8")

            # Small delay to avoid API rate limits
            if i < len(bad_functions) - 1:
                time.sleep(0.5)

        else:
            func["refactored"] = False
            func["refactor_error"] = result.get("error", "Unknown error")
            print(f"         {red('✗ Refactoring failed:')} {result.get('error', 'Verification failed')}")
            print(f"         {yellow('  Keeping original code.')}")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print_summary(bad_functions + good_functions, output_base)

    # Save results as JSON for CI/CD integration
    if args.json:
        json_path = _save_json_report(all_functions, repo_path or Path.cwd(), output_base)
        print(f"  JSON report saved to: {bold(str(json_path))}")

    # Clean up temp directory
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{green('✅ Pipeline completed successfully.')}")
    if args.in_place:
        print(f"   Refactored code written {bold('in-place')} to original source files.")
    elif args.files:
        print(f"   Refactored code written to: {bold(str(output_base))}")
    else:
        print(f"   Refactored code: {bold(str(output_base))}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CodeRefactor AI — Automated Python code refactoring pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --repo /path/to/repo
  python cli.py --repo https://github.com/user/repo.git
  python cli.py --repo /path/to/repo --dry-run
  python cli.py --repo /path/to/repo --function-only "my_function"
  python cli.py --repo /path/to/repo --output /path/to/output
        """,
    )

    parser.add_argument("--repo", "-r",
                        help="Path to local repository or Git URL")
    parser.add_argument("--files", "-F", nargs="+",
                        help="Specific Python file(s) to analyze (instead of whole repo)")
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Show detailed per-function smell analysis")
    parser.add_argument("--api-key", "-k",
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--output", "-o",
                        help="Output directory for refactored code")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Only analyze, do not refactor")
    parser.add_argument("--function-only", "-f",
                        help="Only process functions matching this name")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max retries per API call (default: 3)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="API timeout in seconds (default: 120)")
    parser.add_argument("--json", action="store_true",
                        help="Save results as JSON report")
    parser.add_argument("--in-place", action="store_true",
                        help="Write refactored code directly to original source files "
                             "(instead of a separate output directory)")
    parser.add_argument("--version", "-v", action="version",
                        version=f"CodeRefactor AI v{__version__}")

    args = parser.parse_args()

    if not args.repo and not args.files:
        parser.error("Either --repo or --files must be provided.")
    if args.repo and args.files:
        parser.error("Cannot use both --repo and --files at the same time.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
