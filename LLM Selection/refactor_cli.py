#!/usr/bin/env python3
"""
refactor_cli.py
===============
CLI tool that takes a Python function as input and ranks the 5 LLMs
(claude_opus, claude_sonnet_4_6, gemini_3_1_pro, gemini_flash, gpt_oss)
by predicted refactoring quality score (0-10), generation time, and total tokens.

Quality is measured on a continuous 0-10 scale across 5 pillars:
  - Code Presence    (0-1)
  - Documentation    (0-2)   docstring + type hints
  - Modern Python    (0-1)   future annotations
  - Smell Removal    (0-5)   9 code smells detected; each removed = +1
  - Structural       (0-1)   line count + complexity reduction

Usage:
  python refactor_cli.py                  # interactive mode
  python refactor_cli.py --file foo.py    # read code from file
  python refactor_cli.py --help
"""

import argparse
import re
import sys
import os
import warnings
warnings.filterwarnings("ignore")   # suppress sklearn version / feature-name warnings
import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

MODEL_ORDER = ["claude_opus", "claude_sonnet_4_6", "gemini_3_1_pro", "gemini_flash", "gpt_oss"]
MODEL_DISPLAY = {
    "claude_opus":        "Claude Opus",
    "claude_sonnet_4_6":  "Claude Sonnet 4.6",
    "gemini_3_1_pro":     "Gemini 3.1 Pro",
    "gemini_flash":       "Gemini Flash",
    "gpt_oss":            "GPT-OSS 120B",
}

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours (gracefully disabled on non-TTY)
# ─────────────────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()

def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def bold(t):    return c(t, "1")
def green(t):   return c(t, "32")
def yellow(t):  return c(t, "33")
def red(t):     return c(t, "31")
def cyan(t):    return c(t, "36")
def dim(t):     return c(t, "2")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers  (must match both notebooks exactly)
# ─────────────────────────────────────────────────────────────────────────────
def count_lines(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return sum(1 for l in s.replace("\\n", "\n").split("\n") if l.strip())


def count_keywords(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return sum(s.count(k) for k in ("if ", "for ", "while ", "try:", "except", "with "))


def has_docstring(s: str) -> int:
    return int(isinstance(s, str) and ('"""' in s or "'''" in s))


def has_type_hints(s: str) -> int:
    return int(
        isinstance(s, str)
        and bool(re.search(
            r"->\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)",
            s))
    )


def has_future_import(s: str) -> int:
    return int(isinstance(s, str) and "from __future__ import annotations" in s)


def char_token_est(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return max(0, len(s.replace("\\n", "\n")) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Code Smell Detectors  (9 smells — must match refactoring_quality_ml.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
SMELL_NAMES = [
    "long_method",      # function body > 20 non-empty lines
    "long_param_list",  # > 4 parameters (excl. self/cls)
    "deep_nesting",     # max indentation >= 4 levels (16 spaces)
    "magic_numbers",    # bare numeric literals beyond common ones
    "no_docstring",     # missing docstring
    "no_type_hints",    # no return / param type annotations
    "long_lines",       # any line > 79 chars
    "commented_code",   # commented-out executable code
    "poor_naming",      # single-char variable assignments
]


def detect_code_smells(code_str: str) -> dict:
    """Return {smell_name: 0/1} for all 9 smells."""
    empty = {s: 0 for s in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty

    code     = code_str.replace("\\n", "\n")
    lines    = code.split("\n")
    nonempty = [l for l in lines if l.strip()]
    smells   = {}

    # 1. Long Method
    smells["long_method"] = int(len(nonempty) > 20)

    # 2. Long Parameter List
    sig = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if sig:
        params = [p.strip() for p in sig.group(1).split(",")
                  if p.strip() and p.strip() not in ("self", "cls")]
        smells["long_param_list"] = int(len(params) > 4)
    else:
        smells["long_param_list"] = 0

    # 3. Deep Nesting (>= 4 indent levels x 4 spaces = 16 spaces)
    max_indent = max((len(l) - len(l.lstrip())) for l in nonempty) if nonempty else 0
    smells["deep_nesting"] = int(max_indent >= 16)

    # 4. Magic Numbers (ignore 0,1,2,3,10,100,-1)
    _allowed_nums = {"0", "1", "2", "3", "10", "100", "-1", "0.0", "1.0"}
    nums  = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in nums if n not in _allowed_nums]
    smells["magic_numbers"] = int(len(magic) > 2)

    # 5. No Docstring
    smells["no_docstring"] = int('"""' not in code and "'''" not in code)

    # 6. No Type Hints
    smells["no_type_hints"] = int(not bool(re.search(
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)",
        code)))

    # 7. Long Lines
    smells["long_lines"] = int(any(len(l) > 79 for l in lines))

    # 8. Commented-out Code
    commented = [l for l in lines if re.match(
        r"\s*#\s*(if |for |while |return |self\.|import |def |class |print\()", l)]
    smells["commented_code"] = int(len(commented) >= 1)

    # 9. Poor Naming
    _allowed_vars = {"i", "j", "k", "n", "x", "y", "z", "f", "e", "v", "_", "s", "c", "p", "q"}
    poor = [v for v in re.findall(r"\b([a-zA-Z])\s*(?:\+|-|\*|\/)?=(?!=)", code)
            if v.lower() not in _allowed_vars]
    smells["poor_naming"] = int(len(poor) > 1)

    return smells


def count_smells(code_str: str) -> int:
    return sum(detect_code_smells(code_str).values())


def quality_grade(score: float) -> str:
    """Map a 0-10 quality score to a human-readable grade."""
    if score >= 9.0:  return "Excellent ★★★"
    if score >= 7.0:  return "Good      ★★☆"
    if score >= 5.0:  return "Moderate  ★☆☆"
    if score >= 3.0:  return "Weak      ✦"
    return                    "Minimal   ·"


def quality_color(score: float):
    if score >= 7.0:  return green
    if score >= 5.0:  return yellow
    return red


def smell_summary(orig_code: str) -> tuple:
    """Return (total_smell_count, list_of_detected_smell_names)."""
    smells   = detect_code_smells(orig_code)
    detected = [s for s, v in smells.items() if v == 1]
    return len(detected), detected


# ─────────────────────────────────────────────────────────────────────────────
# Load all saved models
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    def p(fname):
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(red(f"  [ERROR] Missing model file: {path}"))
            print(red("  Run both notebooks and execute their 'Save Models' cells first."))
            sys.exit(1)
        return joblib.load(path)

    print(dim("Loading models..."), end=" ", flush=True)

    # Quality regressors  (regression task: predict continuous 0-10 score)
    q_regressors = {}
    for name in ["ridge_regression", "decision_tree", "random_forest",
                 "gradient_boosting", "xgboost", "svr"]:
        fpath = os.path.join(MODELS_DIR, f"quality_{name}.pkl")
        if os.path.exists(fpath):
            q_regressors[name] = joblib.load(fpath)

    if not q_regressors:
        print(red("  [ERROR] No quality regressor files found in models/."))
        sys.exit(1)

    q_scaler = p("quality_scaler.pkl")
    q_le     = p("quality_label_encoder.pkl")
    q_stats  = p("quality_stats.pkl")
    q_meta   = p("quality_meta.pkl")

    # Metrics regressors
    m_time   = {}
    m_tokens = {}
    for name in ["ridge", "random_forest", "gradient_boosting", "xgboost"]:
        for d, prefix in [(m_time, "metrics_time"), (m_tokens, "metrics_tokens")]:
            fpath = os.path.join(MODELS_DIR, f"{prefix}_{name}.pkl")
            if os.path.exists(fpath):
                d[name] = joblib.load(fpath)

    m_scaler = p("metrics_scaler.pkl")
    m_le     = p("metrics_label_encoder.pkl")
    m_stats  = p("metrics_stats.pkl")
    m_meta   = p("metrics_meta.pkl")

    # Load prices (per-token) if available
    prices_path = os.path.join(SCRIPT_DIR, "price.csv")
    prices = {}
    if os.path.exists(prices_path):
        import csv
        import re as _re
        with open(prices_path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get('Model Family') or row.get('Model') or row.get('model')
                inp = row.get('Input') or row.get('input')
                if not key or not inp:
                    continue
                s = str(inp).strip()
                # Find numbers (handles ranges like "0.0000001 - 0.0000005")
                nums = _re.findall(r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", s)
                try:
                    vals = [float(x) for x in nums]
                    if len(vals) == 0:
                        continue
                    price_val = sum(vals) / len(vals)
                    # Normalize common commas/format issues
                    prices[key.strip()] = price_val
                except Exception:
                    continue
    else:
        prices = {}

    print(green("OK"))
    return (q_regressors, q_scaler, q_le, q_stats, q_meta,
            m_time, m_tokens, m_scaler, m_le, m_stats, m_meta, prices)


# ─────────────────────────────────────────────────────────────────────────────
# Build quality feature vector  (must match FEATURE_COLS in notebook exactly)
#
#   STRUCTURAL_FEATURES (19) + SMELL_FEATURES (31) = 50 features total
# ─────────────────────────────────────────────────────────────────────────────
def quality_features(orig_code: str, refact_code: str, func_name: str,
                     model_enc: int, q_stats: dict) -> np.ndarray:
    # ── Structural features ───────────────────────────────────────────────────
    orig_line_count   = float(count_lines(orig_code))
    long_method_flag  = int(orig_line_count > 15)
    refact_line_count = float(count_lines(refact_code))
    line_delta        = orig_line_count - refact_line_count
    line_delta_ratio  = line_delta / (orig_line_count + 1)
    hfi               = has_future_import(refact_code)
    hds               = has_docstring(refact_code)
    hth               = has_type_hints(refact_code)
    complexity_proxy  = count_keywords(refact_code)
    orig_complexity   = count_keywords(orig_code)
    complexity_delta  = orig_complexity - complexity_proxy
    func_name_len     = len(str(func_name))
    is_private        = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder         = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test_func      = int(str(func_name).startswith("test"))
    repo_avg_lines    = float(q_stats.get("global_repo_mean", orig_line_count))
    file_avg_lines    = float(q_stats.get("global_file_mean", orig_line_count))
    global_max        = float(q_stats.get("global_max_lines", max(orig_line_count, 1)))
    norm_line_count   = orig_line_count / (global_max + 1)

    structural = [
        orig_line_count, long_method_flag, refact_line_count, line_delta,
        line_delta_ratio, hfi, hds, hth,
        complexity_proxy, orig_complexity, complexity_delta,
        func_name_len, is_private, is_dunder, is_test_func,
        repo_avg_lines, file_avg_lines, norm_line_count,
        model_enc,
    ]

    # ── Smell features (31) ───────────────────────────────────────────────────
    orig_smells   = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_smell_count      = sum(orig_smells.values())
    refact_smell_count    = sum(refact_smells.values())
    smell_reduction       = max(0, orig_smell_count - refact_smell_count)
    smell_reduction_ratio = smell_reduction / (orig_smell_count + 1)

    smell_agg  = [orig_smell_count, refact_smell_count, smell_reduction, smell_reduction_ratio]
    orig_per   = [orig_smells[s]   for s in SMELL_NAMES]
    refact_per = [refact_smells[s] for s in SMELL_NAMES]
    fixed_per  = [max(0, orig_smells[s] - refact_smells[s]) for s in SMELL_NAMES]

    smell_features = smell_agg + orig_per + refact_per + fixed_per

    return np.array([structural + smell_features], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Build metrics feature vector
# ─────────────────────────────────────────────────────────────────────────────
def metrics_features(orig_code: str, refact_code: str, func_name: str,
                     model_enc: int, m_stats: dict) -> np.ndarray:
    orig_lines        = float(count_lines(orig_code))
    long_method       = int(orig_lines > 15)
    orig_complexity   = count_keywords(orig_code)
    orig_char_tokens  = char_token_est(orig_code)
    refact_lines      = float(count_lines(refact_code))
    refact_complexity = count_keywords(refact_code)
    refact_char_tokens= char_token_est(refact_code)
    hds               = has_docstring(refact_code)
    hth               = has_type_hints(refact_code)
    hfi               = has_future_import(refact_code)
    line_delta        = orig_lines - refact_lines
    line_delta_ratio  = line_delta / (orig_lines + 1)
    complexity_delta  = orig_complexity - refact_complexity
    token_delta       = orig_char_tokens - refact_char_tokens
    est_input_tokens  = orig_char_tokens + 250
    est_output_tokens = refact_char_tokens
    func_name_len     = len(str(func_name))
    is_private        = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder         = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test_func      = int(str(func_name).startswith("test"))

    repo_avg_orig_lines = float(m_stats.get("global_repo_mean", orig_lines))
    file_avg_orig_lines = float(m_stats.get("global_file_mean", orig_lines))

    return np.array([[
        orig_lines, long_method, orig_complexity, orig_char_tokens,
        refact_lines, refact_complexity, refact_char_tokens,
        hds, hth, hfi,
        line_delta, line_delta_ratio, complexity_delta, token_delta,
        est_input_tokens, est_output_tokens,
        func_name_len, is_private, is_dunder, is_test_func,
        repo_avg_orig_lines, file_avg_orig_lines,
        model_enc,
    ]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Predict for all LLMs
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(orig_code: str, func_name: str,
                q_regressors, q_scaler, q_le, q_stats, q_meta,
                m_time, m_tokens, m_scaler, m_le, m_stats, m_meta,
                prices: dict):
    """
    For each of the 5 LLMs produce:
      - quality_score   (continuous 0-10, predicted by best regression model)
      - pred_time_ms
      - pred_total_tokens
      - pred_cost

    Original code is used as proxy refactored code since we don't run the LLM
    here — smell/structural delta features will be zero, so the score captures
    model-identity and the input code's inherent characteristics.
    """
    refact_code_proxy = orig_code

    best_q_name  = q_meta["best_model"]
    best_q_safe  = best_q_name.lower().replace(" ", "_")
    q_scaled_models = q_stats.get("scaled_models", ["Ridge Regression", "SVR"])

    best_time_name   = m_meta.get("best_time_model", "gradient_boosting").lower().replace(" ", "_")
    best_tokens_name = m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(" ", "_")

    results = {}
    for llm in MODEL_ORDER:
        # ── Encode model ──────────────────────────────────────────────────────
        try:
            q_enc = int(q_le.transform([llm])[0])
        except Exception:
            q_enc = MODEL_ORDER.index(llm)
        try:
            m_enc = int(m_le.transform([llm])[0])
        except Exception:
            m_enc = MODEL_ORDER.index(llm)

        # ── Quality score prediction (regression 0-10) ────────────────────────
        qX  = quality_features(orig_code, refact_code_proxy, func_name, q_enc, q_stats)
        reg = q_regressors.get(best_q_safe) or next(iter(q_regressors.values()))
        qX_input = q_scaler.transform(qX) if best_q_name in q_scaled_models else qX

        try:
            quality_score = float(np.clip(reg.predict(qX_input)[0], 0.0, 10.0))
        except Exception:
            quality_score = 5.0

        # ── Metrics prediction ────────────────────────────────────────────────
        mX         = metrics_features(orig_code, refact_code_proxy, func_name, m_enc, m_stats)
        time_reg   = m_time.get(best_time_name) or next(iter(m_time.values()))
        tokens_reg = m_tokens.get(best_tokens_name) or next(iter(m_tokens.values()))

        pred_time_ms = int(np.expm1(time_reg.predict(mX)[0]))
        pred_tokens  = int(np.expm1(tokens_reg.predict(mX)[0]))

        # ── Cost ──────────────────────────────────────────────────────────────
        price_per_token = 0.0
        try:
            price_per_token = float(prices.get(llm, 0.0))
            if price_per_token == 0.0:
                for k, v in prices.items():
                    kn = k.lower().replace(" ", "_")
                    if kn == llm or llm in kn or kn in llm:
                        price_per_token = float(v)
                        break
        except Exception:
            price_per_token = 0.0

        results[llm] = {
            "quality_score":   quality_score,
            "pred_time_ms":    pred_time_ms,
            "pred_tokens":     pred_tokens,
            "pred_cost":       float(pred_tokens) * price_per_token,
            "price_per_token": price_per_token,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Composite ranking
# ─────────────────────────────────────────────────────────────────────────────
# Composite weights (must sum to 1.0)
W_QUALITY = 0.50   # quality score (0-10)  → higher is better
W_COST    = 0.20   # predicted cost        → lower is better
W_TIME    = 0.20   # generation time       → lower is better
W_TOKENS  = 0.10   # total tokens          → lower is better


def compute_composite(results: dict,
                      w_quality=W_QUALITY, w_cost=W_COST,
                      w_time=W_TIME, w_tokens=W_TOKENS):
    """
    Additive 0-100 composite score.
    Quality score (0-10) is normalised to 0-1; cost/time/tokens are inverted
    so lower actual values score higher. Final score range: 0-100.
    """
    llms   = list(results.keys())
    scores = np.array([results[l]["quality_score"]       for l in llms], dtype=float)
    times  = np.array([results[l]["pred_time_ms"]        for l in llms], dtype=float)
    tokens = np.array([results[l]["pred_tokens"]         for l in llms], dtype=float)
    costs  = np.array([results[l].get("pred_cost", 0.0)  for l in llms], dtype=float)

    def norm(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 0 else np.full_like(arr, 0.5)

    composite = (
        w_quality * norm(scores)           # quality  : higher → better
        + w_cost  * (1.0 - norm(costs))    # cost     : lower  → better
        + w_time  * (1.0 - norm(times))    # speed    : lower  → better
        + w_tokens * (1.0 - norm(tokens))  # tokens   : lower  → better
    ) * 100.0

    for i, llm in enumerate(llms):
        results[llm]["composite"] = float(composite[i])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print tables
# ─────────────────────────────────────────────────────────────────────────────
def print_section(title):
    print()
    line = "─" * 70
    print(bold(cyan(line)))
    print(bold(cyan(f"  {title}")))
    print(bold(cyan(line)))


def print_smell_analysis(orig_code: str):
    """Print code smell analysis of the input function."""
    print_section("PRE-ANALYSIS — Code Smells Detected in Input Function")
    n_smells, detected = smell_summary(orig_code)
    total = len(SMELL_NAMES)

    smell_display = {
        "long_method":     "Long Method         (>20 non-empty lines)",
        "long_param_list": "Long Parameter List (>4 params)",
        "deep_nesting":    "Deep Nesting        (indent >= 4 levels)",
        "magic_numbers":   "Magic Numbers       (bare numeric literals)",
        "no_docstring":    "No Docstring",
        "no_type_hints":   "No Type Hints",
        "long_lines":      "Long Lines          (>79 chars)",
        "commented_code":  "Commented-out Code",
        "poor_naming":     "Poor Naming         (single-char variables)",
    }

    if n_smells > 4:
        smell_count_str = red(str(n_smells))
    elif n_smells > 1:
        smell_count_str = yellow(str(n_smells))
    else:
        smell_count_str = green(str(n_smells))

    print(f"\n  Smells found: {smell_count_str} / {total}")
    print()
    for s in SMELL_NAMES:
        found = s in detected
        icon  = red("  ✗  ") if found else green("  ✓  ")
        label = smell_display.get(s, s)
        print(f"  {icon}{label}")

    print()
    if n_smells == 0:
        print(f"  {green('No smells detected — clean code!')} "
              f"Quality improvement potential is limited to docs/annotations.")
    else:
        print(f"  {dim(f'LLMs that fix more of these {n_smells} smells will score higher.')}")


def print_quality_table(results: dict):
    print_section("SECTION A — Refactoring Quality Score Ranking  (0-10)")
    ranked = sorted(results.items(), key=lambda x: x[1]["quality_score"], reverse=True)

    header = f"  {'Rank':<5}{'LLM':<22}{'Score':>7}  {'Grade'}"
    print(bold(header))
    print("  " + "·" * 55)

    medals = ["🥇", "🥈", "🥉", "  4.", "  5."]
    for i, (llm, v) in enumerate(ranked):
        score = v["quality_score"]
        grade = quality_grade(score)
        col   = quality_color(score)
        disp  = MODEL_DISPLAY[llm]
        score_str = f"{score:.1f}/10"
        row = f"  {medals[i]:<5}{disp:<22}{col(score_str):>7}   {col(grade)}"
        print(row)

    print()
    print(dim("  Score pillars: Code Presence(+1) · Docstring(+1) · Type Hints(+1)"))
    print(dim("                 Future Import(+1) · Smell Removal(+5) · Structural(+1)"))


def print_metrics_table(results: dict):
    print_section("SECTION B — Predicted LLM Generation Metrics")

    all_times   = [v["pred_time_ms"] for _, v in results.items()]
    all_tokens  = [v["pred_tokens"]  for _, v in results.items()]
    all_costs   = [v.get("pred_cost", 0.0) for _, v in results.items()]
    min_t, min_k, min_c = min(all_times), min(all_tokens), min(all_costs)

    print(bold(f"  {'LLM':<20}  {'Time (ms)':>12}  {'Tokens':>8}  {'Cost (USD)':>12}"))
    print("  " + "─" * 62)

    for llm in MODEL_ORDER:
        v      = results[llm]
        t_ms   = v["pred_time_ms"]
        tokens = v["pred_tokens"]
        cost   = v.get("pred_cost", 0.0)
        disp   = MODEL_DISPLAY[llm]

        t_str = f"{t_ms:,} ms"
        k_str = f"{tokens:,}"
        c_str = f"${cost:,.6f}"

        if t_ms == min_t:   t_str = green(t_str + " ★")
        if tokens == min_k: k_str = green(k_str + " ★")
        if cost == min_c:   c_str = green(c_str + " ★")

        print(f"  {disp:<20}  {t_str:>12}  {k_str:>8}  {c_str:>12}")

    print(f"\n  {green('★')} = best (lowest) value")


def print_composite_table(results: dict):
    print_section("SECTION C — Overall Ranking  (Score out of 100)")
    print(dim(f"  Weights: quality {W_QUALITY:.0%}  · cost {W_COST:.0%}  · speed {W_TIME:.0%}  · tokens {W_TOKENS:.0%}"))
    print(dim("  (cost/speed/tokens: lower actual value → higher score component)"))
    print()

    ranked     = sorted(results.items(), key=lambda x: x[1]["composite"], reverse=True)
    best_score = ranked[0][1]["composite"]

    print(bold(f"  {'Rank':<6}{'LLM':<20}  {'Quality':>9}  {'Time ms':>8}  "
               f"{'Tokens':>7}  {'Cost':>10}  {'Score':>6}"))
    print("  " + "─" * 78)

    medals = ["🥇", "🥈", "🥉", "  4.", "  5."]
    for i, (llm, v) in enumerate(ranked):
        disp  = MODEL_DISPLAY[llm]
        qs    = f"{v['quality_score']:.1f}/10"
        t_ms  = f"{v['pred_time_ms']:,}"
        tok   = f"{v['pred_tokens']:,}"
        cost  = f"${v.get('pred_cost', 0.0):,.6f}"
        sc    = v["composite"]
        sc_str = f"{sc:.1f}"

        if i == 0:                    sc_col = bold(green(sc_str))
        elif sc >= best_score * 0.85: sc_col = green(sc_str)
        elif sc >= best_score * 0.65: sc_col = yellow(sc_str)
        else:                         sc_col = red(sc_str)

        prefix = f"  {medals[i]:<6}{disp:<20}  {qs:>9}  {t_ms:>8}  {tok:>7}  {cost:>10}  "
        print((bold(prefix) if i == 0 else prefix) + sc_col)

    print()
    best_llm  = ranked[0][0]
    best_sc   = ranked[0][1]["composite"]
    best_qs   = ranked[0][1]["quality_score"]
    print(bold(f"  ➤  Best overall LLM: {green(MODEL_DISPLAY[best_llm])}"
               f"  ({green(f'{best_sc:.1f}/100')}, quality score {green(f'{best_qs:.1f}/10')})"))
    print()
    cheapest  = min(results.items(), key=lambda x: x[1].get("pred_cost", float("inf")))[0]
    fastest   = min(results.items(), key=lambda x: x[1]["pred_time_ms"])[0]
    highest_q = max(results.items(), key=lambda x: x[1]["quality_score"])[0]
    print(dim(f"  💡 Best quality       : {MODEL_DISPLAY[highest_q]}"
              f"  (score {results[highest_q]['quality_score']:.1f}/10)"))
    print(dim(f"  💰 Most cost-efficient: {MODEL_DISPLAY[cheapest]}"
              f"  (${results[cheapest].get('pred_cost', 0):.6f})"))
    print(dim(f"  ⚡ Fastest            : {MODEL_DISPLAY[fastest]}"
              f"  ({results[fastest]['pred_time_ms']:,} ms)"))


# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_code_interactive() -> str:
    print()
    print(bold("Paste your Python function below."))
    print(dim("  End input with an empty line followed by EOF (Ctrl+D on Linux/Mac, Ctrl+Z on Windows)"))
    print(dim("  or type END on its own line and press Enter."))
    print()
    lines = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def infer_func_name(code: str) -> str:
    """Extract function name from def statement."""
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    return m.group(1) if m else "unknown_function"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Rank LLMs for refactoring quality using trained ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python refactor_cli.py
  python refactor_cli.py --file my_function.py
  cat my_function.py | python refactor_cli.py --stdin
        """,
    )
    parser.add_argument("--file",  "-f", help="Path to .py file containing the function")
    parser.add_argument("--stdin", "-s", action="store_true", help="Read code from stdin pipe")
    parser.add_argument("--best-only", action="store_true",
                        help="Only print the best overall LLM name (machine-readable)")
    parser.add_argument("--no-smell-analysis", action="store_true",
                        help="Skip the code smell pre-analysis section")
    args = parser.parse_args()

    # ── Banner ─────────────────────────────────────────────────────────────
    if not args.best_only:
        print()
        print(bold(cyan("╔══════════════════════════════════════════════════════════════════╗")))
        print(bold(cyan("║       LLM Refactoring Quality Predictor  (CLI)  v2.0            ║")))
        print(bold(cyan("║       Quality Score: continuous 0-10  (smell-aware)             ║")))
        print(bold(cyan("╚══════════════════════════════════════════════════════════════════╝")))

    # ── Load models ─────────────────────────────────────────────────────────
    (q_regressors, q_scaler, q_le, q_stats, q_meta,
     m_time, m_tokens, m_scaler, m_le, m_stats, m_meta, prices) = load_models()

    # ── Get code ─────────────────────────────────────────────────────────────
    if args.file:
        with open(args.file, "r") as fh:
            code = fh.read()
    elif args.stdin or not sys.stdin.isatty():
        code = sys.stdin.read()
    else:
        code = read_code_interactive()

    if not code.strip():
        print(red("No code provided. Exiting."))
        sys.exit(1)

    func_name = infer_func_name(code)
    n_input_smells, _ = smell_summary(code)

    if not args.best_only:
        if n_input_smells > 4:
            smell_count_str = red(str(n_input_smells))
        elif n_input_smells > 1:
            smell_count_str = yellow(str(n_input_smells))
        else:
            smell_count_str = green(str(n_input_smells))
        print(f"\n  {bold('Function detected:')} {cyan(func_name)}")
        print(f"  {bold('Lines:')} {count_lines(code)}   "
              f"{bold('Smells in input:')} {smell_count_str} / {len(SMELL_NAMES)}")

    # ── Code smell pre-analysis ────────────────────────────────────────────
    if not args.best_only and not args.no_smell_analysis:
        print_smell_analysis(code)

    # ── Predict ──────────────────────────────────────────────────────────────
    if not args.best_only:
        print(dim("\n  Running predictions for all 5 LLMs..."), end=" ", flush=True)

    results = predict_all(
        code, func_name,
        q_regressors, q_scaler, q_le, q_stats, q_meta,
        m_time, m_tokens, m_scaler, m_le, m_stats, m_meta,
        prices,
    )
    results = compute_composite(results)

    if not args.best_only:
        print(green("done"))

    # ── Output ───────────────────────────────────────────────────────────────
    if args.best_only:
        best = max(results.items(), key=lambda x: x[1]["composite"])[0]
        print(MODEL_DISPLAY[best])
        return

    print_quality_table(results)
    print_metrics_table(results)
    print_composite_table(results)

    print()
    print(bold(cyan("─" * 72)))
    print(dim("  Models used:"))
    print(dim(f"    Quality regressor  : {q_meta['best_model']}  (regression, score 0-10)"))
    print(dim(f"    Time regressor     : {m_meta.get('best_time_model', '?')}"))
    print(dim(f"    Token regressor    : {m_meta.get('best_tokens_model', '?')}"))
    print(bold(cyan("─" * 72)))
    print()


if __name__ == "__main__":
    main()
