"""
LLM Recommender
===============
Predicts the best LLM for refactoring a given Python function using trained
ML models (quality regression + metrics regression).

Reuses the feature engineering and model loading logic from
codeworks4/refactor_cli.py. The quality model predicts a continuous 0-10
refactoring quality score, and the metrics model predicts generation time
and token usage for each of the 5 supported LLMs.

Composite score (0-100): 50% quality + 20% cost + 20% time + 10% tokens.
"""

import re
import os
import sys
import csv
import warnings
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR   = SCRIPT_DIR / "models"

MODEL_ORDER = ["claude_opus", "claude_sonnet_4_6", "gemini_3_1_pro", "gemini_flash", "gpt_oss"]
MODEL_DISPLAY = {
    "claude_opus":        "Claude Opus",
    "claude_sonnet_4_6":  "Claude Sonnet 4.6",
    "gemini_3_1_pro":     "Gemini 3.1 Pro",
    "gemini_flash":       "Gemini Flash",
    "gpt_oss":            "GPT-OSS 120B",
}

# OpenRouter model slugs
OPENROUTER_MODEL_MAP = {
    "claude_opus":       "anthropic/claude-opus-4.5",
    "claude_sonnet_4_6": "anthropic/claude-sonnet-4.5",
    "gemini_3_1_pro":    "google/gemini-3.1-pro-preview",
    "gemini_flash":      "google/gemini-3.5-flash",
    "gpt_oss":           "openai/gpt-oss-120b",
}

# ── Smell names (must match refactoring_quality_ml.ipynb) ───────────────────
SMELL_NAMES = [
    "long_method", "long_param_list", "deep_nesting", "magic_numbers",
    "no_docstring", "no_type_hints", "long_lines", "commented_code", "poor_naming",
]

# ── Helper functions ─────────────────────────────────────────────────────────


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
    return int(isinstance(s, str) and bool(
        re.search(r"->\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)", s)
    ))


def has_future_import(s: str) -> int:
    return int(isinstance(s, str) and "from __future__ import annotations" in s)


def char_token_est(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return max(0, len(s.replace("\\n", "\n")) // 4)


# ── Code smell detectors ─────────────────────────────────────────────────────


def detect_code_smells(code_str: str) -> dict:
    """Detect all 9 code smells. Returns {smell_name: 0/1}."""
    empty = {s: 0 for s in SMELL_NAMES}
    if not isinstance(code_str, str) or len(code_str.strip()) < 5:
        return empty

    code = code_str.replace("\\n", "\n")
    lines = code.split("\n")
    nonempty = [l for l in lines if l.strip()]
    smells = {}

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

    # 3. Deep Nesting
    max_indent = max((len(l) - len(l.lstrip())) for l in nonempty) if nonempty else 0
    smells["deep_nesting"] = int(max_indent >= 16)

    # 4. Magic Numbers
    _allowed_nums = {"0", "1", "2", "3", "10", "100", "-1", "0.0", "1.0"}
    nums = re.findall(r"(?<!\w)(\d+\.?\d*)(?!\w)", code)
    magic = [n for n in nums if n not in _allowed_nums]
    smells["magic_numbers"] = int(len(magic) > 2)

    # 5. No Docstring
    smells["no_docstring"] = int('"""' not in code and "'''" not in code)

    # 6. No Type Hints
    smells["no_type_hints"] = int(not bool(re.search(
        r"->\s*\w|:\s*(str|int|float|bool|None|list|dict|tuple|Any|Optional|List|Dict|Tuple)", code)))

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


# ── Model loading ────────────────────────────────────────────────────────────

_models_loaded = False
_q_regressors = {}
_q_scaler = None
_q_le = None
_q_stats = None
_q_meta = None
_m_time = {}
_m_tokens = {}
_m_scaler = None
_m_le = None
_m_stats = None
_m_meta = None
_prices = {}


def _safe_joblib_load(path):
    """Try to load a joblib pickle. Return None on failure (version mismatch)."""
    try:
        return joblib.load(str(path))
    except Exception as e:
        print(f"  [llm_recommender] ⚠ Could not load {path.name}: {e}", file=sys.stderr)
        print(f"  [llm_recommender]   Using fallback defaults.", file=sys.stderr)
        return None


def _load_models():
    global _models_loaded, _q_regressors, _q_scaler, _q_le, _q_stats, _q_meta
    global _m_time, _m_tokens, _m_scaler, _m_le, _m_stats, _m_meta, _prices

    if _models_loaded:
        return

    # Quality regressors (best-effort)
    for name in ["ridge_regression", "decision_tree", "random_forest",
                 "gradient_boosting", "xgboost", "svr"]:
        fpath = MODELS_DIR / f"quality_{name}.pkl"
        loaded = _safe_joblib_load(fpath) if fpath.exists() else None
        if loaded is not None:
            _q_regressors[name] = loaded

    _q_scaler   = _safe_joblib_load(MODELS_DIR / "quality_scaler.pkl")
    _q_le       = _safe_joblib_load(MODELS_DIR / "quality_label_encoder.pkl")
    _q_stats    = _safe_joblib_load(MODELS_DIR / "quality_stats.pkl")
    _q_meta     = _safe_joblib_load(MODELS_DIR / "quality_meta.pkl")

    # Metrics regressors (best-effort)
    for name in ["ridge", "random_forest", "gradient_boosting", "xgboost"]:
        for d, prefix in [(_m_time, "metrics_time"), (_m_tokens, "metrics_tokens")]:
            fpath = MODELS_DIR / f"{prefix}_{name}.pkl"
            loaded = _safe_joblib_load(fpath) if fpath.exists() else None
            if loaded is not None:
                d[name] = loaded

    _m_scaler = _safe_joblib_load(MODELS_DIR / "metrics_scaler.pkl")
    _m_le     = _safe_joblib_load(MODELS_DIR / "metrics_label_encoder.pkl")
    _m_stats  = _safe_joblib_load(MODELS_DIR / "metrics_stats.pkl")
    _m_meta   = _safe_joblib_load(MODELS_DIR / "metrics_meta.pkl")

    # Load prices
    prices_path = SCRIPT_DIR / "price.csv"
    if prices_path.exists():
        with open(str(prices_path), newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get("Model Family") or row.get("Model") or row.get("model")
                inp = row.get("Input") or row.get("input")
                if not key or not inp:
                    continue
                s = str(inp).strip()
                nums = re.findall(r"[0-9]+\.?[0-9]*(?:e[+-]?\d+)?", s)
                try:
                    vals = [float(x) for x in nums]
                    if vals:
                        _prices[key.strip()] = sum(vals) / len(vals)
                except Exception:
                    continue

    _models_loaded = True


# ── Feature engineering ──────────────────────────────────────────────────────


def quality_features(orig_code: str, refact_code: str, func_name: str,
                     model_enc: int) -> np.ndarray:
    """Build 50 features (19 structural + 31 smell) for quality prediction."""
    q_stats = _q_stats

    orig_line_count = float(count_lines(orig_code))
    long_method_flag = int(orig_line_count > 15)
    refact_line_count = float(count_lines(refact_code))
    line_delta = orig_line_count - refact_line_count
    line_delta_ratio = line_delta / (orig_line_count + 1)
    hfi = has_future_import(refact_code)
    hds = has_docstring(refact_code)
    hth = has_type_hints(refact_code)
    complexity_proxy = count_keywords(refact_code)
    orig_complexity = count_keywords(orig_code)
    complexity_delta = orig_complexity - complexity_proxy
    func_name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test_func = int(str(func_name).startswith("test"))
    repo_avg_lines = float(q_stats.get("global_repo_mean", orig_line_count))
    file_avg_lines = float(q_stats.get("global_file_mean", orig_line_count))
    global_max = float(q_stats.get("global_max_lines", max(orig_line_count, 1)))
    norm_line_count = orig_line_count / (global_max + 1)

    structural = [
        orig_line_count, long_method_flag, refact_line_count, line_delta,
        line_delta_ratio, hfi, hds, hth,
        complexity_proxy, orig_complexity, complexity_delta,
        func_name_len, is_private, is_dunder, is_test_func,
        repo_avg_lines, file_avg_lines, norm_line_count,
        model_enc,
    ]

    # Smell features (31)
    orig_smells = detect_code_smells(orig_code)
    refact_smells = detect_code_smells(refact_code)

    orig_smell_count = sum(orig_smells.values())
    refact_smell_count = sum(refact_smells.values())
    smell_reduction = max(0, orig_smell_count - refact_smell_count)
    smell_reduction_ratio = smell_reduction / (orig_smell_count + 1)

    smell_agg = [orig_smell_count, refact_smell_count, smell_reduction, smell_reduction_ratio]
    orig_per = [orig_smells[s] for s in SMELL_NAMES]
    refact_per = [refact_smells[s] for s in SMELL_NAMES]
    fixed_per = [max(0, orig_smells[s] - refact_smells[s]) for s in SMELL_NAMES]

    smell_features = smell_agg + orig_per + refact_per + fixed_per

    return np.array([structural + smell_features], dtype=float)


def metrics_features(orig_code: str, refact_code: str, func_name: str,
                     model_enc: int) -> np.ndarray:
    """Build 23 features for metrics (time/tokens) prediction."""
    m_stats = _m_stats

    orig_lines = float(count_lines(orig_code))
    long_method = int(orig_lines > 15)
    orig_complexity = count_keywords(orig_code)
    orig_char_tokens = char_token_est(orig_code)
    refact_lines = float(count_lines(refact_code))
    refact_complexity = count_keywords(refact_code)
    refact_char_tokens = char_token_est(refact_code)
    hds = has_docstring(refact_code)
    hth = has_type_hints(refact_code)
    hfi = has_future_import(refact_code)
    line_delta = orig_lines - refact_lines
    line_delta_ratio = line_delta / (orig_lines + 1)
    complexity_delta = orig_complexity - refact_complexity
    token_delta = orig_char_tokens - refact_char_tokens
    est_input_tokens = orig_char_tokens + 250
    est_output_tokens = refact_char_tokens
    func_name_len = len(str(func_name))
    is_private = int(str(func_name).startswith("_") and not str(func_name).startswith("__"))
    is_dunder = int(str(func_name).startswith("__") and str(func_name).endswith("__"))
    is_test_func = int(str(func_name).startswith("test"))
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


# ── Prediction ───────────────────────────────────────────────────────────────


def predict_all_llms(orig_code: str, func_name: str = "unknown_function") -> dict:
    """
    For each of the 5 LLMs, predict quality score, generation time, tokens,
    and cost. Returns a dict keyed by model name.

    Uses original code as proxy for refactored code (since we haven't
    actually refactored yet — delta features will be zero).
    Falls back to sensible defaults if ML models fail to load.
    """
    _load_models()
    refact_proxy = orig_code

    # Defaults when models are unavailable
    models_available = bool(_q_regressors) and _q_scaler is not None and _q_le is not None
    metrics_available = bool(_m_time) and _m_scaler is not None and _m_le is not None

    if models_available and _q_meta is not None:
        best_q_name = _q_meta["best_model"]
        best_q_safe = best_q_name.lower().replace(" ", "_")
        q_scaled_models = _q_stats.get("scaled_models", ["Ridge Regression", "SVR"]) if _q_stats is not None else []
    else:
        best_q_name = ""
        best_q_safe = ""
        q_scaled_models = []

    if metrics_available and _m_meta is not None:
        best_time_name = _m_meta.get("best_time_model", "gradient_boosting").lower().replace(" ", "_")
        best_tokens_name = _m_meta.get("best_tokens_model", "gradient_boosting").lower().replace(" ", "_")
    else:
        best_time_name = ""
        best_tokens_name = ""

    results = {}
    for llm in MODEL_ORDER:
        try:
            q_enc = int(_q_le.transform([llm])[0]) if _q_le is not None else MODEL_ORDER.index(llm)
        except Exception:
            q_enc = MODEL_ORDER.index(llm)
        try:
            m_enc = int(_m_le.transform([llm])[0]) if _m_le is not None else MODEL_ORDER.index(llm)
        except Exception:
            m_enc = MODEL_ORDER.index(llm)

        # Quality score (0-10)
        if models_available and best_q_safe in _q_regressors:
            qX = quality_features(orig_code, refact_proxy, func_name, q_enc)
            reg = _q_regressors[best_q_safe]
            try:
                if _q_scaler is not None and best_q_name in q_scaled_models:
                    qX_input = _q_scaler.transform(qX)
                else:
                    qX_input = qX
                quality_score = float(np.clip(reg.predict(qX_input)[0], 0.0, 10.0))
            except Exception:
                quality_score = 5.0
        else:
            quality_score = 5.0  # fallback default

        # Metrics (time + tokens)
        if metrics_available:
            mX = metrics_features(orig_code, refact_proxy, func_name, m_enc)
            time_reg = _m_time.get(best_time_name)
            tokens_reg = _m_tokens.get(best_tokens_name)
            if time_reg is not None and tokens_reg is not None:
                try:
                    pred_time_ms = int(np.expm1(time_reg.predict(mX)[0]))
                    pred_tokens = int(np.expm1(tokens_reg.predict(mX)[0]))
                except Exception:
                    pred_time_ms = 2000
                    pred_tokens = 500
            else:
                pred_time_ms = 2000
                pred_tokens = 500
        else:
            pred_time_ms = 2000
            pred_tokens = 500

        # Cost
        price_per_token = 0.0
        try:
            price_per_token = float(_prices.get(llm, 0.0))
            if price_per_token == 0.0:
                for k, v in _prices.items():
                    kn = k.lower().replace(" ", "_")
                    if kn == llm or llm in kn or kn in llm:
                        price_per_token = float(v)
                        break
        except Exception:
            price_per_token = 0.0

        results[llm] = {
            "quality_score": quality_score,
            "pred_time_ms": pred_time_ms,
            "pred_tokens": pred_tokens,
            "pred_cost": float(pred_tokens) * price_per_token,
            "price_per_token": price_per_token,
            "openrouter_slug": OPENROUTER_MODEL_MAP.get(llm, llm),
            "display_name": MODEL_DISPLAY.get(llm, llm),
        }

    # Compute composite score (0-100)
    llms = list(results.keys())
    scores = np.array([results[l]["quality_score"] for l in llms], dtype=float)
    times = np.array([results[l]["pred_time_ms"] for l in llms], dtype=float)
    tokens = np.array([results[l]["pred_tokens"] for l in llms], dtype=float)
    costs = np.array([results[l].get("pred_cost", 0.0) for l in llms], dtype=float)

    def norm(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 0 else np.full_like(arr, 0.5)

    composite = (
        0.50 * norm(scores)
        + 0.20 * (1.0 - norm(costs))
        + 0.20 * (1.0 - norm(times))
        + 0.10 * (1.0 - norm(tokens))
    ) * 100.0

    for i, llm in enumerate(llms):
        results[llm]["composite"] = float(composite[i])

    return results


def get_best_llm(orig_code: str, func_name: str = "unknown_function") -> dict:
    """
    Returns the single best LLM recommendation with full details.

    Returns:
        {
            "model_key": "claude_opus",
            "display_name": "Claude Opus",
            "openrouter_slug": "anthropic/claude-3-opus",
            "quality_score": 8.5,
            "pred_time_ms": 15000,
            "pred_tokens": 1200,
            "pred_cost": 0.0005,
            "composite": 92.3,
            "all_models": { ... }   # full ranking for reference
        }
    """
    results = predict_all_llms(orig_code, func_name)
    best_key = max(results, key=lambda k: results[k]["composite"])
    return {
        "model_key": best_key,
        **results[best_key],
        "all_models": results,
    }


def get_ranking(orig_code: str, func_name: str = "unknown_function") -> list:
    """
    Return a ranked list of all LLMs by composite score (descending).
    Each entry contains all prediction details.
    """
    results = predict_all_llms(orig_code, func_name)
    ranked = sorted(results.items(), key=lambda x: x[1]["composite"], reverse=True)
    return [{"model_key": k, **v} for k, v in ranked]
