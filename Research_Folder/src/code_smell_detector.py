"""
Code Smell Detector
===================
Detects long-method code smell in Python functions using a trained ML model.

Uses the same feature engineering and SVM model approach from
model_synthetic.ipynb — 12 safe features with synthetic augmentation.

The model is trained by `train_smell_model.py` which replicates the
notebook's methodology: Gaussian noise + SMOTE augmentation, then SVM
classification with ~95% CV accuracy.
"""

import re
import os
import sys
import warnings
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent.parent  # app/
MODELS_DIR   = SCRIPT_DIR / "models"
SMELL_MODEL  = MODELS_DIR / "smell_svm.pkl"
SMELL_SCALER = MODELS_DIR / "smell_scaler.pkl"

# ── Safe features (must match model_synthetic.ipynb exactly) ─────────────────
SAFE_FEATURES = [
    "num_lines",
    "code_avg_line_len",
    "code_max_line_len",
    "code_indent_depth",
    "code_has_loop",
    "code_has_conditional",
    "code_has_return",
    "code_has_try",
    "code_num_returns",
    "code_num_ifs",
    "code_num_loops",
    "code_comment_lines",
]


def extract_code_features(code_str: str) -> dict:
    """
    Extract the 12 safe features from a Python function code string.
    Mirrors the logic in model.ipynb and model_synthetic.ipynb.
    """
    code = str(code_str)
    lines = code.replace("\\n", "\n").split("\n")
    nonempty = [l for l in lines if l.strip()]

    num_lines = float(len(nonempty))
    line_lengths = [len(l) for l in nonempty]

    features = {
        "num_lines":             num_lines,
        "code_avg_line_len":     float(np.mean(line_lengths)) if line_lengths else 0.0,
        "code_max_line_len":     float(max(line_lengths)) if line_lengths else 0.0,
        "code_indent_depth":     float(max(len(l) - len(l.lstrip()) for l in nonempty)) if nonempty else 0.0,
        "code_has_loop":         int(any(kw in code for kw in ["for ", "while "])),
        "code_has_conditional":  int(any(kw in code for kw in ["if ", "elif ", "else:"])),
        "code_has_return":       int("return" in code),
        "code_has_try":          int("try:" in code or "except" in code),
        "code_num_returns":      float(code.count("return")),
        "code_num_ifs":          float(code.count("if ")),
        "code_num_loops":        float(code.count("for ") + code.count("while ")),
        "code_comment_lines":    float(sum(1 for l in lines if l.strip().startswith("#"))),
    }
    return features


def extract_features_vector(code_str: str) -> np.ndarray:
    """Return a (1, 12) numpy array for model inference."""
    feats = extract_code_features(code_str)
    return np.array([[feats[f] for f in SAFE_FEATURES]], dtype=float)


# ── Global model cache ───────────────────────────────────────────────────────
_model = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    if not SMELL_MODEL.exists() or not SMELL_SCALER.exists():
        print("[code_smell_detector] Trained model not found. Run 'python src/train_smell_model.py' first.",
              file=sys.stderr)
        sys.exit(1)

    _model = joblib.load(str(SMELL_MODEL))
    _scaler = joblib.load(str(SMELL_SCALER))
    return _model, _scaler


def predict_smell(code_str: str) -> dict:
    """
    Predict whether a Python function has the long-method code smell.

    Returns:
        {
            "is_bad_smell": bool,      # True if predicted as long method
            "probability": float,       # probability of being bad (0-1)
            "num_lines": int,           # non-empty lines in function
            "is_long_method": bool,     # heuristic: > 20 lines
        }
    """
    model, scaler = _load_model()
    X = extract_features_vector(code_str)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0, 1]  # probability of "Yes" (bad smell)
    pred = model.predict(X_scaled)[0]

    nonempty = [l for l in code_str.replace("\\n", "\n").split("\n") if l.strip()]
    num_lines = len(nonempty)

    return {
        "is_bad_smell": bool(pred == 1),
        "probability": float(proba),
        "num_lines": num_lines,
        "is_long_method": num_lines > 20,  # heuristic baseline
    }


def predict_smell_batch(codes: list) -> list:
    """Batch prediction for multiple functions."""
    model, scaler = _load_model()
    X = np.vstack([extract_features_vector(c) for c in codes])
    X_scaled = scaler.transform(X)

    probas = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    results = []
    for i, code in enumerate(codes):
        nonempty = [l for l in code.replace("\\n", "\n").split("\n") if l.strip()]
        results.append({
            "is_bad_smell": bool(preds[i] == 1),
            "probability": float(probas[i]),
            "num_lines": len(nonempty),
            "is_long_method": len(nonempty) > 20,
        })
    return results
