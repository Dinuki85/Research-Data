#!/usr/bin/env python3
"""
Train Smell Detection Model
============================
Trains an SVM classifier to detect long-method code smell in Python functions.

Replicates the approach from codeworks/model_synthetic.ipynb:
  1. Load dataset.csv from codeworks/
  2. Extract 12 safe features (avoids data leakage)
  3. Gaussian noise augmentation (4 copies, sigma=0.25, num_lines_noise=12)
  4. SMOTE oversampling for class balance
  5. Train SVM with standardization
  6. Save model and scaler to app/models/

Usage:
    python src/train_smell_model.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CODEWORKS_CSV = Path(__file__).resolve().parent.parent.parent / "codeworks" / "dataset.csv"
MODELS_DIR    = Path(__file__).resolve().parent.parent / "models"

# ── Feature definitions (must match code_smell_detector.py) ─────────────────
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

BINARY_COLS = [
    "code_has_loop",
    "code_has_conditional",
    "code_has_return",
    "code_has_try",
]

SEED = 42
rng = np.random.default_rng(SEED)


def extract_code_features(code):
    """Same as code_smell_detector.extract_code_features."""
    code = str(code)
    lines = code.split("\n")
    nonempty = [l for l in lines if l.strip()]
    line_lengths = [len(l) for l in nonempty]

    return pd.Series({
        "num_lines":             len(nonempty),
        "code_avg_line_len":     float(np.mean(line_lengths)) if line_lengths else 0.0,
        "code_max_line_len":     float(max(line_lengths)) if line_lengths else 0.0,
        "code_indent_depth":     float(max(len(l) - len(l.lstrip()) for l in nonempty)) if nonempty else 0.0,
        "code_has_loop":         int(any(kw in code for kw in ["for ", "while "])),
        "code_has_conditional":  int(any(kw in code for kw in ["if ", "elif ", "else:"])),
        "code_has_return":       int("return" in code),
        "code_has_try":          int("try:" in code or "except" in code),
        "code_num_returns":      code.count("return"),
        "code_num_ifs":          code.count("if "),
        "code_num_loops":        code.count("for ") + code.count("while "),
        "code_comment_lines":    sum(1 for l in lines if l.strip().startswith("#")),
    })


def generate_synthetic_dataset():
    """Generate a synthetic training dataset when codeworks/dataset.csv is not available.

    Creates sample Python functions with known labels for long-method smell,
    so the model can be trained fresh on any environment (e.g., CI/CD runners).
    """
    print("  codeworks/dataset.csv not found — generating synthetic training data")
    rng = np.random.default_rng(SEED)

    # Templates for short/clean functions (label="No")
    short_templates = [
        # Simple getter/setter
        "\"\"\"Get the value.\"\"\"\nreturn self._value",
        "\"\"\"Set the name.\"\"\"\nself._name = name",
        # One-liner
        "return sum(data) / len(data)",
        "return [x for x in items if x > 0]",
        "return a + b",
        # Short with simple logic
        "\"\"\"Check if valid.\"\"\"\nreturn value is not None and value > 0",
        "\"\"\"Format the string.\"\"\"\nreturn f\"{prefix}: {msg}\"",
        # 3-5 line functions
        "result = []\nfor i in range(n):\n    result.append(i * 2)\nreturn result",
        "total = 0\nfor x in numbers:\n    total += x\nreturn total / len(numbers)",
        "count = 0\nfor item in collection:\n    if item.active:\n        count += 1\nreturn count",
        # With type hints
        "def wrapper(func):\n    \"\"\"Simple decorator.\"\"\"\n    return func",
        # Conditional
        "if condition:\n    return option_a\nreturn option_b",
        "if x < 0:\n    return -1\nelif x == 0:\n    return 0\nreturn 1",
        # Simple try/except
        "try:\n    return int(value)\nexcept ValueError:\n    return 0",
    ]

    # Templates for long/bad-smell functions (label="Yes")
    long_templates = [
        # Long function with loops, conditionals, try/except
        "\"\"\"Process data with full validation pipeline.\"\"\"\n"
        "errors = []\n"
        "results = []\n"
        "for idx, item in enumerate(data):\n"
        "    try:\n"
        "        if not isinstance(item, dict):\n"
        "            errors.append(f\"Item {idx} is not a dict\")\n"
        "            continue\n"
        "        # Validate required fields\n"
        "        required = ['id', 'name', 'value']\n"
        "        missing = [f for f in required if f not in item]\n"
        "        if missing:\n"
        "            errors.append(f\"Item {idx} missing: {missing}\")\n"
        "            continue\n"
        "        # Transform\n"
        "        transformed = {\n"
        "            'id': item['id'],\n"
        "            'name': item['name'].strip(),\n"
        "            'value': float(item['value']),\n"
        "            'processed_at': datetime.now().isoformat(),\n"
        "        }\n"
        "        results.append(transformed)\n"
        "    except Exception as e:\n"
        "        errors.append(f\"Item {idx} error: {e}\")\n"
        "return {'results': results, 'errors': errors, 'count': len(results)}",

        # Long config parser
        "\"\"\"Parse and validate configuration from a file.\"\"\"\n"
        "config = {}\n"
        "with open(filepath, 'r') as f:\n"
        "    for line in f:\n"
        "        line = line.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        if '=' not in line:\n"
        "            continue\n"
        "        key, value = line.split('=', 1)\n"
        "        key = key.strip()\n"
        "        value = value.strip()\n"
        "        # Parse value type\n"
        "        if value.lower() in ('true', 'yes', '1'):\n"
        "            config[key] = True\n"
        "        elif value.lower() in ('false', 'no', '0'):\n"
        "            config[key] = False\n"
        "        else:\n"
        "            try:\n"
        "                config[key] = int(value)\n"
        "            except ValueError:\n"
        "                try:\n"
        "                    config[key] = float(value)\n"
        "                except ValueError:\n"
        "                    config[key] = value\n"
        "# Validate required keys\n"
        "required = ['host', 'port', 'debug']\n"
        "missing = [k for k in required if k not in config]\n"
        "if missing:\n"
        "    raise ValueError(f\"Missing required config: {missing}\")\n"
        "# Apply defaults\n"
        "config.setdefault('timeout', 30)\n"
        "config.setdefault('retries', 3)\n"
        "return config",

        # Batch processor
        "\"\"\"Process items in batches with retry logic.\"\"\"\n"
        "results = []\n"
        "failed = []\n"
        "for i in range(0, len(items), batch_size):\n"
        "    batch = items[i:i + batch_size]\n"
        "    for attempt in range(max_retries):\n"
        "        try:\n"
        "            batch_results = processor.process_batch(batch)\n"
        "            results.extend(batch_results)\n"
        "            break\n"
        "        except ConnectionError as e:\n"
        "            if attempt == max_retries - 1:\n"
        "                failed.extend(batch)\n"
        "            time.sleep(2 ** attempt)\n"
        "        except Exception as e:\n"
        "            failed.append({'batch': batch, 'error': str(e)})\n"
        "            break\n"
        "summary = {\n"
        "    'total': len(items),\n"
        "    'processed': len(results),\n"
        "    'failed': len(failed),\n"
        "    'success_rate': len(results) / len(items) * 100,\n"
        "}\n"
        "return results, failed, summary",
    ]

    # Generate features and labels for short functions
    short_codes = []
    for t in short_templates:
        short_codes.append(t)
    # Generate features for short functions
    short_feats = [extract_code_features(c) for c in short_codes]
    short_labels = ["No"] * len(short_codes)

    # Generate features for long functions
    long_codes = []
    for t in long_templates:
        long_codes.append(t)
    long_feats = [extract_code_features(c) for c in long_codes]
    long_labels = ["Yes"] * len(long_codes)

    # Add variations via feature perturbation to expand dataset
    for _ in range(3):
        for t in short_templates:
            # Perturb short functions slightly
            padded = t + "\n    # extra comment\n    pass"
            short_codes.append(padded)
            short_feats.append(extract_code_features(padded))
            short_labels.append("No")

    for _ in range(3):
        for t in long_templates:
            # Extend long functions further
            extended = t + "\n    # additional logging\n    logger.info(f\"Processed batch\")\n    metrics.record(count=len(results))"
            long_codes.append(extended)
            long_feats.append(extract_code_features(extended))
            long_labels.append("Yes")

    all_feats = short_feats + long_feats
    all_labels = short_labels + long_labels

    synthetic_df = pd.DataFrame(all_feats)
    synthetic_df["long_method"] = all_labels
    print(f"  Generated {len(synthetic_df)} synthetic training examples "
          f"(No={all_labels.count('No')}, Yes={all_labels.count('Yes')})")
    return synthetic_df


def main():
    print("=" * 60)
    print("  Training Long-Method Code Smell Detector")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    if CODEWORKS_CSV.exists():
        df_raw = pd.read_csv(str(CODEWORKS_CSV), encoding="latin-1")
        df_raw = df_raw[["code_section", "Approximate Number of Lines", "Long Method"]].copy()
        df_raw.columns = ["code_section", "num_lines", "long_method"]
        print(f"\nLoaded {len(df_raw)} rows from dataset.csv")
        # ── 2. Extract features ──────────────────────────────────────────────────
        feats = df_raw["code_section"].apply(extract_code_features)
        df_all = pd.concat([feats, df_raw[["long_method"]]], axis=1)
    else:
        df_all = generate_synthetic_dataset()

    le = LabelEncoder()
    y = le.fit_transform(df_all["long_method"])  # No=0, Yes=1
    print(f"Class distribution: No={sum(y==0)}, Yes={sum(y==1)}")

    X = df_all[SAFE_FEATURES].copy()
    print(f"Feature matrix: {X.shape}")

    # ── 3. Train/test split (stratified) ─────────────────────────────────────
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X.values, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"\nTrain: {len(X_train_real)}, Test: {len(X_test_real)}")

    # ── 4. Augment training data ─────────────────────────────────────────────
    feature_stds = X.std()
    binary_idx = [SAFE_FEATURES.index(c) for c in BINARY_COLS]
    num_lines_idx = SAFE_FEATURES.index("num_lines")

    SIGMA = 0.25
    NUM_LINES_NOISE_STD = 12.0
    N_NOISE_COPIES = 4

    X_noise_list, y_noise_list = [], []
    for _ in range(N_NOISE_COPIES):
        noise = rng.normal(loc=0.0, scale=(SIGMA * feature_stds).values, size=X_train_real.shape)
        noise[:, num_lines_idx] = rng.normal(0, NUM_LINES_NOISE_STD, size=len(X_train_real))
        X_n = X_train_real + noise
        X_n[:, binary_idx] = np.clip(np.round(X_n[:, binary_idx]), 0, 1)
        X_n = np.clip(X_n, 0, None)
        X_noise_list.append(X_n)
        y_noise_list.append(y_train_real.copy())

    X_augmented = np.vstack([X_train_real] + X_noise_list)
    y_augmented = np.concatenate([y_train_real] + y_noise_list)

    # SMOTE for balance
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_aug, y_train_aug = smote.fit_resample(X_augmented, y_augmented)

    print(f"Augmented train: {X_train_aug.shape}, classes: {np.bincount(y_train_aug)}")

    # ── 5. Train SVM ─────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=1.0, probability=True, random_state=SEED)),
    ])

    pipeline.fit(X_train_aug, y_train_aug)

    # ── 6. Cross-validation ──────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_acc = cross_val_score(pipeline, X_train_aug, y_train_aug, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(pipeline, X_train_aug, y_train_aug, cv=cv, scoring="f1")

    print(f"\nCV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"CV F1 Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── 7. Evaluate on real test set ─────────────────────────────────────────
    acc = pipeline.score(X_test_real, y_test_real)
    print(f"Test Accuracy (real data): {acc:.4f}")

    # ── 8. Save model ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["svm"], str(MODELS_DIR / "smell_svm.pkl"))
    joblib.dump(pipeline.named_steps["scaler"], str(MODELS_DIR / "smell_scaler.pkl"))
    print(f"\n✅ Model saved to: {MODELS_DIR / 'smell_svm.pkl'}")
    print(f"✅ Scaler saved to: {MODELS_DIR / 'smell_scaler.pkl'}")


if __name__ == "__main__":
    main()
