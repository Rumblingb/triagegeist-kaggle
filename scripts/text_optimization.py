#!/usr/bin/env python3
"""
Text feature optimization for Triagegeist Kaggle.

Runs comprehensive experiments on chief_complaint_raw text features:
  (a) TF-IDF param sweep (ngram_range × max_features) + ComplementNB
  (b) Char n-grams + ComplementNB
  (c) Combined word + char TF-IDF + ComplementNB
  (d) Classifier comparison on best TF-IDF config
  (e) Dense SVD features as structured add-on
  (f) Best config with further refined params

Saves results to artifacts/text_metrics.json and best config to artifacts/text_best_config.json.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

warnings.filterwarnings("ignore")

DATA_DIR = Path("/root/triagegeist_data")
ARTIFACTS_DIR = Path("/root/triagegeist-kaggle/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 3

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    chief = pd.read_csv(DATA_DIR / "chief_complaints.csv")

    train = train.merge(chief[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left")
    test = test.merge(chief[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left")

    # fill missing text
    train["chief_complaint_raw"] = train["chief_complaint_raw"].fillna("")
    test["chief_complaint_raw"] = test["chief_complaint_raw"].fillna("")

    y = train["triage_acuity"].values
    X_text = train["chief_complaint_raw"].values
    X_text_test = test["chief_complaint_raw"].values

    print(f"Train samples: {len(train)}, Test samples: {len(test)}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"Non-empty text: {(X_text != '').sum()} / {len(X_text)}")
    return X_text, y, X_text_test


# ---------------------------------------------------------------------------
# cross-validation helper
# ---------------------------------------------------------------------------
def cv_macro_f1(model, X_text, y, n_splits=N_FOLDS, random_state=RANDOM_STATE):
    """Return mean macro-F1 across StratifiedKFold folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, valid_idx in cv.split(X_text, y):
        fold_model = clone(model)
        fold_model.fit(X_text[train_idx], y[train_idx])
        preds = fold_model.predict(X_text[valid_idx])
        scores.append(f1_score(y[valid_idx], preds, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def run_experiment(name: str, model, X_text, y, extra: dict | None = None) -> dict:
    """Run CV and return result dict."""
    mean_f1, std_f1 = cv_macro_f1(model, X_text, y)
    result = {
        "experiment": name,
        "macro_f1_mean": round(mean_f1, 6),
        "macro_f1_std": round(std_f1, 6),
    }
    if extra:
        result.update(extra)
    print(f"  {name:60s} → MF1 = {mean_f1:.6f} ± {std_f1:.6f}")
    return result


# ---------------------------------------------------------------------------
# Experiment (a): TF-IDF parameter sweep
# ---------------------------------------------------------------------------
def experiment_a(X_text, y):
    print("\n" + "=" * 70)
    print("(a) TF-IDF parameter sweep (word n-grams) + ComplementNB")
    print("=" * 70)

    results = []
    ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 3), (1, 4)]
    max_features_list = [5000, 10000, 25000, 50000, 100000]

    for ngram in ngram_ranges:
        for mf in max_features_list:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram,
                    min_df=5,
                    max_features=mf,
                    sublinear_tf=True,
                )),
                ("clf", ComplementNB(alpha=0.25)),
            ])
            name = f"TF-IDF word ({ngram[0]},{ngram[1]}) max_f={mf}"
            extra = {"ngram_range": str(ngram), "max_features": mf, "analyzer": "word"}
            results.append(run_experiment(name, model, X_text, y, extra))
    return results


# ---------------------------------------------------------------------------
# Experiment (b): Character n-grams
# ---------------------------------------------------------------------------
def experiment_b(X_text, y):
    print("\n" + "=" * 70)
    print("(b) Character n-grams + ComplementNB")
    print("=" * 70)

    results = []
    char_configs = [
        {"ngram_range": (2, 5), "max_features": 25000},
        {"ngram_range": (2, 6), "max_features": 25000},
        {"ngram_range": (3, 7), "max_features": 25000},
        {"ngram_range": (2, 6), "max_features": 50000},
        {"ngram_range": (2, 8), "max_features": 25000},
    ]

    for cfg in char_configs:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="char",
                ngram_range=cfg["ngram_range"],
                min_df=5,
                max_features=cfg["max_features"],
                sublinear_tf=True,
            )),
            ("clf", ComplementNB(alpha=0.25)),
        ])
        nr = cfg["ngram_range"]
        name = f"TF-IDF char ({nr[0]},{nr[1]}) max_f={cfg['max_features']}"
        extra = {
            "ngram_range": str(cfg["ngram_range"]),
            "max_features": cfg["max_features"],
            "analyzer": "char",
        }
        results.append(run_experiment(name, model, X_text, y, extra))
    return results


# ---------------------------------------------------------------------------
# Experiment (c): Combined word + char TF-IDF (stacked features)
# ---------------------------------------------------------------------------
def experiment_c(X_text, y):
    print("\n" + "=" * 70)
    print("(c) Combined word + char TF-IDF features + ComplementNB")
    print("=" * 70)

    results = []

    # We'll pre-transform and stack features, then train ComplementNB on stacked sparse matrix
    from scipy.sparse import hstack

    word_tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        ngram_range=(1, 2), min_df=5, max_features=25000, sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        analyzer="char", ngram_range=(2, 6), min_df=5, max_features=25000, sublinear_tf=True,
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in cv.split(X_text, y):
        # Fit both vectorizers on fold train
        word_train = word_tfidf.fit_transform(X_text[train_idx])
        char_train = char_tfidf.fit_transform(X_text[train_idx])
        X_fold_train = hstack([word_train, char_train])

        word_valid = word_tfidf.transform(X_text[valid_idx])
        char_valid = char_tfidf.transform(X_text[valid_idx])
        X_fold_valid = hstack([word_valid, char_valid])

        clf = ComplementNB(alpha=0.25)
        clf.fit(X_fold_train, y[train_idx])
        preds = clf.predict(X_fold_valid)
        scores.append(f1_score(y[valid_idx], preds, average="macro"))

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))
    print(f"  {'Combined word (1,2)+char (2,6) TF-IDF + CNB':60s} → MF1 = {mean_f1:.6f} ± {std_f1:.6f}")
    results.append({
        "experiment": "Combined word+char TF-IDF + ComplementNB",
        "macro_f1_mean": round(mean_f1, 6),
        "macro_f1_std": round(std_f1, 6),
        "analyzer": "combined_word_char",
    })

    # Also try with LogisticRegression on combined features
    from scipy.sparse import hstack
    word_tfidf2 = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        ngram_range=(1, 2), min_df=5, max_features=25000, sublinear_tf=True,
    )
    char_tfidf2 = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        analyzer="char", ngram_range=(2, 6), min_df=5, max_features=25000, sublinear_tf=True,
    )

    cv2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores2 = []
    for train_idx, valid_idx in cv2.split(X_text, y):
        word_train = word_tfidf2.fit_transform(X_text[train_idx])
        char_train = char_tfidf2.fit_transform(X_text[train_idx])
        X_fold_train = hstack([word_train, char_train])

        word_valid = word_tfidf2.transform(X_text[valid_idx])
        char_valid = char_tfidf2.transform(X_text[valid_idx])
        X_fold_valid = hstack([word_valid, char_valid])

        clf2 = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=RANDOM_STATE)
        clf2.fit(X_fold_train, y[train_idx])
        preds2 = clf2.predict(X_fold_valid)
        scores2.append(f1_score(y[valid_idx], preds2, average="macro"))

    mean_f1_2 = float(np.mean(scores2))
    std_f1_2 = float(np.std(scores2))
    print(f"  {'Combined word (1,2)+char (2,6) TF-IDF + LogReg':60s} → MF1 = {mean_f1_2:.6f} ± {std_f1_2:.6f}")
    results.append({
        "experiment": "Combined word+char TF-IDF + LogisticRegression",
        "macro_f1_mean": round(mean_f1_2, 6),
        "macro_f1_std": round(std_f1_2, 6),
        "analyzer": "combined_word_char_logreg",
    })
    return results


# ---------------------------------------------------------------------------
# Experiment (d): Classifier comparison on best TF-IDF
# ---------------------------------------------------------------------------
def experiment_d(X_text, y, best_tfidf_params: dict):
    print("\n" + "=" * 70)
    print("(d) Classifier comparison on best TF-IDF config")
    print("=" * 70)

    ngram_range = eval(best_tfidf_params.get("ngram_range", "(1, 2)"))
    max_features = best_tfidf_params.get("max_features", 25000)

    tfidf_kwargs = {
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": ngram_range,
        "min_df": 5,
        "max_features": max_features,
        "sublinear_tf": True,
    }

    classifiers = {
        "ComplementNB": ComplementNB(alpha=0.25),
        "ComplementNB(alpha=1.0)": ComplementNB(alpha=1.0),
        "MultinomialNB(alpha=0.1)": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, multi_class="multinomial", random_state=RANDOM_STATE
        ),
        "LinearSVC": LinearSVC(max_iter=2000, dual="auto", random_state=RANDOM_STATE),
        "SGDClassifier": SGDClassifier(
            max_iter=1000, random_state=RANDOM_STATE, loss="log_loss"
        ),
    }

    results = []
    for clf_name, clf in classifiers.items():
        model = Pipeline([("tfidf", TfidfVectorizer(**tfidf_kwargs)), ("clf", clf)])
        extra = {
            "classifier": clf_name,
            "tfidf_ngram": str(ngram_range),
            "tfidf_max_features": max_features,
        }
        results.append(run_experiment(clf_name, model, X_text, y, extra))
    return results


# ---------------------------------------------------------------------------
# Experiment (e): Dense SVD features
# ---------------------------------------------------------------------------
def experiment_e(X_text, y):
    print("\n" + "=" * 70)
    print("(e) Dense SVD features from TF-IDF")
    print("=" * 70)

    results = []

    # SVD with ComplementNB on reduced features
    for n_components in [50, 100, 200, 300]:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=(1, 2), min_df=5, max_features=25000, sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
            ("clf", ComplementNB(alpha=0.25)),
        ])
        name = f"SVD n={n_components} + ComplementNB"
        extra = {"n_components": n_components, "method": "svd_cnb"}
        results.append(run_experiment(name, model, X_text, y, extra))

    # SVD + LogisticRegression
    for n_components in [100, 200]:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=(1, 2), min_df=5, max_features=25000, sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=RANDOM_STATE)),
        ])
        name = f"SVD n={n_components} + LogisticRegression"
        extra = {"n_components": n_components, "method": "svd_logreg"}
        results.append(run_experiment(name, model, X_text, y, extra))

    return results


# ---------------------------------------------------------------------------
# Experiment (f): Best config with refined params
# ---------------------------------------------------------------------------
def experiment_f(X_text, y, all_results):
    print("\n" + "=" * 70)
    print("(f) Best config with refined parameters")
    print("=" * 70)

    # Find the best experiment so far
    best = max(all_results, key=lambda r: r["macro_f1_mean"])
    print(f"Best so far: {best['experiment']} → MF1={best['macro_f1_mean']}")

    results = []

    # Refinement 1: ComplementNB alpha sweep on best TF-IDF
    # Assume it's a word TF-IDF config
    ngram = eval(best.get("ngram_range", "(1, 2)"))
    max_f = best.get("max_features", 25000)
    analyzer = best.get("analyzer", "word")

    for alpha in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=ngram, min_df=5, max_features=max_f, sublinear_tf=True,
            )),
            ("clf", ComplementNB(alpha=alpha)),
        ])
        name = f"Best TF-IDF + ComplementNB(alpha={alpha})"
        extra = {"alpha": alpha, "refinement": "alpha_sweep"}
        results.append(run_experiment(name, model, X_text, y, extra))

    # Refinement 2: Try with min_df sweep
    for min_df in [1, 2, 3, 5, 10]:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=ngram, min_df=min_df, max_features=max_f, sublinear_tf=True,
            )),
            ("clf", ComplementNB(alpha=0.25)),
        ])
        name = f"Best TF-IDF min_df={min_df} + ComplementNB(0.25)"
        extra = {"min_df": min_df, "refinement": "min_df_sweep"}
        results.append(run_experiment(name, model, X_text, y, extra))

    # Refinement 3: Try with/without sublinear_tf
    for sublinear in [True, False]:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=ngram, min_df=5, max_features=max_f, sublinear_tf=sublinear,
            )),
            ("clf", ComplementNB(alpha=0.25)),
        ])
        name = f"Best TF-IDF sublinear_tf={sublinear} + ComplementNB(0.25)"
        extra = {"sublinear_tf": sublinear, "refinement": "sublinear_sweep"}
        results.append(run_experiment(name, model, X_text, y, extra))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TRIAGEGEIST TEXT FEATURE OPTIMIZATION")
    print("=" * 70)

    X_text, y, _ = load_data()

    all_results = []

    # (a) TF-IDF param sweep
    all_results.extend(experiment_a(X_text, y))

    # (b) Char n-grams
    all_results.extend(experiment_b(X_text, y))

    # (c) Combined word + char
    all_results.extend(experiment_c(X_text, y))

    # Identify best TF-IDF config from (a)
    a_results = [r for r in all_results if "ngram_range" in r and r.get("analyzer") == "word"]
    best_a = max(a_results, key=lambda r: r["macro_f1_mean"])
    print(f"\n>>> Best TF-IDF config: {best_a['experiment']} → MF1={best_a['macro_f1_mean']}")

    # (d) Classifier comparison on best TF-IDF
    all_results.extend(experiment_d(X_text, y, best_a))

    # (e) Dense SVD features
    all_results.extend(experiment_e(X_text, y))

    # (f) Best config with refinements
    all_results.extend(experiment_f(X_text, y, all_results))

    # -----------------------------------------------------------------------
    # Final results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — ALL EXPERIMENTS (sorted by Macro-F1)")
    print("=" * 70)

    sorted_results = sorted(all_results, key=lambda r: r["macro_f1_mean"], reverse=True)

    for i, r in enumerate(sorted_results, 1):
        print(f"  {i:2d}. {r['experiment']:65s} MF1={r['macro_f1_mean']:.6f}")

    best_overall = sorted_results[0]
    print("\n" + "-" * 70)
    print(f"BEST: {best_overall['experiment']}")
    print(f"  Macro-F1 = {best_overall['macro_f1_mean']:.6f} ± {best_overall['macro_f1_std']:.6f}")

    # Baseline: TF-IDF (1,2) max_f=25000 + ComplementNB(alpha=0.25) — the reference
    baseline_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=5, max_features=25000, sublinear_tf=True,
        )),
        ("clf", ComplementNB(alpha=0.25)),
    ])
    baseline_mean, baseline_std = cv_macro_f1(baseline_model, X_text, y)
    print(f"\nBaseline (TF-IDF 1-2, 25K, CNB alpha=0.25): MF1={baseline_mean:.6f} ± {baseline_std:.6f}")
    improvement = best_overall["macro_f1_mean"] - baseline_mean
    print(f"Improvement over baseline: +{improvement:.6f}")

    # Save all results
    metrics = {
        "baseline_config": {
            "vectorizer": "TfidfVectorizer(ngram_range=(1,2), max_features=25000, min_df=5, sublinear_tf=True)",
            "classifier": "ComplementNB(alpha=0.25)",
            "macro_f1_mean": round(baseline_mean, 6),
            "macro_f1_std": round(baseline_std, 6),
        },
        "best_config": {
            "experiment": best_overall["experiment"],
            "macro_f1_mean": best_overall["macro_f1_mean"],
            "macro_f1_std": best_overall["macro_f1_std"],
            "improvement_over_baseline": round(improvement, 6),
        },
        "all_experiments": sorted_results,
    }

    with open(ARTIFACTS_DIR / "text_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {ARTIFACTS_DIR / 'text_metrics.json'}")

    # Save best config in a simple, merge-friendly format
    best_config = {
        "vectorizer": "TfidfVectorizer",
        "ngram_range": best_overall.get("ngram_range", "(1, 2)"),
        "max_features": best_overall.get("max_features", 25000),
        "min_df": best_overall.get("min_df", 5),
        "analyzer": best_overall.get("analyzer", "word"),
        "sublinear_tf": best_overall.get("sublinear_tf", True),
        "classifier": best_overall.get("classifier", "ComplementNB"),
        "alpha": best_overall.get("alpha", 0.25),
        "macro_f1_mean": best_overall["macro_f1_mean"],
        "macro_f1_std": best_overall["macro_f1_std"],
        "improvement_over_baseline": round(improvement, 6),
        "baseline_macro_f1": round(baseline_mean, 6),
    }

    # If it's from combined or SVD, adjust
    if "combined_word_char" in best_overall.get("analyzer", ""):
        best_config["vectorizer"] = "Combined word+char TfidfVectorizer"
    if "svd" in best_overall.get("method", ""):
        best_config["vectorizer"] = "TfidfVectorizer + TruncatedSVD"
        best_config["n_components"] = best_overall.get("n_components", 100)

    with open(ARTIFACTS_DIR / "text_best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best config to {ARTIFACTS_DIR / 'text_best_config.json'}")

    print("\n" + "=" * 70)
    print(f"TEXT OPTIMIZATION: Best config [{best_overall['experiment']}], "
          f"MF1={best_overall['macro_f1_mean']:.4f}, "
          f"Improvement over baseline TF-IDF=+{improvement:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
