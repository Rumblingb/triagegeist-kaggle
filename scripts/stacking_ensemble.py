#!/usr/bin/env python3
"""
Triagegeist Final Ensemble: Weighted Blending + Stacking
Tests multiple blend weights, a stacking meta-learner, and selects the best.
"""

import os, sys, json, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

DATA_DIR = Path("/root/triagegeist_data")
ARTIFACTS_DIR = Path("/root/triagegeist-kaggle/artifacts")
SCRIPTS_DIR = Path("/root/triagegeist-kaggle/scripts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR / "models", exist_ok=True)

TARGET_COL = "triage_acuity"
TEXT_COL = "chief_complaint_raw"
LEAK_COLS = ["patient_id", "disposition", "ed_los_hours", TARGET_COL]
ID_COL = "patient_id"
N_FOLDS = 3
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ── 1. Load & Merge ─────────────────────────────────────────────────
print("=" * 60)
print("TRIAGEGEIST FINAL ENSEMBLE")
print("=" * 60)

print("\n1. Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
chief = pd.read_csv(DATA_DIR / "chief_complaints.csv")
history = pd.read_csv(DATA_DIR / "patient_history.csv")

train = train.merge(chief, on="patient_id", how="left").merge(history, on="patient_id", how="left")
test = test.merge(chief, on="patient_id", how="left").merge(history, on="patient_id", how="left")
print(f"  Train: {train.shape}, Test: {test.shape}")

y = train[TARGET_COL].values
classes = np.sort(y)

# ── 2. Feature Engineering ──────────────────────────────────────────
print("\n2. Adding engineered features...")

def add_engineered_features(df):
    df = df.copy()
    complaint = df[TEXT_COL].fillna("").str.lower()

    # Pain flags
    df["pain_unrecorded"] = (df["pain_score"] == -1).astype(int)
    df["pain_score_clean"] = df["pain_score"].replace(-1, np.nan)

    # Vital sign flags
    df["flag_low_oxygen"] = (df["spo2"] < 92).astype(int)
    df["flag_fever"] = (df["temperature_c"] >= 38.0).astype(int)
    df["flag_tachycardia"] = (df["heart_rate"] >= 100).astype(int)
    df["flag_tachypnea"] = (df["respiratory_rate"] >= 22).astype(int)
    df["flag_hypotension"] = (df["systolic_bp"] < 90).astype(int)
    df["flag_gcs_abnormal"] = (df["gcs_total"] < 15).astype(int)
    df["flag_high_news2"] = (df["news2_score"] >= 5).astype(int)
    df["flag_high_shock_index"] = (df["shock_index"] >= 0.9).astype(int)

    # Complaint text stats
    df["complaint_len"] = complaint.str.len()
    df["complaint_word_count"] = complaint.str.split().str.len()
    df["complaint_has_comma"] = complaint.str.contains(",", regex=False).astype(int)

    # Keyword flags
    kw_patterns = {
        "kw_chest_pain": r"chest pain|thoracic pain|crushing chest",
        "kw_stroke": r"stroke|seizure|thunderclap|loss of vision|weakness|aphasia",
        "kw_respiratory": r"shortness of breath|asthma|hypoxia|wheeze|near-drowning",
        "kw_trauma": r"trauma|fracture|haemothorax|stab|wound|fall|injury",
        "kw_overdose": r"overdose|poison|toxic|substance",
        "kw_bleeding": r"bleed|haemorrhage|hemorrhage|melena|hematemesis|haematemesis",
        "kw_pregnancy": r"pregnan|ectopic|postpartum|miscarriage",
        "kw_infection": r"sepsis|fever|necrotising|infection|cellulitis",
    }
    for name, pat in kw_patterns.items():
        df[name] = complaint.str.contains(pat, flags=re.IGNORECASE, regex=True).astype(int)

    # Comorbidity composites
    cardio = ["hx_hypertension", "hx_heart_failure", "hx_atrial_fibrillation",
              "hx_coronary_artery_disease", "hx_peripheral_vascular_disease", "hx_stroke_prior"]
    resp = ["hx_asthma", "hx_copd"]
    neuro = ["hx_dementia", "hx_epilepsy", "hx_stroke_prior"]
    frailty = ["hx_dementia", "hx_ckd", "hx_malignancy", "hx_immunosuppressed"]

    df["cardio_burden"] = df[[c for c in cardio if c in df.columns]].sum(axis=1)
    df["respiratory_burden"] = df[[c for c in resp if c in df.columns]].sum(axis=1)
    df["neuro_burden"] = df[[c for c in neuro if c in df.columns]].sum(axis=1)
    df["frailty_burden"] = df[[c for c in frailty if c in df.columns]].sum(axis=1)

    return df

train_fe = add_engineered_features(train)
test_fe = add_engineered_features(test)

# ── 3. Feature Columns ──────────────────────────────────────────────
feature_cols = [c for c in train_fe.columns if c not in LEAK_COLS]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_fe[c]) and c != TEXT_COL]
categorical_cols = [c for c in feature_cols if c not in numeric_cols and c != TEXT_COL]

print(f"  Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

# Separate text
X_train_text = train_fe[TEXT_COL].fillna("")
X_test_text = test_fe[TEXT_COL].fillna("")

# Build structured data
def build_structured_data(df, numeric_cols, categorical_cols):
    X = df[numeric_cols + categorical_cols].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X

X_train_str = build_structured_data(train_fe, numeric_cols, categorical_cols)
X_test_str = build_structured_data(test_fe, numeric_cols, categorical_cols)

print(f"  X_train_str: {X_train_str.shape}, X_test_str: {X_test_str.shape}")

# ── Preprocessing Pipelines ─────────────────────────────────────────
numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# Text pipeline
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        ngram_range=(1, 2), min_df=5, max_features=30000, sublinear_tf=True,
    )),
    ("cnb", ComplementNB(alpha=0.3)),
])

# ── 4. Baseline: HGB with 3-fold CV ─────────────────────────────────
print("\n3. Baseline: HistGradientBoosting (3-fold CV)...")

hgb = HistGradientBoostingClassifier(
    max_depth=7, learning_rate=0.05, max_iter=250,
    min_samples_leaf=50, random_state=RANDOM_SEED,
)

hgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", hgb),
])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
hgb_cv = cross_val_score(hgb_pipe, X_train_str, y, cv=skf, scoring="f1_macro")
hgb_baseline = hgb_cv.mean()
print(f"  HGB Macro-F1: {hgb_baseline:.4f} ± {hgb_cv.std():.4f}")

# ── 5. APPROACH A: Weighted Blending ────────────────────────────────
print("\n4. APPROACH A: Weighted Blending...")

# We'll do 3-fold CV for each model, collecting OOF probabilities
# Then blend with various weights and measure CV performance

# 5a. Cross-val fit for HGB
print("  Fitting HGB (3-fold CV)...")
oof_hgb = np.zeros((len(X_train_str), len(classes)))
test_hgb = np.zeros((len(X_test_str), len(classes)))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_str, y)):
    print(f"    Fold {fold+1}/{N_FOLDS}...")
    X_tr, X_va = X_train_str.iloc[tr_idx], X_train_str.iloc[va_idx]
    y_tr = y[tr_idx]

    pipe = clone(hgb_pipe)
    pipe.fit(X_tr, y_tr)
    oof_hgb[va_idx] = pipe.predict_proba(X_va)
    test_hgb += pipe.predict_proba(X_test_str) / N_FOLDS

# 5b. Cross-val fit for RF
print("  Fitting RF (3-fold CV)...")
rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf),
])

oof_rf = np.zeros((len(X_train_str), len(classes)))
test_rf = np.zeros((len(X_test_str), len(classes)))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_str, y)):
    print(f"    Fold {fold+1}/{N_FOLDS}...")
    X_tr, X_va = X_train_str.iloc[tr_idx], X_train_str.iloc[va_idx]
    y_tr = y[tr_idx]

    pipe = clone(rf_pipe)
    pipe.fit(X_tr, y_tr)
    oof_rf[va_idx] = pipe.predict_proba(X_va)
    test_rf += pipe.predict_proba(X_test_str) / N_FOLDS

# 5c. Cross-val fit for ComplementNB (text-only)
print("  Fitting ComplementNB (text, 3-fold CV)...")
oof_cnb = np.zeros((len(X_train_str), len(classes)))
test_cnb = np.zeros((len(X_test_str), len(classes)))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_str, y)):
    print(f"    Fold {fold+1}/{N_FOLDS}...")
    X_tr_text, X_va_text = X_train_text.iloc[tr_idx], X_train_text.iloc[va_idx]
    y_tr = y[tr_idx]

    pipe = clone(text_pipeline)
    pipe.fit(X_tr_text, y_tr)
    oof_cnb[va_idx] = pipe.predict_proba(X_va_text)
    test_cnb += pipe.predict_proba(X_test_text) / N_FOLDS

# 5d. Try different blend weights
blend_weights = [
    ("HGB(0.5)+RF(0.25)+CNB(0.25)", 0.50, 0.25, 0.25),
    ("HGB(0.6)+RF(0.2)+CNB(0.2)",  0.60, 0.20, 0.20),
    ("HGB(0.4)+RF(0.4)+CNB(0.2)",  0.40, 0.40, 0.20),
    ("HGB(0.7)+RF(0.15)+CNB(0.15)", 0.70, 0.15, 0.15),
    ("HGB(0.5)+RF(0.3)+CNB(0.2)",   0.50, 0.30, 0.20),
]

best_blend_name = None
best_blend_score = -1
best_blend_weights = None
best_blend_oof = None
best_blend_test = None

print("\n  Weight Blend Results:")
for name, w_hgb, w_rf, w_cnb in blend_weights:
    blend_oof = w_hgb * oof_hgb + w_rf * oof_rf + w_cnb * oof_cnb
    blend_pred = classes[np.argmax(blend_oof, axis=1)]
    mf1 = f1_score(y, blend_pred, average="macro")

    # High risk recall
    hr_mask = y <= 2
    hr_recall = recall_score(y[hr_mask], blend_pred[hr_mask], average=None).mean()

    # Undertriage
    ut_rate = np.mean((blend_pred - y) >= 2)

    print(f"    {name:40s} → MF1={mf1:.4f}, HR-Recall={hr_recall:.4f}, Undertriage={ut_rate:.4f}")

    if mf1 > best_blend_score:
        best_blend_score = mf1
        best_blend_name = name
        best_blend_weights = (w_hgb, w_rf, w_cnb)
        best_blend_oof = blend_oof
        best_blend_test = w_hgb * test_hgb + w_rf * test_rf + w_cnb * test_cnb

print(f"\n  BEST BLEND: {best_blend_name} → MF1={best_blend_score:.4f}")

# ── 6. APPROACH B: Stacking with Meta-Learner ───────────────────────
print("\n5. APPROACH B: Stacking (Logistic Regression meta-learner)...")

# Check if other agents' models exist
other_model_files = list((ARTIFACTS_DIR / "models").glob("*.joblib"))
other_results = list(ARTIFACTS_DIR.glob("*.json"))

print(f"  Found model artifacts: {[f.name for f in other_model_files]}")
print(f"  Found result JSONs: {[f.name for f in other_results]}")

# Build stacking features from our 3 models (HGB, RF, CNB)
# Stacking: Use OOF probabilities as meta-features
stack_oof = np.column_stack([oof_hgb, oof_rf, oof_cnb])  # (n_train, 3*n_classes)
stack_test = np.column_stack([test_hgb, test_rf, test_cnb])  # (n_test, 3*n_classes)

print(f"  Stacking meta-features shape: {stack_oof.shape}")

# Meta-learner: LogisticRegression with CV
print("  Training meta-learner (LogisticRegression, 3-fold CV on stacking features)...")
meta_lr = LogisticRegression(
    multi_class="multinomial", solver="lbfgs",
    max_iter=1000, C=1.0, random_state=RANDOM_SEED
)

skf_stack = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
stack_cv_scores = cross_val_score(meta_lr, stack_oof, y, cv=skf_stack, scoring="f1_macro")
stack_cv_mean = stack_cv_scores.mean()
print(f"  Stacking (LR meta) CV Macro-F1: {stack_cv_mean:.4f} ± {stack_cv_scores.std():.4f}")

# Full fit on all training data
meta_lr.fit(stack_oof, y)
stack_full_pred = meta_lr.predict(stack_oof)
stack_full_mf1 = f1_score(y, stack_full_pred, average="macro")
print(f"  Stacking full-train Macro-F1: {stack_full_mf1:.4f}")

# Also try RF as meta-learner
print("  Training meta-learner (RF, 3-fold CV on stacking features)...")
meta_rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
meta_rf_cv = cross_val_score(meta_rf, stack_oof, y, cv=skf_stack, scoring="f1_macro")
meta_rf_cv_mean = meta_rf_cv.mean()
print(f"  Stacking (RF meta) CV Macro-F1: {meta_rf_cv_mean:.4f} ± {meta_rf_cv.std():.4f}")

# Select best stacking
stack_use_lr = stack_cv_mean >= meta_rf_cv_mean
best_stack_name = "LR" if stack_use_lr else "RF"
best_stack_score = max(stack_cv_mean, meta_rf_cv_mean)
print(f"  BEST STACKING: {best_stack_name} → MF1={best_stack_score:.4f}")

if stack_use_lr:
    meta_rf.fit(stack_oof, y)  # train RF anyway for comparison
    best_meta_clf = meta_lr
    test_stack_preds = best_meta_clf.predict(stack_test)
else:
    meta_rf.fit(stack_oof, y)
    best_meta_clf = meta_rf
    test_stack_preds = best_meta_clf.predict(stack_test)

# ── 7. Choose best approach ─────────────────────────────────────────
print("\n6. SELECTING BEST ENSEMBLE...")

# Compare blend vs stacking
if best_blend_score >= best_stack_score:
    print(f"  ✓ WEIGHTED BLENDING wins: {best_blend_name} (MF1={best_blend_score:.4f})")
    print(f"    vs Stacking {best_stack_name} (MF1={best_stack_score:.4f})")
    winning_method = "weighted_blend"
    winning_name = best_blend_name
    winning_weights = best_blend_weights
    winning_score = best_blend_score
    winning_test_preds = classes[np.argmax(best_blend_test, axis=1)]
else:
    print(f"  ✓ STACKING wins: {best_stack_name} meta-learner (MF1={best_stack_score:.4f})")
    print(f"    vs Best Blend {best_blend_name} (MF1={best_blend_score:.4f})")
    winning_method = "stacking"
    winning_name = f"Stacking-{best_stack_name}"
    winning_weights = None
    winning_score = best_stack_score
    winning_test_preds = test_stack_preds

# ── 8. Final Metrics on OOF ─────────────────────────────────────────
print("\n7. FINAL METRICS...")

# For final reporting, use the best approach's OOF predictions
if winning_method == "weighted_blend":
    final_oof_pred = classes[np.argmax(best_blend_oof, axis=1)]
else:
    final_oof_pred = best_meta_clf.predict(stack_oof)

final_mf1 = f1_score(y, final_oof_pred, average="macro")

# Per-class F1
per_class = f1_score(y, final_oof_pred, average=None)
print(f"  Macro-F1: {final_mf1:.4f}")
print(f"  Per-class F1: {dict(zip(classes, [f'{v:.4f}' for v in per_class]))}")

# High-risk recall
hr_mask = y <= 2
hr_recall = recall_score(y[hr_mask], final_oof_pred[hr_mask], average=None).mean()
print(f"  High-risk (ESI 1-2) recall: {hr_recall:.4f}")

# Undertriage
ut_rate = np.mean((final_oof_pred - y) >= 2)
print(f"  Severe undertriage rate: {ut_rate:.4f}")

# ── 9. Final Submission ─────────────────────────────────────────────
print("\n8. GENERATING FINAL SUBMISSION...")

# Train best model on FULL data for final submission
print(f"  Training final {winning_name} on ALL training data...")

if winning_method == "weighted_blend":
    # Train each model on full data
    w_hgb, w_rf, w_cnb = winning_weights

    print("    Training full HGB...")
    full_hgb = clone(hgb_pipe)
    full_hgb.fit(X_train_str, y)

    print("    Training full RF...")
    full_rf = clone(rf_pipe)
    full_rf.fit(X_train_str, y)

    print("    Training full CNB...")
    full_cnb = clone(text_pipeline)
    full_cnb.fit(X_train_text, y)

    # Predict
    p_hgb = full_hgb.predict_proba(X_test_str)
    p_rf = full_rf.predict_proba(X_test_str)
    p_cnb = full_cnb.predict_proba(X_test_text)

    blend_test = w_hgb * p_hgb + w_rf * p_rf + w_cnb * p_cnb
    final_preds = classes[np.argmax(blend_test, axis=1)]
else:
    # Train stacking on full data
    print("    Training full HGB on all data...")
    full_hgb_s = clone(hgb_pipe)
    full_hgb_s.fit(X_train_str, y)
    p_hgb_full = full_hgb_s.predict_proba(X_test_str)

    print("    Training full RF on all data...")
    full_rf_s = clone(rf_pipe)
    full_rf_s.fit(X_train_str, y)
    p_rf_full = full_rf_s.predict_proba(X_test_str)

    print("    Training full CNB on all data...")
    full_cnb_s = clone(text_pipeline)
    full_cnb_s.fit(X_train_text, y)
    p_cnb_full = full_cnb_s.predict_proba(X_test_text)

    # Get OOF predictions for meta training (already have from CV)
    # Fit meta on all OOF data
    print("    Fitting stacking meta-learner on full OOF data...")
    full_stack_train = np.column_stack([
        oof_hgb, oof_rf, oof_cnb
    ])
    full_stack_test = np.column_stack([
        p_hgb_full, p_rf_full, p_cnb_full
    ])
    if stack_use_lr:
        best_meta_clf.fit(full_stack_train, y)
    else:
        best_meta_clf.fit(full_stack_train, y)
    final_preds = best_meta_clf.predict(full_stack_test)

# Save submission
submission = pd.DataFrame({
    "patient_id": test["patient_id"].values,
    TARGET_COL: final_preds,
})
submission.to_csv(ARTIFACTS_DIR / "final_submission.csv", index=False)
print(f"  ✓ Saved: {ARTIFACTS_DIR / 'final_submission.csv'}")
print(f"  Predictions distribution: {pd.Series(final_preds).value_counts().sort_index().to_dict()}")

# ── 10. Save Metrics ────────────────────────────────────────────────
metrics = {
    "ensemble_method": winning_method,
    "ensemble_name": winning_name,
    "blend_weights": list(winning_weights) if winning_weights else None,
    "baseline_hgb_mf1": round(hgb_baseline, 4),
    "best_blend_mf1": round(best_blend_score, 4),
    "best_blend_name": best_blend_name,
    "best_stack_mf1": round(best_stack_score, 4),
    "best_stack_name": best_stack_name,
    "final_macro_f1": round(final_mf1, 4),
    "high_risk_recall": round(float(hr_recall), 4),
    "undertriage_rate": round(float(ut_rate), 4),
    "per_class_f1": {int(k): round(float(v), 4) for k, v in zip(classes, per_class)},
    "all_blends": {},
    "stacking_scores": {
        "lr_meta_cv": round(float(stack_cv_mean), 4),
        "rf_meta_cv": round(float(meta_rf_cv_mean), 4),
        "chosen": best_stack_name,
    },
}

# Record all blend results
for name, w_hgb, w_rf, w_cnb in blend_weights:
    blend_oof = w_hgb * oof_hgb + w_rf * oof_rf + w_cnb * oof_cnb
    blend_pred = classes[np.argmax(blend_oof, axis=1)]
    mf1 = f1_score(y, blend_pred, average="macro")
    metrics["all_blends"][name] = round(float(mf1), 4)

with open(ARTIFACTS_DIR / "ensemble_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ Saved: {ARTIFACTS_DIR / 'ensemble_metrics.json'}")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ENSEMBLE SUMMARY")
print("=" * 60)
if winning_weights:
    print(f"Best weights [{winning_weights[0]:.2f},{winning_weights[1]:.2f},{winning_weights[2]:.2f}]")
print(f"MF1={winning_score:.4f}")
print(f"Stacking MF1={best_stack_score:.4f}")
print(f"Final submission at final_submission.csv")
print("=" * 60)

print(f"\nENSEMBLE: Best weights {list(winning_weights) if winning_weights else 'N/A'}=MF1 {best_blend_score:.4f}, "
      f"Stacking MF1={best_stack_score:.4f}, Final submission at submission.csv")
