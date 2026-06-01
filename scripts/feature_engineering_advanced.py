#!/usr/bin/env python3
"""
Triagegeist Advanced Feature Engineering
Compares HGB model with original features vs original + new engineered features.
Reports CV Macro-F1 improvement and feature importances.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/triagegeist_data'
ARTIFACTS_DIR = '/root/triagegeist-kaggle/artifacts'
RANDOM_STATE = 42

print("=" * 70)
print("TRIAGEGEIST — ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────
print("\n[1] Loading data...")
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
chief = pd.read_csv(f'{DATA_DIR}/chief_complaints.csv')
history = pd.read_csv(f'{DATA_DIR}/patient_history.csv')

print(f"  Train: {train.shape}, Test: {test.shape}")
print(f"  Chief: {chief.shape}, History: {history.shape}")

# ── 2. MERGE ──────────────────────────────────────────────────────────
print("\n[2] Merging data...")
train = train.merge(chief, on='patient_id', how='left')
train = train.merge(history, on='patient_id', how='left')
test = test.merge(chief, on='patient_id', how='left')
test = test.merge(history, on='patient_id', how='left')
print(f"  Train merged: {train.shape}, Test merged: {test.shape}")

# ── 3. COLUMN LISTS ───────────────────────────────────────────────────
target = 'triage_acuity'
id_cols = ['patient_id']
leak_cols = ['disposition', 'ed_los_hours']
text_cols = ['chief_complaint_raw', 'chief_complaint_system']

# Original numeric features (as used in baseline)
orig_numeric = [
    'arrival_hour', 'arrival_month', 'age',
    'num_prior_ed_visits_12m', 'num_prior_admissions_12m',
    'num_active_medications', 'num_comorbidities',
    'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure', 'pulse_pressure',
    'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2',
    'gcs_total', 'pain_score', 'weight_kg', 'height_cm', 'bmi',
    'shock_index', 'news2_score',
]

# Categorical columns (as used in baseline)
cat_cols = [
    'site_id', 'triage_nurse_id', 'arrival_mode', 'arrival_day',
    'arrival_season', 'shift', 'age_group', 'sex', 'language',
    'insurance_type', 'transport_origin', 'pain_location',
    'mental_status_triage', 'chief_complaint_system',
]

# hx_* columns
hx_cols = [c for c in history.columns if c.startswith('hx_') and c != 'patient_id']
print(f"  History features: {len(hx_cols)} ({hx_cols[0]}...{hx_cols[-1]})")

# ── 4. BUILD ORIGINAL FEATURES ────────────────────────────────────────
print("\n[3] Building feature matrices...")

def build_orig_features(df):
    """Build original numeric + categorical feature matrix (no text)."""
    X = pd.DataFrame(index=df.index)
    for c in orig_numeric:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return X

X_train_orig = build_orig_features(train)
X_test_orig = build_orig_features(test)

# Encode categoricals
for c in cat_cols:
    if c not in train.columns:
        continue
    le = LabelEncoder()
    X_train_orig[c] = le.fit_transform(train[c].astype(str).fillna('missing'))
    # Handle unseen categories in test
    known = list(le.classes_)
    test_vals = test[c].astype(str).fillna('missing').values
    X_test_orig[c] = [le.transform([v])[0] if v in known else -1 for v in test_vals]

print(f"  Original features shape: {X_train_orig.shape}")

# ── 5. ENGINEER NEW FEATURES ──────────────────────────────────────────
print("\n[4] Engineering advanced features...")

def engineer_advanced_features(df):
    """Create all new features on a merged dataframe."""
    X = pd.DataFrame(index=df.index)

    # ── 5a. VITAL RATIOS ──
    sbp = pd.to_numeric(df['systolic_bp'], errors='coerce').fillna(120)
    dbp = pd.to_numeric(df['diastolic_bp'], errors='coerce').fillna(80)
    pp = pd.to_numeric(df['pulse_pressure'], errors='coerce').fillna(40)
    map_val = pd.to_numeric(df['mean_arterial_pressure'], errors='coerce').fillna(93)
    hr = pd.to_numeric(df['heart_rate'], errors='coerce').fillna(75)
    rr = pd.to_numeric(df['respiratory_rate'], errors='coerce').fillna(18)
    spo2 = pd.to_numeric(df['spo2'], errors='coerce').fillna(97)
    bmi = pd.to_numeric(df['bmi'], errors='coerce').fillna(25)

    X['pulse_pressure_ratio'] = np.where(sbp > 0, pp / sbp, 0)
    X['map_bp_ratio'] = np.where(sbp > 0, map_val / sbp, 0)
    X['spo2_fraction'] = spo2 / 100.0
    X['hr_rr_ratio'] = np.where(rr > 0, hr / rr, 0)
    X['bmi_oxygen'] = bmi * (spo2 / 100.0)

    # ── 5b. NEWS2 DECOMPOSITION ──
    temp_c = pd.to_numeric(df['temperature_c'], errors='coerce').fillna(37)
    gcs = pd.to_numeric(df['gcs_total'], errors='coerce').fillna(15)

    X['news2_resp'] = (rr > 20).astype(int)
    X['news2_oxygen'] = (spo2 < 95).astype(int)
    X['news2_circulation'] = (sbp < 100).astype(int)
    X['news2_consciousness'] = (gcs < 15).astype(int)
    X['news2_temp'] = (temp_c > 38).astype(int)
    X['news2_decomposed_sum'] = (X['news2_resp'] + X['news2_oxygen'] +
                                  X['news2_circulation'] + X['news2_consciousness'] +
                                  X['news2_temp'])

    # ── 5c. AGE INTERACTIONS ──
    age = pd.to_numeric(df['age'], errors='coerce').fillna(50)
    si = pd.to_numeric(df['shock_index'], errors='coerce').fillna(0.7)

    X['age_hr'] = age * hr
    X['age_rr'] = age * rr
    X['age_sbp'] = age * sbp
    X['age_shock_index'] = age * si

    # ── 5d. COMORBIDITY COMBOS ──
    hx_data = {c: pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
               for c in hx_cols if c in df.columns}

    # cardiorespiratory = hx_heart_failure + hx_copd
    X['cardiorespiratory'] = hx_data.get('hx_heart_failure', 0) + hx_data.get('hx_copd', 0)

    # cardio_renal = hx_heart_failure + hx_ckd
    X['cardio_renal'] = hx_data.get('hx_heart_failure', 0) + hx_data.get('hx_ckd', 0)

    # diabetes_cardio = (hx_diabetes_type1|hx_diabetes_type2) + hx_coronary_artery_disease
    diabetes = (hx_data.get('hx_diabetes_type1', 0) | hx_data.get('hx_diabetes_type2', 0))
    X['diabetes_cardio'] = diabetes + hx_data.get('hx_coronary_artery_disease', 0)

    # total_comorbidity_burden = sum of all hx_* columns
    X['total_comorbidity_burden'] = sum(hx_data.values())

    # ── 5e. TEMPORAL FEATURES ──
    hour = pd.to_numeric(df['arrival_hour'], errors='coerce').fillna(12)
    month = pd.to_numeric(df['arrival_month'], errors='coerce').fillna(6)

    X['arrival_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    X['arrival_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    X['arrival_month_sin'] = np.sin(2 * np.pi * month / 12)
    X['arrival_month_cos'] = np.cos(2 * np.pi * month / 12)

    # ── 5f. KEYWORD COUNTS from chief_complaint_raw ──
    cc_raw = df['chief_complaint_raw'].fillna('').astype(str).str.lower()

    keyword_patterns = {
        'chest_pain': ['chest pain', 'chest discomfort', 'angina', 'cp'],
        'stroke': ['stroke', 'facial droop', 'slurred speech', 'hemiparesis',
                   'cva', 'tia', 'transient ischaemic attack'],
        'trauma': ['trauma', 'mva', 'motor vehicle', 'fall', 'injury',
                   'fracture', 'laceration', 'burn'],
        'sepsis': ['sepsis', 'septic', 'infection', 'fever', 'rigors', 'chills'],
        'overdose': ['overdose', 'poisoning', 'intoxication', 'toxin'],
        'bleeding': ['bleeding', 'hemorrhage', 'haemorrhage', 'blood loss',
                     'gi bleed', 'melena', 'hematemesis'],
        'pregnancy': ['pregnant', 'pregnancy', 'labour', 'labor', 'obstetric',
                      'contraction', 'antenatal'],
    }

    for kw, patterns in keyword_patterns.items():
        X[f'kw_{kw}'] = cc_raw.apply(lambda x: sum(1 for p in patterns if p in x))

    # ── 5g. TEXT META ──
    X['complaint_char_len'] = cc_raw.apply(len)
    X['complaint_word_count'] = cc_raw.apply(lambda x: len(x.split()))
    X['complaint_has_comma'] = cc_raw.apply(lambda x: 1 if ',' in x else 0)

    print(f"  New features engineered: {len(X.columns)} columns")
    for col in X.columns:
        print(f"    {col}: mean={X[col].mean():.4f}, std={X[col].std():.4f}, nulls={X[col].isna().sum()}")

    return X

X_train_new = engineer_advanced_features(train)
X_test_new = engineer_advanced_features(test)

# Combine original + new
X_train_combined = pd.concat([X_train_orig, X_train_new], axis=1)
X_test_combined = pd.concat([X_test_orig, X_test_new], axis=1)

print(f"\n  Combined features shape: {X_train_combined.shape}")
print(f"  New features added: {X_train_new.shape[1]}")

# ── 6. SETUP HGB MODEL ────────────────────────────────────────────────
print("\n[5] Training HGB with 3-fold stratified CV...")

hgb_params = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'max_iter': 220,
    'min_samples_leaf': 50,
    'random_state': RANDOM_STATE,
}

y = train[target].values.astype(int)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# ── 7. CV: ORIGINAL FEATURES ──────────────────────────────────────────
print("\n  --- Model A: Original features only ---")
hgb_orig = HistGradientBoostingClassifier(**hgb_params)
cv_preds_orig = cross_val_predict(hgb_orig, X_train_orig, y, cv=skf, method='predict', n_jobs=-1)
orig_mf1 = f1_score(y, cv_preds_orig, average='macro')
print(f"  CV Macro-F1 (original): {orig_mf1:.4f}")

# ── 8. CV: ORIGINAL + NEW FEATURES ────────────────────────────────────
print("\n  --- Model B: Original + New features ---")
hgb_new = HistGradientBoostingClassifier(**hgb_params)
cv_preds_new = cross_val_predict(hgb_new, X_train_combined, y, cv=skf, method='predict', n_jobs=-1)
new_mf1 = f1_score(y, cv_preds_new, average='macro')
print(f"  CV Macro-F1 (original+new): {new_mf1:.4f}")

improvement = new_mf1 - orig_mf1
print(f"\n  Improvement: {improvement:+.4f}")

# ── 9. FEATURE IMPORTANCE ─────────────────────────────────────────────
print("\n[6] Feature importance analysis...")
hgb_new.fit(X_train_combined, y)
importances = hgb_new.feature_importances_
feature_names = X_train_combined.columns.tolist()

# Sort by importance
sorted_idx = np.argsort(importances)[::-1]
print(f"\n  Top 30 feature importances:")
for i, idx in enumerate(sorted_idx[:30]):
    is_new = "★" if feature_names[idx] in X_train_new.columns else " "
    print(f"    {i+1:2d}. [{is_new}] {feature_names[idx]:40s} {importances[idx]:.6f}")

# Separate original vs new importances
orig_feat_names = set(X_train_orig.columns)
new_feat_names = set(X_train_new.columns)

orig_imp = sum(importances[i] for i, f in enumerate(feature_names) if f in orig_feat_names)
new_imp = sum(importances[i] for i, f in enumerate(feature_names) if f in new_feat_names)

print(f"\n  Cumulative importance:")
print(f"    Original features ({len(orig_feat_names)}): {orig_imp:.4f} ({orig_imp*100:.1f}%)")
print(f"    New features ({len(new_feat_names)}):      {new_imp:.4f} ({new_imp*100:.1f}%)")

# Top new features
new_importances = [(f, importances[i]) for i, f in enumerate(feature_names) if f in new_feat_names]
new_importances.sort(key=lambda x: -x[1])
print(f"\n  Top 10 new features:")
for f, imp in new_importances[:10]:
    print(f"    {f:40s} {imp:.6f}")

# ── 10. SAVE RESULTS ──────────────────────────────────────────────────
print("\n[7] Saving results...")
results = {
    'original_macro_f1': round(orig_mf1, 6),
    'new_macro_f1': round(new_mf1, 6),
    'improvement': round(improvement, 6),
    'original_feature_count': len(orig_feat_names),
    'new_feature_count': len(new_feat_names),
    'combined_feature_count': len(feature_names),
    'best_params': hgb_params,
    'top_10_new_features': [{'feature': f, 'importance': round(imp, 6)}
                            for f, imp in new_importances[:10]],
    'original_importance_fraction': round(float(orig_imp), 6),
    'new_importance_fraction': round(float(new_imp), 6),
}

with open(f'{ARTIFACTS_DIR}/feature_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to {ARTIFACTS_DIR}/feature_metrics.json")

# ── 11. FINAL REPORT ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"FEATURE ENGINEERING: Original MF1={orig_mf1:.4f}, "
      f"New MF1={new_mf1:.4f}, Improvement={improvement:+.4f}")
print("=" * 70)

# Per-class F1
from sklearn.metrics import f1_score as f1s
print("\n  Per-class F1 (Original):")
for c in sorted(np.unique(y)):
    print(f"    Class {c}: {f1s(y[cv_preds_orig == c], y[cv_preds_orig == c], average='macro'):.4f}")

print("\n  Per-class F1 (New):")
for c in sorted(np.unique(y)):
    print(f"    Class {c}: {f1s(y[cv_preds_new == c], y[cv_preds_new == c], average='macro'):.4f}")

# Confusion matrices
from sklearn.metrics import confusion_matrix
print(f"\n  Confusion Matrix (Original):\n{confusion_matrix(y, cv_preds_orig)}")
print(f"\n  Confusion Matrix (New):\n{confusion_matrix(y, cv_preds_new)}")

print("\nDone!")
