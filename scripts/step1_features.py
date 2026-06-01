#!/usr/bin/env python3
"""Step 1: Baseline + Feature Engineering test"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')

print("Loading data...", flush=True)
train = pd.read_csv('/root/triagegeist_data/train.csv')
test = pd.read_csv('/root/triagegeist_data/test.csv')
chief = pd.read_csv('/root/triagegeist_data/chief_complaints.csv')
hist = pd.read_csv('/root/triagegeist_data/patient_history.csv')
train = train.merge(chief, on='patient_id', how='left').merge(hist, on='patient_id', how='left')
test = test.merge(chief, on='patient_id', how='left').merge(hist, on='patient_id', how='left')
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)
y = train['triage_acuity'].values
print(f"Target dist: {dict(sorted(Counter(y).items()))}", flush=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

def engineer_base(df):
    d = df.copy()
    text = d['chief_complaint_raw'].fillna('').astype(str).str.lower()
    d['pain_unrecorded'] = (d['pain_score'] == -1).astype(int)
    d['pain_score'] = d['pain_score'].replace(-1, np.nan)
    for col, thr, name in [
        ('spo2', 92, 'low_o2'), ('temperature_c', 38, 'fever'), ('heart_rate', 100, 'tachy'),
        ('respiratory_rate', 22, 'tachyp'), ('systolic_bp', 90, 'hypot'), 
        ('gcs_total', 15, 'gcs_ab'), ('shock_index', 0.9, 'shock'), ('news2_score', 5, 'high_news2')
    ]:
        d[f'flag_{name}'] = (d[col] >= thr if name not in ['low_o2','gcs_ab','hypot'] else (d[col] < thr)).astype(int)
    d['complaint_len'] = text.str.len()
    d['complaint_word'] = text.str.split().str.len()
    kw_maps = {'kw_chest':r'chest|thoracic|crushing', 'kw_stroke':r'stroke|seizure|weakness|aphasia',
               'kw_resp':r'shortness of breath|asthma|hypoxia|wheeze', 'kw_trauma':r'trauma|fracture|stab|wound|fall',
               'kw_sepsis':r'sepsis|fever|infection', 'kw_bleed':r'bleed|hemorrhage|melena', 
               'kw_overdose':r'overdose|poison|toxic', 'kw_preg':r'pregnan|ectopic|miscarriage'}
    for name, pat in kw_maps.items():
        d[name] = text.str.contains(pat, regex=True).astype(int)
    for grp, cols in [
        ('cardio', ['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation','hx_coronary_artery_disease','hx_peripheral_vascular_disease','hx_stroke_prior']),
        ('resp', ['hx_asthma','hx_copd']), ('neuro', ['hx_dementia','hx_epilepsy','hx_stroke_prior']),
        ('frailty', ['hx_dementia','hx_ckd','hx_malignancy','hx_immunosuppressed'])
    ]:
        d[f'{grp}_burden'] = d[[c for c in cols if c in d.columns]].sum(axis=1)
    return d

print("Engineering base features...", flush=True)
train_b = engineer_base(train)
test_b = engineer_base(test)
exclude = {'triage_acuity', 'disposition', 'ed_los_hours', 'patient_id', 'chief_complaint_raw'}
feat_cols = [c for c in train_b.columns if c not in exclude]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train_b[c])]
cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_b[c])]
print(f"Features: {len(num_cols)} num + {len(cat_cols)} cat = {len(feat_cols)} total", flush=True)

def preprocess(df):
    Xn = df[num_cols].values.astype(np.float64)
    Xc = df[cat_cols].fillna('missing').astype(str).values
    Xc_enc = np.zeros_like(Xc, dtype=np.float64)
    for j in range(Xc.shape[1]):
        uniq = sorted(set(Xc[:, j]))
        m = {v:i for i,v in enumerate(uniq)}
        for i in range(len(Xc)):
            Xc_enc[i,j] = m.get(Xc[i,j], -1)
    return np.concatenate([Xn, Xc_enc], axis=1)

X = preprocess(train_b)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# BASELINE
print("Training baseline HGB...", flush=True)
preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    m = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=250, min_samples_leaf=50, random_state=42)
    m.fit(X[tr], y[tr])
    preds[va] = m.predict(X[va])
    print(f"  Fold {fold+1} done", flush=True)
base_mf1 = float(f1_score(y, preds, average='macro'))
print(f"BASELINE MF1: {base_mf1:.4f}", flush=True)

# ADVANCED FEATURES
print("\nAdding advanced features...", flush=True)
def engineer_adv(df):
    d = engineer_base(df)
    text = d['chief_complaint_raw'].fillna('').astype(str).str.lower()
    d['pulse_pressure_ratio'] = d['pulse_pressure'] / (d['systolic_bp'] + 1e-6)
    d['map_bp_ratio'] = d['mean_arterial_pressure'] / (d['systolic_bp'] + 1e-6)
    d['spo2_fraction'] = d['spo2'] / 100.0
    d['hr_rr_ratio'] = d['heart_rate'] / (d['respiratory_rate'] + 1e-6)
    d['bmi_oxygen'] = d['bmi'] * d['spo2'] / 100.0
    d['hr_x_temp'] = d['heart_rate'] * d['temperature_c'] / 100.0
    
    d['news2_resp'] = (d['respiratory_rate'] > 20).astype(int) + (d['respiratory_rate'] > 24).astype(int)
    d['news2_o2'] = (d['spo2'] < 96).astype(int) + (d['spo2'] < 94).astype(int) + (d['spo2'] < 92).astype(int)
    d['news2_conscious'] = (d['gcs_total'] < 15).astype(int)
    d['news2_hr'] = (d['heart_rate'] >= 91).astype(int) + (d['heart_rate'] >= 111).astype(int)
    
    d['age_hr'] = d['age'] * d['heart_rate'] / 100.0
    d['age_rr'] = d['age'] * d['respiratory_rate'] / 100.0
    d['age_sbp'] = d['age'] * d['systolic_bp'] / 100.0
    d['age_shock'] = d['age'] * d['shock_index']
    d['age_news2'] = d['age'] * d['news2_score'] / 100.0
    
    hx = [c for c in d.columns if c.startswith('hx_')]
    d['total_hx'] = d[hx].sum(axis=1)
    d['cardio_resp'] = ((d['hx_heart_failure']>0) & (d['hx_copd']>0)).astype(int)
    d['cardio_renal'] = ((d['hx_heart_failure']>0) & (d['hx_ckd']>0)).astype(int)
    d['metabolic'] = ((d['hx_hypertension']>0) & ((d['hx_diabetes_type1']>0)|(d['hx_diabetes_type2']>0)) & (d['hx_obesity']>0)).astype(int)
    
    d['hour_sin'] = np.sin(2*np.pi*d['arrival_hour']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['arrival_hour']/24)
    d['month_sin'] = np.sin(2*np.pi*d['arrival_month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['arrival_month']/12)
    
    return d

train_a = engineer_adv(train)
test_a = engineer_adv(test)
new_feats = [c for c in train_a.columns if c not in set(train_b.columns)]
new_feats = [c for c in new_feats if c not in exclude]
print(f"New features: {len(new_feats)}", flush=True)

all_num = num_cols + [c for c in new_feats if pd.api.types.is_numeric_dtype(train_a[c])]
all_cat = [c for c in new_feats if not pd.api.types.is_numeric_dtype(train_a[c])] + cat_cols

def preprocess_adv(df):
    Xn = df[all_num].values.astype(np.float64)
    Xc = df[all_cat].fillna('missing').astype(str).values
    Xc_enc = np.zeros_like(Xc, dtype=np.float64)
    for j in range(Xc.shape[1]):
        uniq = sorted(set(Xc[:, j]))
        m = {v:i for i,v in enumerate(uniq)}
        for i in range(len(Xc)):
            Xc_enc[i,j] = m.get(Xc[i,j], -1)
    return np.concatenate([Xn, Xc_enc], axis=1)

Xa = preprocess_adv(train_a)
Xa_nonan = np.nan_to_num(Xa, nan=0)

print("Training HGB with advanced features...", flush=True)
preds_a = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(Xa_nonan, y)):
    m = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=250, min_samples_leaf=50, random_state=42)
    m.fit(Xa_nonan[tr], y[tr])
    preds_a[va] = m.predict(Xa_nonan[va])
    print(f"  Fold {fold+1} done", flush=True)
adv_mf1 = float(f1_score(y, preds_a, average='macro'))
print(f"ADV FEATURES MF1: {adv_mf1:.4f}", flush=True)
print(f"IMPROVEMENT: +{adv_mf1 - base_mf1:.4f}", flush=True)

# Save intermediate
results = {'baseline_mf1': base_mf1, 'adv_feats_mf1': adv_mf1, 'improvement': adv_mf1 - base_mf1, 'new_features': new_feats}
with open('/root/triagegeist-kaggle/artifacts/step1_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nStep 1 complete!", flush=True)
