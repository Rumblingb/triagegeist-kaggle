#!/usr/bin/env python3
"""Step 2: Advanced model comparison (LGBM, CatBoost, XGBoost)"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=False)

def print_flush(*args):
    print(*args, flush=True)

print_flush("Loading data...")
train = pd.read_csv('/root/triagegeist_data/train.csv')
test = pd.read_csv('/root/triagegeist_data/test.csv')
chief = pd.read_csv('/root/triagegeist_data/chief_complaints.csv')
hist = pd.read_csv('/root/triagegeist_data/patient_history.csv')
train = train.merge(chief, on='patient_id', how='left').merge(hist, on='patient_id', how='left')
test = test.merge(chief, on='patient_id', how='left').merge(hist, on='patient_id', how='left')
print_flush(f"Train: {train.shape}, Test: {test.shape}")
y = train['triage_acuity'].values

# Engineer features (base + advanced)
def engineer_all(df):
    d = df.copy()
    text = d['chief_complaint_raw'].fillna('').astype(str).str.lower()
    d['pain_unrecorded'] = (d['pain_score'] == -1).astype(int)
    d['pain_score'] = d['pain_score'].replace(-1, np.nan)
    
    # Base flags
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
    
    # Advanced features
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
    d['cardio_renal'] = ((d['hx_heart_failure']>0) & (d['hx_ckd']>0)).astype(int)
    
    d['hour_sin'] = np.sin(2*np.pi*d['arrival_hour']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['arrival_hour']/24)
    d['month_sin'] = np.sin(2*np.pi*d['arrival_month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['arrival_month']/12)
    
    return d

print_flush("Engineering all features...")
train_a = engineer_all(train)
test_a = engineer_all(test)

exclude = {'triage_acuity', 'disposition', 'ed_los_hours', 'patient_id', 'chief_complaint_raw'}
feat_cols = [c for c in train_a.columns if c not in exclude]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train_a[c])]
cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_a[c])]
print_flush(f"Features: {len(num_cols)} num + {len(cat_cols)} cat = {len(feat_cols)} total")

# Preprocess
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

X = preprocess(train_a)
Xn = np.nan_to_num(X, nan=0)
X_test = preprocess(test_a)
X_test_n = np.nan_to_num(X_test, nan=0)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
models_results = {}
all_oof = {}
all_test_probs = {}
classes = np.array([1,2,3,4,5])
n_classes = 5

# 1. LIGHTGBM
print_flush("\n--- LightGBM ---")
import lightgbm as lgb
lgb_oof = np.zeros((len(y), n_classes))
lgb_test = np.zeros((len(X_test_n), n_classes))
for fold, (tr, va) in enumerate(skf.split(Xn, y)):
    print_flush(f"  Fold {fold+1}/3...")
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, max_depth=8, num_leaves=63,
                           subsample=0.85, colsample_bytree=0.85, class_weight='balanced',
                           reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(Xn[tr], y[tr])
    lgb_oof[va] = m.predict_proba(Xn[va])
    lgb_test += m.predict_proba(X_test_n) / 3

lgb_preds = classes[np.argmax(lgb_oof, axis=1)]
lgb_mf1 = float(f1_score(y, lgb_preds, average='macro'))
models_results['lightgbm'] = lgb_mf1
print_fluff = lambda *args: print(*args, flush=True)
print_flush(f"  LightGBM CV MF1: {lgb_mf1:.4f}")

# 2. CATBOOST
print_flush("\n--- CatBoost ---")
import catboost as cb
cb_oof = np.zeros((len(y), n_classes))
cb_test = np.zeros((len(X_test_n), n_classes))
for fold, (tr, va) in enumerate(skf.split(Xn, y)):
    print_flush(f"  Fold {fold+1}/3...")
    m = cb.CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=8,
                               auto_class_weights='Balanced', random_seed=42, verbose=0,
                               l2_leaf_reg=3, loss_function='MultiClass')
    m.fit(Xn[tr], y[tr])
    cb_oof[va] = m.predict_proba(Xn[va])
    cb_test += m.predict_proba(X_test_n) / 3

cb_preds = classes[np.argmax(cb_oof, axis=1)]
cb_mf1 = float(f1_score(y, cb_preds, average='macro'))
models_results['catboost'] = cb_mf1
print_flush(f"  CatBoost CV MF1: {cb_mf1:.4f}")

# 3. XGBoost
print_flush("\n--- XGBoost ---")
import xgboost as xgb
xgb_oof = np.zeros((len(y), n_classes))
xgb_test = np.zeros((len(X_test_n), n_classes))
for fold, (tr, va) in enumerate(skf.split(Xn, y)):
    print_flush(f"  Fold {fold+1}/3...")
    dtrain = xgb.DMatrix(Xn[tr], label=y[tr]-1)
    dval = xgb.DMatrix(Xn[va], label=y[va]-1)
    params = {'max_depth':8, 'learning_rate':0.03, 'subsample':0.85, 'colsample_bytree':0.85,
              'tree_method':'hist', 'objective':'multi:softprob', 'num_class':5,
              'random_state':42, 'n_jobs':-1, 'eval_metric':'mlogloss',
              'min_child_weight':5, 'gamma':0.1, 'reg_alpha':0.1, 'reg_lambda':0.1}
    m = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval,'val')], verbose_eval=False, early_stopping_rounds=30)
    xgb_oof[va] = m.predict(dval)
    dtest = xgb.DMatrix(X_test_n)
    xgb_test += m.predict(dtest) / 3

xgb_preds = classes[np.argmax(xgb_oof, axis=1)]
xgb_mf1 = float(f1_score(y, xgb_preds, average='macro'))
models_results['xgboost'] = xgb_mf1
print_flush(f"  XGBoost CV MF1: {xgb_mf1:.4f}")

# 4. Random Forest
print_flush("\n--- RandomForest ---")
from sklearn.ensemble import RandomForestClassifier
rf_oof = np.zeros((len(y), n_classes))
rf_test = np.zeros((len(X_test_n), n_classes))
for fold, (tr, va) in enumerate(skf.split(Xn, y)):
    print_flush(f"  Fold {fold+1}/3...")
    m = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    m.fit(Xn[tr], y[tr])
    rf_oof[va] = m.predict_proba(Xn[va])
    rf_test += m.predict_proba(X_test_n) / 3

rf_preds = classes[np.argmax(rf_oof, axis=1)]
rf_mf1 = float(f1_score(y, rf_preds, average='macro'))
models_results['randomforest'] = rf_mf1
print_flush(f"  RandomForest CV MF1: {rf_mf1:.4f}")

# First train HGB for the baseline (needed for blending)
print_flush("\n--- HGB Baseline ---")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
hgb_oof = np.zeros((len(y), n_classes))
hgb_test = np.zeros((len(X_test_n), n_classes))
for fold, (tr, va) in enumerate(skf.split(Xn, y)):
    print_flush(f"  HGB Fold {fold+1}/3...")
    m = HGB(max_depth=7, learning_rate=0.05, max_iter=250, min_samples_leaf=50, random_state=42)
    m.fit(Xn[tr], y[tr])
    hgb_oof[va] = m.predict_proba(Xn[va])
    hgb_test += m.predict_proba(X_test_n) / 3
rcv_hgb = hgb_oof.copy()
hgb_preds = classes[np.argmax(hgb_oof, axis=1)]
hgb_mf1 = float(f1_score(y, hgb_preds, average='macro'))
models_results['histgradientboost'] = hgb_mf1
print_flush(f"  HGB CV MF1: {hgb_mf1:.4f}")

# 5. ENSEMBLE BLENDING
print_flush("\n--- Ensemble Blending ---")
blend_results = {}
for hw, lw, cw, rw in [
    (0.4, 0.3, 0.2, 0.1), (0.35, 0.35, 0.2, 0.1), (0.3, 0.4, 0.2, 0.1),
    (0.3, 0.3, 0.3, 0.1), (0.5, 0.25, 0.15, 0.1), (0.25, 0.35, 0.3, 0.1),
    (0.4, 0.4, 0.1, 0.1), (0.2, 0.5, 0.2, 0.1), (0.33, 0.33, 0.17, 0.17),
    (0.3, 0.3, 0.2, 0.2), (0.25, 0.4, 0.25, 0.1),
]:
    blend = hw*rcv_hgb + lw*lgb_oof + cw*cb_oof + rw*rf_oof
    preds = classes[np.argmax(blend, axis=1)]
    mf1 = float(f1_score(y, preds, average='macro'))
    key = f"h{hw}_l{lw}_c{cw}_r{rw}"
    blend_results[key] = mf1

best_key = max(blend_results, key=blend_results.get)
best_mf1 = blend_results[best_key]
print_flush(f"  Best blend: {best_key} = {best_mf1:.4f}")

# Summary
print_flush("\n" + "="*50)
print_flush("MODEL COMPARISON SUMMARY")
print_flush("="*50)
for name, mf1 in sorted(models_results.items(), key=lambda x: -x[1]):
    print_flush(f"  {name:20s}: {mf1:.4f}")
print_flush(f"  {'BEST BLEND':20s}: {best_mf1:.4f}")

# Save oof probs for ensemble
np.save('/root/triagegeist-kaggle/artifacts/oof_hgb.npy', hgb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_lgb.npy', lgb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_cb.npy', cb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_rf.npy', rf_oof)
np.save('/root/triagegeist-kaggle/artifacts/test_hgb.npy', hgb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_lgb.npy', lgb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_cb.npy', cb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_rf.npy', rf_test)

results = {
    'models': models_results,
    'best_blend_key': best_key,
    'best_blend_mf1': best_mf1,
    'blend_all': blend_results
}
with open('/root/triagegeist-kaggle/artifacts/step2_models.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print_flush(f"\nDone! Best model OOF saved.")
