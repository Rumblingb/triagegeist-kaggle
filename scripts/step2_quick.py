#!/usr/bin/env python3
"""Quick model comparison — 2-fold CV, 500 estimators per model"""
import sys, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

print("Loading data...", flush=True)
train = pd.read_csv('/root/triagegeist_data/train.csv').merge(
    pd.read_csv('/root/triagegeist_data/chief_complaints.csv'), on='patient_id', how='left'
).merge(pd.read_csv('/root/triagegeist_data/patient_history.csv'), on='patient_id', how='left')
test = pd.read_csv('/root/triagegeist_data/test.csv').merge(
    pd.read_csv('/root/triagegeist_data/chief_complaints.csv'), on='patient_id', how='left'
).merge(pd.read_csv('/root/triagegeist_data/patient_history.csv'), on='patient_id', how='left')
y = train['triage_acuity'].values
print(f"Train: {train.shape}", flush=True)

# Single feature engineering function
def fe(df):
    d = df.copy()
    txt = d['chief_complaint_raw'].fillna('').astype(str).str.lower()
    d['pain_unrecorded'] = (d['pain_score'] == -1).astype(int); d['pain_score'] = d['pain_score'].replace(-1, np.nan)
    for col, thr, name in [('spo2',92,'low_o2'),('temperature_c',38,'fever'),('heart_rate',100,'tachy'),('respiratory_rate',22,'tachyp'),('systolic_bp',90,'hypot'),('gcs_total',15,'gcs_ab'),('shock_index',0.9,'shock'),('news2_score',5,'high_news2')]:
        d[f'flag_{name}'] = (d[col] >= thr if name not in ['low_o2','gcs_ab','hypot'] else (d[col] < thr)).astype(int)
    d['complaint_len'] = txt.str.len(); d['complaint_word'] = txt.str.split().str.len()
    kw = {'kw_chest':r'chest|thoracic|crushing','kw_stroke':r'stroke|seizure|weakness|aphasia','kw_resp':r'shortness of breath|asthma|hypoxia|wheeze','kw_trauma':r'trauma|fracture|stab|wound|fall','kw_sepsis':r'sepsis|fever|infection','kw_bleed':r'bleed|hemorrhage|melena','kw_overdose':r'overdose|poison|toxic','kw_preg':r'pregnan|ectopic|miscarriage'}
    for n,p in kw.items(): d[n] = txt.str.contains(p, regex=True).astype(int)
    for grp, cols in [('cardio',['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation','hx_coronary_artery_disease','hx_peripheral_vascular_disease','hx_stroke_prior']),('resp',['hx_asthma','hx_copd']),('neuro',['hx_dementia','hx_epilepsy','hx_stroke_prior']),('frailty',['hx_dementia','hx_ckd','hx_malignancy','hx_immunosuppressed'])]:
        d[f'{grp}_burden'] = d[[c for c in cols if c in d.columns]].sum(axis=1)
    # Advanced
    d['pulse_pressure_ratio'] = d['pulse_pressure'] / (d['systolic_bp']+1e-6); d['map_bp_ratio'] = d['mean_arterial_pressure']/(d['systolic_bp']+1e-6)
    d['spo2_fraction'] = d['spo2']/100.0; d['hr_rr_ratio'] = d['heart_rate']/(d['respiratory_rate']+1e-6)
    d['bmi_oxygen'] = d['bmi']*d['spo2']/100.0; d['hr_x_temp'] = d['heart_rate']*d['temperature_c']/100.0
    d['news2_resp'] = (d['respiratory_rate']>20).astype(int)+(d['respiratory_rate']>24).astype(int)
    d['news2_o2'] = (d['spo2']<96).astype(int)+(d['spo2']<94).astype(int)+(d['spo2']<92).astype(int)
    d['news2_conscious'] = (d['gcs_total']<15).astype(int); d['news2_hr'] = (d['heart_rate']>=91).astype(int)+(d['heart_rate']>=111).astype(int)
    d['age_hr'] = d['age']*d['heart_rate']/100.0; d['age_rr'] = d['age']*d['respiratory_rate']/100.0
    d['age_sbp'] = d['age']*d['systolic_bp']/100.0; d['age_shock'] = d['age']*d['shock_index']; d['age_news2'] = d['age']*d['news2_score']/100.0
    hx = [c for c in d.columns if c.startswith('hx_')]
    d['total_hx'] = d[hx].sum(axis=1); d['cardio_renal'] = ((d['hx_heart_failure']>0)&(d['hx_ckd']>0)).astype(int)
    d['hour_sin'] = np.sin(2*np.pi*d['arrival_hour']/24); d['hour_cos'] = np.cos(2*np.pi*d['arrival_hour']/24)
    d['month_sin'] = np.sin(2*np.pi*d['arrival_month']/12); d['month_cos'] = np.cos(2*np.pi*d['arrival_month']/12)
    return d

train = fe(train); test = fe(test)
exc = {'triage_acuity','disposition','ed_los_hours','patient_id','chief_complaint_raw'}
num = [c for c in train.columns if c not in exc and pd.api.types.is_numeric_dtype(train[c])]
cat = [c for c in train.columns if c not in exc and not pd.api.types.is_numeric_dtype(train[c])]

def pre(df):
    Xn = df[num].values.astype(np.float64)
    Xc = df[cat].fillna('missing').astype(str).values
    Xe = np.zeros_like(Xc, dtype=np.float64)
    for j in range(Xc.shape[1]):
        u = sorted(set(Xc[:,j])); m = {v:i for i,v in enumerate(u)}
        for i in range(len(Xc)): Xe[i,j] = m.get(Xc[i,j], -1)
    return np.concatenate([Xn, Xe], axis=1)

X = np.nan_to_num(pre(train), nan=0)
Xt = np.nan_to_num(pre(test), nan=0)
print(f"Features: {X.shape[1]} total", flush=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # 2-fold for speed
classes = np.array([1,2,3,4,5]); nc = 5
results = {}

# HGB
print("HGB...", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
hgb_oof = np.zeros((len(y), nc)); hgb_test = np.zeros((len(Xt), nc))
for f, (tr, va) in enumerate(skf.split(X, y)):
    m = HGB(max_depth=7, learning_rate=0.05, max_iter=250, min_samples_leaf=50, random_state=42)
    m.fit(X[tr], y[tr]); hgb_oof[va] = m.predict_proba(X[va]); hgb_test += m.predict_proba(Xt)/2
    print(f"  Fold {f+1}/2 done", flush=True)
results['HGB'] = float(f1_score(y, classes[np.argmax(hgb_oof, axis=1)], average='macro'))
print(f"  HGB: {results['HGB']:.4f}", flush=True)

# LightGBM
print("LightGBM...", flush=True)
import lightgbm as lgb
lgb_oof = np.zeros((len(y), nc)); lgb_test = np.zeros((len(Xt), nc))
for f, (tr, va) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, max_depth=8, num_leaves=63,
        subsample=0.85, colsample_bytree=0.85, class_weight='balanced',
        reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X[tr], y[tr]); lgb_oof[va] = m.predict_proba(X[va]); lgb_test += m.predict_proba(Xt)/2
    print(f"  Fold {f+1}/2 done", flush=True)
results['LightGBM'] = float(f1_score(y, classes[np.argmax(lgb_oof, axis=1)], average='macro'))
print(f"  LightGBM: {results['LightGBM']:.4f}", flush=True)

# CatBoost
print("CatBoost...", flush=True)
import catboost as cb
cb_oof = np.zeros((len(y), nc)); cb_test = np.zeros((len(Xt), nc))
for f, (tr, va) in enumerate(skf.split(X, y)):
    m = cb.CatBoostClassifier(iterations=800, learning_rate=0.03, depth=8,
        auto_class_weights='Balanced', random_seed=42, verbose=0, l2_leaf_reg=3, loss_function='MultiClass')
    m.fit(X[tr], y[tr]); cb_oof[va] = m.predict_proba(X[va]); cb_test += m.predict_proba(Xt)/2
    print(f"  Fold {f+1}/2 done", flush=True)
results['CatBoost'] = float(f1_score(y, classes[np.argmax(cb_oof, axis=1)], average='macro'))
print(f"  CatBoost: {results['CatBoost']:.4f}", flush=True)

# XGBoost
print("XGBoost...", flush=True)
import xgboost as xgb
xgb_oof = np.zeros((len(y), nc)); xgb_test = np.zeros((len(Xt), nc))
for f, (tr, va) in enumerate(skf.split(X, y)):
    dtrain = xgb.DMatrix(X[tr], label=y[tr]-1); dval = xgb.DMatrix(X[va], label=y[va]-1)
    params = {'max_depth':8,'learning_rate':0.03,'subsample':0.85,'colsample_bytree':0.85,
        'tree_method':'hist','objective':'multi:softprob','num_class':5,'random_state':42,
        'n_jobs':-1,'eval_metric':'mlogloss','min_child_weight':5,'gamma':0.1,'reg_alpha':0.1,'reg_lambda':0.1}
    m = xgb.train(params, dtrain, num_boost_round=800, evals=[(dval,'val')], verbose_eval=False, early_stopping_rounds=30)
    xgb_oof[va] = m.predict(dval)
    xgb_test += m.predict(xgb.DMatrix(Xt))/2
    print(f"  Fold {f+1}/2 done", flush=True)
results['XGBoost'] = float(f1_score(y, classes[np.argmax(xgb_oof, axis=1)], average='macro'))
print(f"  XGBoost: {results['XGBoost']:.4f}", flush=True)

# Random Forest
print("RandomForest...", flush=True)
from sklearn.ensemble import RandomForestClassifier as RF
rf_oof = np.zeros((len(y), nc)); rf_test = np.zeros((len(Xt), nc))
for f, (tr, va) in enumerate(skf.split(X, y)):
    m = RF(n_estimators=300, max_depth=15, min_samples_leaf=5, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    m.fit(X[tr], y[tr]); rf_oof[va] = m.predict_proba(X[va]); rf_test += m.predict_proba(Xt)/2
    print(f"  Fold {f+1}/2 done", flush=True)
results['RF'] = float(f1_score(y, classes[np.argmax(rf_oof, axis=1)], average='macro'))
print(f"  RF: {results['RF']:.4f}", flush=True)

# Ensemble blending
print("\nBlending...", flush=True)
blends = {}
for hw,lw,cw,xw,rw in [
    (0.3,0.3,0.2,0.1,0.1),(0.25,0.35,0.2,0.1,0.1),(0.35,0.35,0.15,0.1,0.05),
    (0.3,0.25,0.25,0.1,0.1),(0.4,0.25,0.2,0.1,0.05),(0.25,0.4,0.2,0.1,0.05),
    (0.33,0.33,0.17,0.17,0.0),(0.3,0.3,0.2,0.1,0.1),(0.2,0.4,0.25,0.1,0.05),
    (0.35,0.3,0.2,0.1,0.05),(0.3,0.2,0.3,0.1,0.1),(0.4,0.2,0.2,0.1,0.1),
]:
    blend = hw*hgb_oof + lw*lgb_oof + cw*cb_oof + xw*xgb_oof + rw*rf_oof
    mf1 = float(f1_score(y, classes[np.argmax(blend, axis=1)], average='macro'))
    blends[f'h{hw}_l{lw}_c{cw}_x{xw}_r{rw}'] = mf1
best_key = max(blends, key=blends.get)
best_mf1 = blends[best_key]

# Summary
print("\n" + "="*50, flush=True)
print("MODEL COMPARISON (2-fold CV)", flush=True)
print("="*50, flush=True)
for name, mf1 in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:15s}: {mf1:.4f}", flush=True)
print(f"  {'BEST BLEND':15s}: {best_mf1:.4f}", flush=True)
print(f"  Weights: {best_key}", flush=True)

# Save
np.save('/root/triagegeist-kaggle/artifacts/oof_hgb.npy', hgb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_lgb.npy', lgb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_cb.npy', cb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_xgb.npy', xgb_oof)
np.save('/root/triagegeist-kaggle/artifacts/oof_rf.npy', rf_oof)
np.save('/root/triagegeist-kaggle/artifacts/test_hgb.npy', hgb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_lgb.npy', lgb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_cb.npy', cb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_xgb.npy', xgb_test)
np.save('/root/triagegeist-kaggle/artifacts/test_rf.npy', rf_test)

with open('/root/triagegeist-kaggle/artifacts/models_comparison.json', 'w') as f:
    json.dump({'models': results, 'best_blend': best_key, 'best_mf1': best_mf1, 'all_blends': blends}, f, indent=2, default=str)

print(f"\nDone! Best blend: {best_mf1:.4f}", flush=True)
