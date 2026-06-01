#!/usr/bin/env python3
"""Step 3: Text optimization + Step 4: Final submission generation"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

def pf(*args):
    print(*args, flush=True)

pf("Loading data...")
train = pd.read_csv('/root/triagegeist_data/train.csv')
test = pd.read_csv('/root/triagegeist_data/test.csv')
chief = pd.read_csv('/root/triagegeist_data/chief_complaints.csv')
hist = pd.read_csv('/root/triagegeist_data/patient_history.csv')
train = train.merge(chief, on='patient_id', how='left').merge(hist, on='patient_id', how='left')
test = test.merge(chief, on='patient_id', how='left').merge(hist, on='patient_id', how='left')
y = train['triage_acuity'].values

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
text = train['chief_complaint_raw'].fillna('').astype(str).values
text_test = test['chief_complaint_raw'].fillna('').astype(str).values
classes = np.array([1,2,3,4,5])

results = {}

# Experiment 3a: TF-IDF param sweep
pf("\n--- 3a: TF-IDF param sweep ---")
configs = [
    ('w(1,1)_mf5k', {'ngram_range':(1,1), 'max_features':5000}),
    ('w(1,2)_mf5k', {'ngram_range':(1,2), 'max_features':5000}),
    ('w(1,2)_mf25k', {'ngram_range':(1,2), 'max_features':25000}),
    ('w(1,2)_mf100k', {'ngram_range':(1,2), 'max_features':100000}),
    ('w(1,3)_mf25k', {'ngram_range':(1,3), 'max_features':25000}),
    ('w(2,3)_mf25k', {'ngram_range':(2,3), 'max_features':25000}),
]
for name, cfg in configs:
    t = TfidfVectorizer(lowercase=True, strip_accents='unicode', sublinear_tf=True, min_df=3, **cfg)
    Xt = t.fit_transform(text)
    preds = np.zeros_like(y)
    for fold, (tr, va) in enumerate(skf.split(Xt, y)):
        m = ComplementNB(alpha=0.25)
        m.fit(Xt[tr], y[tr])
        preds[va] = m.predict(Xt[va])
    mf1 = float(f1_score(y, preds, average='macro'))
    results[f'tfidf_{name}'] = mf1
    pf(f"  {name}: {mf1:.4f}")

# 3b: Char n-grams
pf("\n--- 3b: Char n-grams ---")
for ng in [(2,5), (2,6), (3,6)]:
    t = TfidfVectorizer(analyzer='char', lowercase=True, ngram_range=ng, max_features=25000, min_df=3)
    Xt = t.fit_transform(text)
    preds = np.zeros_like(y)
    for fold, (tr, va) in enumerate(skf.split(Xt, y)):
        m = ComplementNB(alpha=0.25)
        m.fit(Xt[tr], y[tr])
        preds[va] = m.predict(Xt[va])
    mf1 = float(f1_score(y, preds, average='macro'))
    results[f'char_{ng[0]}-{ng[1]}'] = mf1
    pf(f"  char_{ng[0]}-{ng[1]}: {mf1:.4f}")

# 3c: Word+Char combined
pf("\n--- 3c: Word+Char combined ---")
tw = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), max_features=25000, sublinear_tf=True, min_df=3)
tc = TfidfVectorizer(analyzer='char', lowercase=True, ngram_range=(2,6), max_features=20000, min_df=3)
Xtw = tw.fit_transform(text)
Xtc = tc.fit_transform(text)
Xt_comb = hstack([Xtw, Xtc])
preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(Xt_comb, y)):
    m = ComplementNB(alpha=0.25)
    m.fit(Xt_comb[tr], y[tr])
    preds[va] = m.predict(Xt_comb[va])
results['word_char_combined'] = float(f1_score(y, preds, average='macro'))
pf(f"  word+char combined: {results['word_char_combined']:.4f}")

# 3d: Classifier comparison (on best TF-IDF)
pf("\n--- 3d: Classifier comparison ---")
t_best = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), max_features=25000, sublinear_tf=True, min_df=3)
Xt_best = t_best.fit_transform(text)
for name, m in [
    ('ComplementNB', ComplementNB(alpha=0.25)),
    ('MultinomialNB', MultinomialNB(alpha=0.1)),
    ('LogReg', LogisticRegression(max_iter=2000, multi_class='multinomial', random_state=42, n_jobs=-1)),
    ('LinearSVC', LinearSVC(max_iter=2000, random_state=42)),
]:
    preds = np.zeros_like(y)
    for fold, (tr, va) in enumerate(skf.split(Xt_best, y)):
        m_clone = m.__class__(**{k:v for k,v in m.get_params().items()}) if hasattr(m, 'get_params') else m
        m_clone.fit(Xt_best[tr], y[tr])
        preds[va] = m_clone.predict(Xt_best[va])
    results[f'clf_{name}'] = float(f1_score(y, preds, average='macro'))
    pf(f"  {name}: {results[f'clf_{name}']:.4f}")

# 3e: TF-IDF SVD features blended with structured
pf("\n--- 3e: TF-IDF SVD + structured blend ---")

# Engineer structured features (same as step2)
def engineer_all(df):
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
    kw_maps = {'kw_chest':r'chest|thoracic|crushing','kw_stroke':r'stroke|seizure|weakness|aphasia',
               'kw_resp':r'shortness of breath|asthma|hypoxia|wheeze','kw_trauma':r'trauma|fracture|stab|wound|fall',
               'kw_sepsis':r'sepsis|fever|infection','kw_bleed':r'bleed|hemorrhage|melena',
               'kw_overdose':r'overdose|poison|toxic','kw_preg':r'pregnan|ectopic|miscarriage'}
    for name, pat in kw_maps.items():
        d[name] = text.str.contains(pat, regex=True).astype(int)
    for grp, cols in [
        ('cardio', ['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation','hx_coronary_artery_disease','hx_peripheral_vascular_disease','hx_stroke_prior']),
        ('resp', ['hx_asthma','hx_copd']), ('neuro', ['hx_dementia','hx_epilepsy','hx_stroke_prior']),
        ('frailty', ['hx_dementia','hx_ckd','hx_malignancy','hx_immunosuppressed'])
    ]:
        d[f'{grp}_burden'] = d[[c for c in cols if c in d.columns]].sum(axis=1)
    # Advanced
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
    d['hour_sin'] = np.sin(2*np.pi*d['arrival_hour']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['arrival_hour']/24)
    d['month_sin'] = np.sin(2*np.pi*d['arrival_month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['arrival_month']/12)
    return d

train_a = engineer_all(train)
exclude = {'triage_acuity', 'disposition', 'ed_los_hours', 'patient_id', 'chief_complaint_raw'}
feat_cols = [c for c in train_a.columns if c not in exclude]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train_a[c])]
cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_a[c])]

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

X_struct = np.nan_to_num(preprocess(train_a), nan=0)
test_a = engineer_all(test)
X_test_struct = np.nan_to_num(preprocess(test_a), nan=0)

# Test SVD features
t_svd = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), max_features=50000, sublinear_tf=True, min_df=3)
Xt_svd_fit = t_svd.fit_transform(text)
xt_svd_te = t_svd.transform(text_test)
from sklearn.ensemble import HistGradientBoostingClassifier
for n_comp in [50, 100]:
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Xt_dense = svd.fit_transform(Xt_svd_fit)
    Xt_dense_te = svd.transform(xt_svd_te)
    X_combined = np.concatenate([X_struct, Xt_dense], axis=1)
    X_test_comb = np.concatenate([X_test_struct, Xt_dense_te], axis=1)
    preds = np.zeros_like(y)
    for fold, (tr, va) in enumerate(skf.split(X_combined, y)):
        m = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=250, min_samples_leaf=50, random_state=42)
        m.fit(X_combined[tr], y[tr])
        preds[va] = m.predict(X_combined[va])
    results[f'svd{n_comp}_hgb'] = float(f1_score(y, preds, average='macro'))
    pf(f"  SVD-{n_comp}+HGB: {results[f'svd{n_comp}_hgb']:.4f}")

pf(f"\nText results saved.")
with open('/root/triagegeist-kaggle/artifacts/text_metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
pf("Step 3 complete!")
