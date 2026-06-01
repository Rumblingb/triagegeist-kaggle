#!/usr/bin/env python3
"""Triagegeist — multi-agent experiment sweep: features, models, text, ensemble"""
import json, os, warnings, time, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

DATA = Path('/root/triagegeist_data')
ARTIFACTS = Path('/root/triagegeist-kaggle/artifacts')
ARTIFACTS.mkdir(parents=True, exist_ok=True)
(ARTIFACTS / 'models').mkdir(exist_ok=True)

np.random.seed(42)
SEED = 42
N_FOLDS = 3
TARGET = 'triage_acuity'
TEXT_COL = 'chief_complaint_raw'
LEAK = ['disposition', 'ed_los_hours', 'patient_id']
HIGH_RISK = 2  # acuity 1-2 = high risk

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

results = {}
t0 = time.time()

# ========= LOAD DATA =========
print("Loading data...")
train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')
chief = pd.read_csv(DATA / 'chief_complaints.csv')
hist = pd.read_csv(DATA / 'patient_history.csv')

train = train.merge(chief, on='patient_id', how='left')
train = train.merge(hist, on='patient_id', how='left')
test = test.merge(chief, on='patient_id', how='left')
test = test.merge(hist, on='patient_id', how='left')
print(f"Train: {train.shape}, Test: {test.shape}")
target_dist = Counter(train[TARGET])
print(f"Target: {dict(sorted(target_dist.items()))}")

y = train[TARGET].values
test_ids = test['patient_id'].values

# ========= BASE FEATURE ENGINEERING =========
def engineer_base(df, is_train=True):
    d = df.copy()
    text = d[TEXT_COL].fillna('').astype(str).str.lower()
    d['pain_unrecorded'] = (d['pain_score'] == -1).astype(int)
    d['pain_score_clean'] = d['pain_score'].replace(-1, np.nan)
    d['flag_low_oxygen'] = (d['spo2'] < 92).astype(int)
    d['flag_fever'] = (d['temperature_c'] >= 38.0).astype(int)
    d['flag_tachycardia'] = (d['heart_rate'] >= 100).astype(int)
    d['flag_tachypnea'] = (d['respiratory_rate'] >= 22).astype(int)
    d['flag_hypotension'] = (d['systolic_bp'] < 90).astype(int)
    d['flag_gcs_abnormal'] = (d['gcs_total'] < 15).astype(int)
    d['flag_high_shock'] = (d['shock_index'] >= 0.9).astype(int)
    d['flag_high_news2'] = (d['news2_score'] >= 5).astype(int)
    d['complaint_len'] = text.str.len()
    d['complaint_words'] = text.str.split().str.len()
    d['has_comma'] = text.str.contains(',', regex=False).astype(int)
    
    cardio = ['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation',
              'hx_coronary_artery_disease','hx_peripheral_vascular_disease','hx_stroke_prior']
    resp = ['hx_asthma','hx_copd']
    neuro = ['hx_dementia','hx_epilepsy','hx_stroke_prior']
    frailty = ['hx_dementia','hx_ckd','hx_malignancy','hx_immunosuppressed']
    d['cardio_burden'] = d[[c for c in cardio if c in d.columns]].sum(axis=1)
    d['resp_burden'] = d[[c for c in resp if c in d.columns]].sum(axis=1)
    d['neuro_burden'] = d[[c for c in neuro if c in d.columns]].sum(axis=1)
    d['frailty_burden'] = d[[c for c in frailty if c in d.columns]].sum(axis=1)
    
    kw = {
        'kw_chest': r'chest pain|thoracic|crushing',
        'kw_stroke': r'stroke|seizure|thunderclap|weakness|aphasia',
        'kw_resp': r'shortness of breath|asthma|hypoxia|wheeze',
        'kw_trauma': r'trauma|fracture|stab|wound|fall|injury',
        'kw_sepsis': r'sepsis|fever|infection|cellulitis',
        'kw_bleed': r'bleed|haemorrhage|melena|hematemesis',
        'kw_overdose': r'overdose|poison|toxic|substance',
        'kw_preg': r'pregnan|ectopic|postpartum|miscarriage',
    }
    for name, pat in kw.items():
        d[name] = text.str.contains(pat, regex=True).astype(int)
    return d

print("\nEngineering base features...")
train_base = engineer_base(train)
test_base = engineer_base(test)

# Columns to exclude from features
exclude = {TARGET, *LEAK}
feature_cols = [c for c in train_base.columns if c not in exclude]
text_cols_actual = [TEXT_COL]
numeric_cols = [c for c in feature_cols if c != TEXT_COL and pd.api.types.is_numeric_dtype(train_base[c])]
categorical_cols = [c for c in feature_cols if c != TEXT_COL and not pd.api.types.is_numeric_dtype(train_base[c])]
print(f"Features: {len(feature_cols)} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

# ========= CV UTILITY =========
def cv_score_hgb(X_train, y_train, **hgb_params):
    """3-fold CV for HGB, returns macro-f1"""
    defaults = dict(max_depth=7, learning_rate=0.05, max_iter=250, min_samples_leaf=50, random_state=SEED)
    defaults.update(hgb_params)
    model = HistGradientBoostingClassifier(**defaults)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y_train)
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
        model.fit(X_train[tr], y_train[tr])
        preds[va] = model.predict(X_train[va])
    return float(f1_score(y_train, preds, average='macro'))

def preprocess_structured(df, numeric_cols, categorical_cols):
    """Preprocess structured features for HGB (no preprocessor needed — HGB handles NaN)"""
    X_num = df[numeric_cols].values.astype(np.float64)
    X_cat = df[categorical_cols].fillna('missing').astype(str).values
    # Encode categoricals
    le_dict = {}
    X_cat_enc = np.zeros_like(X_cat, dtype=np.float64)
    for j in range(X_cat.shape[1]):
        uniq = sorted(set(X_cat[:, j]))
        mapping = {v: i for i, v in enumerate(uniq)}
        for i in range(X_cat.shape[0]):
            X_cat_enc[i, j] = mapping.get(X_cat[i, j], -1)
    return np.concatenate([X_num, X_cat_enc], axis=1)

# ========= EXPERIMENT 1: BASELINE =========
print("\n" + "="*60)
print("EXPERIMENT 1: BASELINE (Base features only)")
print("="*60)
X_base = preprocess_structured(train_base, numeric_cols, categorical_cols)
baseline_mf1 = cv_score_hgb(X_base, y)
print(f"Baseline MF1: {baseline_mf1:.4f}")
results['baseline'] = {'mf1': baseline_mf1}

# ========= EXPERIMENT 2: ADVANCED FEATURES =========
print("\n" + "="*60)
print("EXPERIMENT 2: ADVANCED FEATURE ENGINEERING")
print("="*60)

def engineer_advanced(df, is_train=True):
    d = engineer_base(df)  # start with base
    text = d[TEXT_COL].fillna('').astype(str).str.lower()
    
    # Vital sign ratios
    d['pulse_pressure_ratio'] = d['pulse_pressure'] / (d['systolic_bp'] + 1e-6)
    d['map_bp_ratio'] = d['mean_arterial_pressure'] / (d['systolic_bp'] + 1e-6)
    d['spo2_fraction'] = d['spo2'] / 100.0
    d['hr_rr_ratio'] = d['heart_rate'] / (d['respiratory_rate'] + 1e-6)
    d['bmi_oxygen'] = d['bmi'] * d['spo2'] / 100.0
    d['bp_product'] = d['systolic_bp'] * d['diastolic_bp'] / 1000.0
    
    # NEWS2 decomposition
    d['news2_resp'] = (d['respiratory_rate'] > 20).astype(int) + (d['respiratory_rate'] > 24).astype(int)
    d['news2_o2'] = (d['spo2'] < 96).astype(int) + (d['spo2'] < 94).astype(int) + (d['spo2'] < 92).astype(int)
    d['news2_conscious'] = (d['gcs_total'] < 15).astype(int) + (d['gcs_total'] < 9).astype(int)
    d['news2_temp'] = (d['temperature_c'] <= 35.0).astype(int) + (d['temperature_c'] >= 38.0).astype(int)
    d['news2_hr'] = (d['heart_rate'] <= 40).astype(int) + (d['heart_rate'] >= 91).astype(int) + (d['heart_rate'] >= 111).astype(int) + (d['heart_rate'] >= 131).astype(int)
    
    # Age interactions
    d['age_hr'] = d['age'] * d['heart_rate'] / 100.0
    d['age_rr'] = d['age'] * d['respiratory_rate'] / 100.0
    d['age_sbp'] = d['age'] * d['systolic_bp'] / 100.0
    d['age_shock'] = d['age'] * d['shock_index']
    d['age_news2'] = d['age'] * d['news2_score'] / 100.0
    d['age_temp'] = d['age'] * d['temperature_c'] / 100.0
    
    # Comorbidity interactions
    hx_cols = [c for c in d.columns if c.startswith('hx_')]
    d['total_comorbidity'] = d[hx_cols].sum(axis=1)
    d['cardiorespiratory'] = ((d['hx_heart_failure'] > 0) & (d['hx_copd'] > 0)).astype(int)
    d['cardio_renal'] = ((d['hx_heart_failure'] > 0) & (d['hx_ckd'] > 0)).astype(int)
    d['diabetes_cardio'] = (((d['hx_diabetes_type1']>0)|(d['hx_diabetes_type2']>0)) & (d['hx_coronary_artery_disease']>0)).astype(int)
    d['metabolic_syndrome'] = ((d['hx_hypertension']>0) & ((d['hx_diabetes_type1']>0)|(d['hx_diabetes_type2']>0)) & (d['hx_obesity']>0)).astype(int)
    
    # Temporal features
    d['hour_sin'] = np.sin(2 * np.pi * d['arrival_hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['arrival_hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * d['arrival_month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['arrival_month'] / 12)
    
    # Vital sign interactions
    d['shock_x_spo2'] = d['shock_index'] * d['spo2'] / 100.0
    d['hr_x_temp'] = d['heart_rate'] * d['temperature_c'] / 100.0
    d['rr_x_temp'] = d['respiratory_rate'] * d['temperature_c'] / 100.0
    d['map_x_spo2'] = d['mean_arterial_pressure'] * d['spo2'] / 10000.0
    
    # Keyword count
    total_keywords = 0
    for name in ['kw_chest','kw_stroke','kw_resp','kw_trauma','kw_sepsis','kw_bleed','kw_overdose','kw_preg']:
        if name in d.columns:
            total_keywords += d[name]
    d['total_keywords'] = total_keywords
    
    return d

train_adv = engineer_advanced(train)
test_adv = engineer_advanced(test)

# Identify advanced feature columns (exclude base + leak)
advanced_exclude = {TARGET, *LEAK, TEXT_COL}
base_feats = set(engineer_base(train).columns)
adv_feats_all = set(train_adv.columns)
new_feats_names = list(adv_feats_all - base_feats - {TARGET} - set(LEAK) - {TEXT_COL})
print(f"New features added: {len(new_feats_names)}")

adv_numeric = [c for c in feature_cols + new_feats_names 
               if c != TEXT_COL and pd.api.types.is_numeric_dtype(train_adv[c])]
adv_categorical = [c for c in feature_cols + new_feats_names 
                   if c != TEXT_COL and not pd.api.types.is_numeric_dtype(train_adv[c])]

X_adv = preprocess_structured(train_adv, adv_numeric, adv_categorical)
adv_mf1 = cv_score_hgb(X_adv, y)
print(f"Baseline MF1: {baseline_mf1:.4f}")
print(f"With advanced feats MF1: {adv_mf1:.4f}")
print(f"Improvement: +{adv_mf1 - baseline_mf1:.4f}")
results['feature_engineering'] = {
    'base_mf1': baseline_mf1,
    'advanced_mf1': adv_mf1,
    'improvement': adv_mf1 - baseline_mf1,
    'new_feature_count': len(new_feats_names),
    'new_features': new_feats_names
}

# ========= EXPERIMENT 3: TEXT OPTIMIZATION =========
print("\n" + "="*60)
print("EXPERIMENT 3: TEXT OPTIMIZATION")
print("="*60)

text_corpus = train[TEXT_COL].fillna('').astype(str).values
text_corpus_test = test[TEXT_COL].fillna('').astype(str).values

text_results = {}
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 3a: TF-IDF param sweep
ngram_configs = [(1,1), (1,2), (1,3), (2,3), (1,4)]
max_feat_configs = [5000, 10000, 25000, 50000, 100000]
best_text_mf1 = 0
best_text_config = {}
for ngram in ngram_configs[:3]:  # Test 3 ngram ranges
    for mf in [5000, 25000, 100000]:  # 3 feature sizes
        t = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=ngram,
                           max_features=mf, sublinear_tf=True, min_df=3)
        Xt = t.fit_transform(text_corpus)
        cv_preds = np.zeros_like(y)
        for fold, (tr, va) in enumerate(skf.split(Xt, y)):
            cnb = ComplementNB(alpha=0.25)
            cnb.fit(Xt[tr], y[tr])
            cv_preds[va] = cnb.predict(Xt[va])
        mf1 = float(f1_score(y, cv_preds, average='macro'))
        key = f"tfidf_w{ngram[0]}-{ngram[1]}_mf{mf}"
        text_results[key] = mf1
        if mf1 > best_text_mf1:
            best_text_mf1 = mf1
            best_text_config = {'type': 'tfidf_word', 'ngram': ngram, 'max_features': mf}
        print(f"  {key}: MF1={mf1:.4f}")

# 3b: Char n-grams
for an in ['char']:
    for ng in [(2,5), (2,6), (3,6)]:
        t = TfidfVectorizer(analyzer=an, lowercase=True, ngram_range=ng,
                           max_features=25000, min_df=3, sublinear_tf=True)
        Xt = t.fit_transform(text_corpus)
        cv_preds = np.zeros_like(y)
        for fold, (tr, va) in enumerate(skf.split(Xt, y)):
            cnb = ComplementNB(alpha=0.25)
            cnb.fit(Xt[tr], y[tr])
            cv_preds[va] = cnb.predict(Xt[va])
        mf1 = float(f1_score(y, cv_preds, average='macro'))
        key = f"tfidf_char_{ng[0]}-{ng[1]}"
        text_results[key] = mf1
        if mf1 > best_text_mf1:
            best_text_mf1 = mf1
            best_text_config = {'type': 'tfidf_char', 'ngram': ng, 'max_features': 25000}
        print(f"  {key}: MF1={mf1:.4f}")

# 3c: Combined char + word TF-IDF
best_ngram_w = best_text_config.get('ngram', (1,2)) if best_text_config.get('type') == 'tfidf_word' else (1,2)
best_ngram_c = best_text_config.get('ngram', (2,5)) if best_text_config.get('type') == 'tfidf_char' else (2,5)
tw = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=best_ngram_w if best_text_config.get('type')=='tfidf_word' else (1,2),
                     max_features=25000, sublinear_tf=True, min_df=3)
tc = TfidfVectorizer(analyzer='char', lowercase=True, ngram_range=(2,6),
                     max_features=15000, sublinear_tf=True, min_df=3)
Xtw = tw.fit_transform(text_corpus)
Xtc = tc.fit_transform(text_corpus)
from scipy.sparse import hstack
Xt_combined = hstack([Xtw, Xtc])
cv_preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(Xt_combined, y)):
    cnb = ComplementNB(alpha=0.25)
    cnb.fit(Xt_combined[tr], y[tr])
    cv_preds[va] = cnb.predict(Xt_combined[va])
combined_mf1 = float(f1_score(y, cv_preds, average='macro'))
text_results['word+char_combined'] = combined_mf1
print(f"  word+char_combined: MF1={combined_mf1:.4f}")
if combined_mf1 > best_text_mf1:
    best_text_mf1 = combined_mf1
    best_text_config = {'type': 'combined_word_char'}

# 3d: Classifier comparison
best_tfidf = TfidfVectorizer(lowercase=True, strip_accents='unicode', 
                             ngram_range=(1,2), max_features=25000, sublinear_tf=True, min_df=3)
Xt_best = best_tfidf.fit_transform(text_corpus)
for clf_name, clf in [
    ('ComplementNB', ComplementNB(alpha=0.25)),
    ('MultinomialNB', MultinomialNB(alpha=0.1)),
    ('LogisticRegression', LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=SEED, n_jobs=-1)),
    ('LinearSVC', LinearSVC(max_iter=2000, random_state=SEED)),
    ('SGD', SGDClassifier(loss='log_loss', max_iter=1000, random_state=SEED, n_jobs=-1)),
]:
    cv_preds = np.zeros_like(y)
    for fold, (tr, va) in enumerate(skf.split(Xt_best, y)):
        clf.fit(Xt_best[tr], y[tr])
        cv_preds[va] = clf.predict(Xt_best[va])
    mf1 = float(f1_score(y, cv_preds, average='macro'))
    text_results[f'clf_{clf_name}'] = mf1
    print(f"  {clf_name}: MF1={mf1:.4f}")

# 3e: SVD on TF-IDF as dense features
t_svd = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2),
                        max_features=50000, sublinear_tf=True, min_df=3)
Xt_svd = t_svd.fit_transform(text_corpus)
for n_comp in [50, 100, 200]:
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    Xt_dense = svd.fit_transform(Xt_svd)
    # Add dense features to structured data
    X_combined = np.concatenate([X_base, Xt_dense], axis=1)
    mf1 = cv_score_hgb(X_combined, y)
    text_results[f'svd_{n_comp}_dense'] = mf1
    print(f"  SVD-{n_comp} + HGB: MF1={mf1:.4f}")

text_results['best_config'] = best_text_config
text_results['best_mf1'] = best_text_mf1
with open(ARTIFACTS / 'text_best_config.json', 'w') as f:
    json.dump(best_text_config, f, indent=2)
with open(ARTIFACTS / 'text_metrics.json', 'w') as f:
    json.dump(text_results, f, indent=2, default=str)
results['text_optimization'] = text_results
print(f"Best text MF1: {best_text_mf1:.4f} (config: {best_text_config})")

# ========= EXPERIMENT 4: ADVANCED MODELS =========
print("\n" + "="*60)
print("EXPERIMENT 4: ADVANCED MODEL COMPARISON")
print("="*60)

model_results = {}
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Use advanced features for all models
X_adv_nonan = np.nan_to_num(X_adv, nan=0)

# 4a: LightGBM
print("\n4a: LightGBM...")
lgb_preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_nonan, y)):
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=-1, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        random_state=SEED, verbose=-1, n_jobs=-1
    )
    model.fit(X_adv_nonan[tr], y[tr])
    lgb_preds[va] = model.predict(X_adv_nonan[va])
lgb_mf1 = float(f1_score(y, lgb_preds, average='macro'))
model_results['lightgbm_default'] = lgb_mf1
print(f"  LightGBM MF1: {lgb_mf1:.4f}")

# LightGBM tuned
lgb_preds_t = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_nonan, y)):
    model = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.05, max_depth=8, num_leaves=63,
        subsample=0.85, colsample_bytree=0.85, class_weight='balanced',
        reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
        random_state=SEED, verbose=-1, n_jobs=-1
    )
    model.fit(X_adv_nonan[tr], y[tr])
    lgb_preds_t[va] = model.predict(X_adv_nonan[va])
lgb_tuned_mf1 = float(f1_score(y, lgb_preds_t, average='macro'))
model_results['lightgbm_tuned'] = lgb_tuned_mf1
print(f"  LightGBM tuned MF1: {lgb_tuned_mf1:.4f}")

# 4b: CatBoost
print("\n4b: CatBoost...")
# CatBoost needs NaN handling
X_adv_cat = np.nan_to_num(X_adv, nan=-999)
cb_preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_cat, y)):
    model = cb.CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        auto_class_weights='Balanced', random_seed=SEED, verbose=0, thread_count=4,
        loss_function='MultiClass'
    )
    model.fit(X_adv_cat[tr], y[tr])
    cb_preds[va] = model.predict(X_adv_cat[va])
cb_mf1 = float(f1_score(y, cb_preds, average='macro'))
model_results['catboost'] = cb_mf1
print(f"  CatBoost MF1: {cb_mf1:.4f}")

# CatBoost tuned
cb_preds_t = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_cat, y)):
    model = cb.CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=8,
        auto_class_weights='Balanced', random_seed=SEED, verbose=0, thread_count=4,
        l2_leaf_reg=3, loss_function='MultiClass'
    )
    model.fit(X_adv_cat[tr], y[tr])
    cb_preds_t[va] = model.predict(X_adv_cat[va])
cb_tuned_mf1 = float(f1_score(y, cb_preds_t, average='macro'))
model_results['catboost_tuned'] = cb_tuned_mf1
print(f"  CatBoost tuned MF1: {cb_tuned_mf1:.4f}")

# 4c: XGBoost
print("\n4c: XGBoost...")
xgb_preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_nonan, y)):
    dtrain = xgb.DMatrix(X_adv_nonan[tr], label=y[tr] - 1)  # XGB expects 0-indexed
    dval = xgb.DMatrix(X_adv_nonan[va], label=y[va] - 1)
    params = {
        'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 1000,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist',
        'objective': 'multi:softmax', 'num_class': 5, 'random_state': SEED,
        'n_jobs': 4, 'eval_metric': 'mlogloss',
    }
    model = xgb.train(params, dtrain, num_boost_round=500, 
                      evals=[(dval, 'val')], verbose_eval=False,
                      early_stopping_rounds=30)
    xgb_preds[va] = model.predict(dval).astype(int) + 1
xgb_mf1 = float(f1_score(y, xgb_preds, average='macro'))
model_results['xgboost'] = xgb_mf1
print(f"  XGBoost MF1: {xgb_mf1:.4f}")

# XGBoost tuned
xgb_preds_t = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_nonan, y)):
    dtrain = xgb.DMatrix(X_adv_nonan[tr], label=y[tr] - 1)
    dval = xgb.DMatrix(X_adv_nonan[va], label=y[va] - 1)
    params = {
        'max_depth': 8, 'learning_rate': 0.03, 'n_estimators': 1500,
        'subsample': 0.85, 'colsample_bytree': 0.85, 'tree_method': 'hist',
        'objective': 'multi:softmax', 'num_class': 5, 'random_state': SEED,
        'n_jobs': 4, 'eval_metric': 'mlogloss',
        'min_child_weight': 5, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    }
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, 'val')], verbose_eval=False,
                      early_stopping_rounds=30)
    xgb_preds_t[va] = model.predict(dval).astype(int) + 1
xgb_tuned_mf1 = float(f1_score(y, xgb_preds_t, average='macro'))
model_results['xgboost_tuned'] = xgb_tuned_mf1
print(f"  XGBoost tuned MF1: {xgb_tuned_mf1:.4f}")

# 4d: RandomForest (baseline comparison - already tuned)
rf_preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_adv_nonan, y)):
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=SEED, n_jobs=-1
    )
    rf.fit(X_adv_nonan[tr], y[tr])
    rf_preds[va] = rf.predict(X_adv_nonan[va])
rf_mf1 = float(f1_score(y, rf_preds, average='macro'))
model_results['randomforest'] = rf_mf1
print(f"  RandomForest MF1: {rf_mf1:.4f}")

# Save model comparison
with open(ARTIFACTS / 'model_comparison.json', 'w') as f:
    json.dump(model_results, f, indent=2)
results['model_comparison'] = model_results

print("\n" + "="*60)
print(f"MODEL COMPARISON SUMMARY")
print("="*60)
for name, mf1 in sorted(model_results.items(), key=lambda x: -x[1]):
    print(f"  {name:20s}: MF1={mf1:.4f}")

best_model_name = max(model_results, key=model_results.get)
best_model_mf1 = model_results[best_model_name]
print(f"\nBest: {best_model_name} = {best_model_mf1:.4f}")
print(f"Over baseline (+{best_model_mf1 - baseline_mf1:.4f})")

# ========= EXPERIMENT 5: ENSEMBLE (TEXT + STRUCTURED BLEND) =========
print("\n" + "="*60)
print("EXPERIMENT 5: ENSEMBLE BLENDING")
print("="*60)

from sklearn.model_selection import StratifiedKFold

# Train HGB on advanced features with CV for OOF predictions
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
classes = np.array([1, 2, 3, 4, 5])
n_classes = 5

# OOF and test predictions
n_train = len(y)
n_test = len(test)
oof_hgb = np.zeros((n_train, n_classes))
oof_lgb = np.zeros((n_train, n_classes))
oof_cb = np.zeros((n_train, n_classes))
test_hgb = np.zeros((n_test, n_classes))
test_lgb = np.zeros((n_test, n_classes))
test_cb = np.zeros((n_test, n_classes))

print("Generating OOF predictions for 3 base models...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv_nonan, y), 1):
    print(f"  Fold {fold}/3...")
    
    # HGB
    hgb = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=250, 
                                          min_samples_leaf=50, random_state=SEED)
    hgb.fit(X_adv_nonan[tr_idx], y[tr_idx])
    oof_hgb[va_idx] = hgb.predict_proba(X_adv_nonan[va_idx])
    test_hgb += hgb.predict_proba(X_adv_nonan[n_train:]) / N_FOLDS  # test is conjoined
    
    # Wait - I need to handle test separately. Let me just use the first n_train rows for training.
    # Actually X_adv was built only on train. Test = test_adv. Let me fix this.
    
print("Building test features...")
X_test_adv = preprocess_structured(test_adv, adv_numeric, adv_categorical)
X_test_adv_nonan = np.nan_to_num(X_test_adv, nan=0)
n_test = len(X_test_adv_nonan)

# Re-do with proper test handling
oof_hgb = np.zeros((n_train, n_classes))
oof_lgb = np.zeros((n_train, n_classes))
oof_cb = np.zeros((n_train, n_classes))
test_hgb = np.zeros((n_test, n_classes))
test_lgb = np.zeros((n_test, n_classes))
test_cb = np.zeros((n_test, n_classes))

print("Regenerating OOF + test predictions...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv_nonan, y), 1):
    print(f"  Fold {fold}/3...")
    
    # HGB
    hgb = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=250,
                                          min_samples_leaf=50, random_state=SEED)
    hgb.fit(X_adv_nonan[tr_idx], y[tr_idx])
    oof_hgb[va_idx] = hgb.predict_proba(X_adv_nonan[va_idx])
    test_hgb += hgb.predict_proba(X_test_adv_nonan) / N_FOLDS
    
    # LightGBM (tuned)
    lgb_m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, max_depth=8, num_leaves=63,
                                subsample=0.85, colsample_bytree=0.85, class_weight='balanced',
                                reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
                                random_state=SEED, verbose=-1, n_jobs=-1)
    lgb_m.fit(X_adv_nonan[tr_idx], y[tr_idx])
    oof_lgb[va_idx] = lgb_m.predict_proba(X_adv_nonan[va_idx])
    test_lgb += lgb_m.predict_proba(X_test_adv_nonan) / N_FOLDS
    
    # CatBoost (tuned)
    cb_m = cb.CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=8,
                                  auto_class_weights='Balanced', random_seed=SEED, verbose=0,
                                  l2_leaf_reg=3, loss_function='MultiClass')
    cb_m.fit(X_adv_cat[tr_idx], y[tr_idx])  # X_adv_cat = nan filled
    oof_cb[va_idx] = cb_m.predict_proba(X_adv_cat[va_idx])
    test_cb += cb_m.predict_proba(X_test_adv_nonan) / N_FOLDS

# Try different blend weights
blend_results = {}
for hgb_w, lgb_w, cb_w in [
    (0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2),
    (0.6, 0.2, 0.2), (0.4, 0.3, 0.3), (0.5, 0.2, 0.3),
    (0.33, 0.33, 0.34), (0.2, 0.6, 0.2), (0.2, 0.4, 0.4),
    (0.7, 0.15, 0.15), (0.25, 0.5, 0.25),
]:
    blend = hgb_w * oof_hgb + lgb_w * oof_lgb + cb_w * oof_cb
    preds = classes[np.argmax(blend, axis=1)]
    mf1 = float(f1_score(y, preds, average='macro'))
    hr_recall = float(recall_score(y[y <= HIGH_RISK], preds[y <= HIGH_RISK], average='macro'))
    ut = float(np.mean((preds - y) >= 2))
    blend_results[f'HGB={hgb_w}_LGB={lgb_w}_CB={cb_w}'] = {
        'mf1': mf1, 'high_risk_recall': hr_recall, 'undertriage': ut
    }
    
# Find best blend
best_blend = max(blend_results, key=lambda k: blend_results[k]['mf1'])
best_blend_mf1 = blend_results[best_blend]['mf1']
print(f"\nBest blend: {best_blend}")
print(f"  MF1={best_blend_mf1:.4f}, HR={blend_results[best_blend]['high_risk_recall']:.4f}, "
      f"UT={blend_results[best_blend]['undertriage']:.4f}")

# Parse weights
import re
wm = re.match(r'HGB=([\d.]+)_LGB=([\d.]+)_CB=([\d.]+)', best_blend)
hgb_w_best, lgb_w_best, cb_w_best = float(wm.group(1)), float(wm.group(2)), float(wm.group(3))

results['ensemble'] = {'blend_results': blend_results, 'best_blend': best_blend, 'best_mf1': best_blend_mf1}

# ========= FINAL SUBMISSION =========
print("\n" + "="*60)
print("FINAL SUBMISSION")
print("="*60)

# Train best models on full data
print("Training final models on all data...")
hgb_final = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=250,
                                            min_samples_leaf=50, random_state=SEED)
hgb_final.fit(X_adv_nonan, y)
test_hgb_final = hgb_final.predict_proba(X_test_adv_nonan)

lgb_final = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, max_depth=8, num_leaves=63,
                                subsample=0.85, colsample_bytree=0.85, class_weight='balanced',
                                reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
                                random_state=SEED, verbose=-1, n_jobs=-1)
lgb_final.fit(X_adv_nonan, y)
test_lgb_final = lgb_final.predict_proba(X_test_adv_nonan)

cb_final = cb.CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=8,
                                  auto_class_weights='Balanced', random_seed=SEED, verbose=0,
                                  l2_leaf_reg=3, loss_function='MultiClass')
cb_final.fit(X_adv_cat, y)
test_cb_final = cb_final.predict_proba(X_test_adv_nonan)

# Blend
test_blend = hgb_w_best * test_hgb_final + lgb_w_best * test_lgb_final + cb_w_best * test_cb_final
test_preds = classes[np.argmax(test_blend, axis=1)]

# Save
sub = pd.DataFrame({'patient_id': test_ids, 'triage_acuity': test_preds})
sub.to_csv(ARTIFACTS / 'final_submission.csv', index=False)
sub.to_csv('/root/triagegeist-kaggle/submission.csv', index=False)
print(f"Final submission saved: {len(sub)} predictions")
print(f"Prediction distribution: {dict(sorted(Counter(test_preds).items()))}")

# Save all metrics
elapsed = time.time() - t0
results['final'] = {
    'best_cv_mf1': best_blend_mf1,
    'best_model_name': best_model_name,
    'best_single_mf1': best_model_mf1,
    'blend_hgb_w': hgb_w_best,
    'blend_lgb_w': lgb_w_best,
    'blend_cb_w': cb_w_best,
    'elapsed_seconds': elapsed,
    'submission_file': 'artifacts/final_submission.csv',
    'test_pred_distribution': dict(sorted(Counter(test_preds).items()))
}

with open(ARTIFACTS / 'ensemble_metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*60)
print(" COMPLETE!")
print("="*60)
print(f"\nBaseline:         {baseline_mf1:.4f}")
print(f"Adv Features:     {adv_mf1:.4f} (+{adv_mf1 - baseline_mf1:.4f})")
print(f"Best Text:        {best_text_mf1:.4f}")
print(f"Best Single:      {best_model_mf1:.4f} ({best_model_name})")
print(f"Best Blend (CV):  {best_blend_mf1:.4f}")
print(f"Elapsed:          {elapsed:.0f}s")
print(f"Submission:       /root/triagegeist-kaggle/artifacts/final_submission.csv")
