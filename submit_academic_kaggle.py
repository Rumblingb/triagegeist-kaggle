#!/usr/bin/env python3
# %% [markdown]
# # Beyond Acuity Prediction: An Interpretable Triage Support Pipeline
# ## Structured vitals, complaint text, and patient history for emergency triage decision support
#
# This notebook builds a **second-reader safety layer** for emergency triage using
# structured vitals, derived physiology features, complaint-text signals, and
# patient comorbidity history. The model is framed as clinical decision support,
# not autonomous replacement.

# %% Setup
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
import warnings, os, re
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 3
STRUCTURED_WEIGHT = 0.75  # optimal blend from param sweep
ID_COL = 'patient_id'
TARGET_COL = 'triage_acuity'
TEXT_COL = 'chief_complaint_raw'
INPUT_DIR = '/kaggle/input/competitions/triagegeist'
HIGH_RISK_THRESHOLD = 2

print("Loading data...")
train = pd.read_csv(f'{INPUT_DIR}/train.csv', low_memory=False)
test = pd.read_csv(f'{INPUT_DIR}/test.csv', low_memory=False)
cc = pd.read_csv(f'{INPUT_DIR}/chief_complaints.csv')
ph = pd.read_csv(f'{INPUT_DIR}/patient_history.csv')
print(f"Train: {train.shape}, Test: {test.shape}, Complaints: {cc.shape}, History: {ph.shape}")

# %% Merge tables
train = train.merge(cc, on=ID_COL, how='left').merge(ph, on=ID_COL, how='left')
test = test.merge(cc, on=ID_COL, how='left').merge(ph, on=ID_COL, how='left')
train[TEXT_COL] = train[TEXT_COL].fillna('')
test[TEXT_COL] = test[TEXT_COL].fillna('')

# %% Feature Engineering (clinically motivated)
def engineer_features(df, is_train=True):
    df = df.copy()
    
    # Age grouping
    df['age_group'] = pd.cut(df['age'], bins=[0,18,35,50,65,80,200],
                             labels=['0-18','19-35','36-50','51-65','66-80','80+'])
    
    # Derived vitals
    df['bp_map'] = df['diastolic_bp'] + (df['systolic_bp']-df['diastolic_bp'])/3
    df['shock_index'] = df['heart_rate'] / df['systolic_bp'].replace(0,1)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['hr_resp_ratio'] = df['heart_rate'] / df['respiratory_rate'].replace(0,1)
    
    # Clinical flags
    df['pain_unrecorded'] = (df['pain_score'] == -1).astype(int)
    df['flag_low_oxygen'] = (df['spo2'] < 92).astype(int)
    df['flag_fever'] = (df['temperature_c'] >= 38.0).astype(int)
    df['flag_tachycardia'] = (df['heart_rate'] >= 100).astype(int)
    df['flag_tachypnea'] = (df['respiratory_rate'] >= 22).astype(int)
    df['flag_hypotension'] = (df['systolic_bp'] < 90).astype(int)
    df['flag_gcs_abnormal'] = (df['gcs_total'] < 15).astype(int)
    df['flag_high_news2'] = (df['news2_score'] >= 5).astype(int)
    df['flag_high_shock_index'] = (df['shock_index'] >= 0.9).astype(int)
    df['hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    df['tachycardic'] = (df['heart_rate'] > 100).astype(int)
    df['tachypneic'] = (df['respiratory_rate'] > 20).astype(int)
    df['abnormal_count'] = df[['hypotensive','tachycardic','tachypneic']].sum(axis=1)
    
    # Temporal features
    df['arrival_hour_sin'] = np.sin(2*np.pi*df['arrival_hour']/24)
    df['arrival_hour_cos'] = np.cos(2*np.pi*df['arrival_hour']/24)
    df['night_arrival'] = ((df['arrival_hour']>=22)|(df['arrival_hour']<6)).astype(int)
    
    # Text-based features
    complaint = df[TEXT_COL].fillna("").astype(str).str.lower()
    df['chief_complaint_len'] = complaint.str.len()
    df['chief_complaint_word_count'] = complaint.str.split().str.len()
    
    # Keyword flags
    kw_patterns = {
        'kw_chest_pain': r'chest pain|thoracic pain|crushing chest',
        'kw_stroke_neuro': r'stroke|seizure|thunderclap|loss of vision|weakness|aphasia',
        'kw_respiratory_distress': r'shortness of breath|asthma|hypoxia|wheeze|near-drowning',
        'kw_trauma': r'trauma|fracture|haemothorax|stab|wound|fall|injury',
        'kw_overdose_toxic': r'overdose|poison|toxic|substance',
        'kw_bleeding': r'bleed|haemorrhage|melena|hematemesis',
        'kw_pregnancy': r'pregnan|ectopic|postpartum|miscarriage',
        'kw_infection_sepsis': r'sepsis|fever|necrotising|infection|cellulitis',
    }
    for name, pattern in kw_patterns.items():
        df[name] = complaint.str.contains(pattern, flags=re.IGNORECASE).astype(int)
    
    # Comorbidity burden
    cardio_cols = ['hx_hypertension', 'hx_heart_failure', 'hx_atrial_fibrillation',
                   'hx_coronary_artery_disease', 'hx_peripheral_vascular_disease', 'hx_stroke_prior']
    resp_cols = ['hx_asthma', 'hx_copd']
    neuro_cols = ['hx_dementia', 'hx_epilepsy', 'hx_stroke_prior']
    frailty_cols = ['hx_dementia', 'hx_ckd', 'hx_malignancy', 'hx_immunosuppressed']
    
    df['cardio_burden'] = df[[c for c in cardio_cols if c in df.columns]].sum(axis=1)
    df['respiratory_burden'] = df[[c for c in resp_cols if c in df.columns]].sum(axis=1)
    df['neuro_burden'] = df[[c for c in neuro_cols if c in df.columns]].sum(axis=1)
    df['frailty_burden'] = df[[c for c in frailty_cols if c in df.columns]].sum(axis=1)
    
    return df

print("Engineering features...")
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Drop leakage columns
LEAKAGE = ['disposition', 'ed_los_hours']
for col in LEAKAGE:
    if col in train_fe.columns: train_fe.drop(columns=[col], inplace=True)
    if col in test_fe.columns: test_fe.drop(columns=[col], inplace=True)

# %% Identify feature groups
excluded = {ID_COL, TARGET_COL, TEXT_COL}
numeric_cols = [c for c in train_fe.columns if c not in excluded 
                and pd.api.types.is_numeric_dtype(train_fe[c])]
categorical_cols = [c for c in train_fe.columns if c not in excluded 
                    and not pd.api.types.is_numeric_dtype(train_fe[c])]

# %% Encode categoricals
for c in categorical_cols:
    le = LabelEncoder()
    train_fe[c] = train_fe[c].astype(str)
    test_fe[c] = test_fe[c].astype(str)
    le.fit(pd.concat([train_fe[c], test_fe[c]]).unique())
    train_fe[c] = le.transform(train_fe[c])
    test_fe[c] = le.transform(test_fe[c])

# %% Text → Dense via SVD
print("Processing text features...")
vec = TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True, min_df=5)
Xt_tr = vec.fit_transform(train_fe[TEXT_COL])
Xt_te = vec.transform(test_fe[TEXT_COL])
svd = TruncatedSVD(n_components=150, random_state=42)
Xtd_tr = svd.fit_transform(Xt_tr)
Xtd_te = svd.transform(Xt_te)
print(f"Text SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

# %% Build structured feature matrix
X_struct_tr = train_fe[numeric_cols + categorical_cols].fillna(0).values
X_struct_te = test_fe[numeric_cols + categorical_cols].fillna(0).values
y_tr = train[TARGET_COL].values
test_ids = test[ID_COL].values

# %% 3-Fold CV with Ensemble (HGB + Text blend)
print("\n3-Fold Cross-Validation Ensemble:\n" + "="*50)
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
classes = np.sort(np.unique(y_tr))

oof_struct = np.zeros((len(X_struct_tr), len(classes)))
oof_text = np.zeros((len(X_struct_tr), len(classes)))
test_struct = np.zeros((len(X_struct_te), len(classes)))
test_text = np.zeros((len(X_struct_te), len(classes)))

struct_model = HistGradientBoostingClassifier(
    max_depth=8, learning_rate=0.05, max_iter=300,
    min_samples_leaf=40, l2_regularization=1.0, random_state=SEED
)
text_model = ComplementNB(alpha=0.3)

for f, (ti, vi) in enumerate(skf.split(X_struct_tr, y_tr), 1):
    # Structured model
    sm = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.05, max_iter=300,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED
    )
    sm.fit(X_struct_tr[ti], y_tr[ti])
    oof_struct[vi] = sm.predict_proba(X_struct_tr[vi])
    test_struct += sm.predict_proba(X_struct_te) / N_FOLDS
    
    # Text model
    tm = ComplementNB(alpha=0.3)
    tm.fit(Xtd_tr[ti], y_tr[ti])
    oof_text[vi] = tm.predict_proba(Xtd_tr[vi])
    test_text += tm.predict_proba(Xtd_te) / N_FOLDS
    
    # Fold metrics
    fold_probs = STRUCTURED_WEIGHT * oof_struct[vi] + (1-STRUCTURED_WEIGHT) * oof_text[vi]
    fold_preds = classes[np.argmax(fold_probs, axis=1)]
    mf1 = f1_score(y_tr[vi], fold_preds, average='macro')
    hr_recall = np.mean(fold_preds[y_tr[vi] <= HIGH_RISK_THRESHOLD] <= HIGH_RISK_THRESHOLD)
    ut_rate = np.mean((fold_preds - y_tr[vi]) >= 2)
    print(f"  Fold {f}: MF1={mf1:.4f}  HR-Recall={hr_recall:.4f}  UT-Rate={ut_rate:.4f}")

# %% Full ensemble metrics
oof_probs = STRUCTURED_WEIGHT * oof_struct + (1-STRUCTURED_WEIGHT) * oof_text
train_preds = classes[np.argmax(oof_probs, axis=1)]
mf1 = f1_score(y_tr, train_preds, average='macro')
hr_recall = np.mean(train_preds[y_tr <= HIGH_RISK_THRESHOLD] <= HIGH_RISK_THRESHOLD)
ut_rate = np.mean((train_preds - y_tr) >= 2)

print(f"\n{'='*50}")
print(f"CV ENSEMBLE RESULTS:")
print(f"  Macro-F1:          {mf1:.4f}")
print(f"  High-Risk Recall:  {hr_recall:.4f}")
print(f"  Undertriage Rate:  {ut_rate:.4f}")
print(f"{'='*50}")

# %% Full Training on all data
print("\nTraining final model on full data...")
sm_full = HistGradientBoostingClassifier(
    max_depth=8, learning_rate=0.05, max_iter=300,
    min_samples_leaf=40, l2_regularization=1.0, random_state=SEED
).fit(X_struct_tr, y_tr)
tm_full = ComplementNB(alpha=0.3).fit(Xtd_tr, y_tr)

test_struct_final = sm_full.predict_proba(X_struct_te)
test_text_final = tm_full.predict_proba(Xtd_te)
test_probs = STRUCTURED_WEIGHT * test_struct_final + (1-STRUCTURED_WEIGHT) * test_text_final
test_preds = classes[np.argmax(test_probs, axis=1)]

# %% Create submission
sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})
sub.to_csv('/kaggle/working/submission.csv', index=False)
print(f"Submission saved: {sub.shape}")
print(f"Distribution: {sub[TARGET_COL].value_counts().sort_index().to_dict()}")
print("DONE")
