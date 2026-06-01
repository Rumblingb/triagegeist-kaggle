#!/usr/bin/env python3
"""Triagegeist Kaggle Submission — Self-contained script for Kaggle environment.
Loads data from /kaggle/input/triagegeist/, trains HGB, generates submission.csv"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

ID_COL = 'patient_id'
TARGET_COL = 'triage_acuity'
TEXT_COL = 'chief_complaint_raw'
INPUT_DIR = '/kaggle/input/competitions/triagegeist'

print("Loading data...")
train = pd.read_csv(f'{INPUT_DIR}/train.csv', low_memory=False)
test = pd.read_csv(f'{INPUT_DIR}/test.csv', low_memory=False)
cc = pd.read_csv(f'{INPUT_DIR}/chief_complaints.csv')
print(f"Train: {train.shape}, Test: {test.shape}, Complaints: {cc.shape}")

# Merge chief complaints
train = train.merge(cc, on=ID_COL, how='left')
test = test.merge(cc, on=ID_COL, how='left')
train[TEXT_COL] = train[TEXT_COL].fillna('')
test[TEXT_COL] = test[TEXT_COL].fillna('')

# Feature engineering
def engineer_features(df, is_train=True):
    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins=[0,18,35,50,65,80,200], labels=['0-18','19-35','36-50','51-65','66-80','80+'])
    df['bp_map'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp'])/3
    df['shock_index'] = df['heart_rate'] / df['systolic_bp'].replace(0, 1)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['hr_resp_ratio'] = df['heart_rate'] / df['respiratory_rate'].replace(0, 1)
    df['hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    df['tachycardic'] = (df['heart_rate'] > 100).astype(int)
    df['tachypneic'] = (df['respiratory_rate'] > 20).astype(int)
    df['abnormal_count'] = df[['hypotensive','tachycardic','tachypneic']].sum(axis=1)
    df['arrival_hour_sin'] = np.sin(2*np.pi*df['arrival_hour']/24)
    df['arrival_hour_cos'] = np.cos(2*np.pi*df['arrival_hour']/24)
    df['night_arrival'] = ((df['arrival_hour'] >= 22) | (df['arrival_hour'] < 6)).astype(int)
    return df

print("Engineering features...")
train_fe = engineer_features(train)
test_fe = engineer_features(test, is_train=False)

# Encode categoricals
cat_cols = ['arrival_mode', 'age_group', 'sex', 'night_arrival']
num_cols = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
            'bp_map', 'shock_index', 'pulse_pressure', 'hr_resp_ratio',
            'arrival_hour_sin', 'arrival_hour_cos', 'abnormal_count',
            'hypotensive', 'tachycardic', 'tachypneic']

for c in cat_cols:
    le = LabelEncoder()
    train_fe[c] = train_fe[c].astype(str)
    test_fe[c] = test_fe[c].astype(str)
    le.fit(pd.concat([train_fe[c], test_fe[c]]).unique())
    train_fe[c] = le.transform(train_fe[c])
    test_fe[c] = le.transform(test_fe[c])

# Text features → dense via TruncatedSVD
print("Building text features...")
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,2), sublinear_tf=True)
X_text_train = vectorizer.fit_transform(train_fe[TEXT_COL])
X_text_test = vectorizer.transform(test_fe[TEXT_COL])

svd = TruncatedSVD(n_components=100, random_state=42)
X_text_dense_train = svd.fit_transform(X_text_train)
X_text_dense_test = svd.transform(X_text_test)
print(f"Text SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

# Combined dense features
X_dense_train = np.hstack([train_fe[num_cols + cat_cols].values, X_text_dense_train])
X_dense_test = np.hstack([test_fe[num_cols + cat_cols].values, X_text_dense_test])
y_train = train[TARGET_COL].values
test_ids = test[ID_COL].values

print(f"X_train shape: {X_dense_train.shape}, X_test shape: {X_dense_test.shape}")

# Train HGB with 3-fold CV
print("Training HGB with 3-fold CV...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []

model = HistGradientBoostingClassifier(
    max_depth=7, learning_rate=0.05, max_iter=220, 
    min_samples_leaf=50, random_state=42
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_dense_train, y_train)):
    X_fold_train, X_fold_val = X_dense_train[train_idx], X_dense_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    preds = model.predict(X_fold_val)
    mf1 = f1_score(y_fold_val, preds, average='macro')
    cv_scores.append(mf1)
    print(f"  Fold {fold+1}: MF1 = {mf1:.4f}")

print(f"CV MF1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Retrain on full data
print("Training on full data...")
model.fit(X_dense_train, y_train)
test_preds = model.predict(X_dense_test)

# Create submission
submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"Submission saved: {submission.shape}")
print(f"Distribution: {submission[TARGET_COL].value_counts().sort_index().to_dict()}")
print("DONE")
