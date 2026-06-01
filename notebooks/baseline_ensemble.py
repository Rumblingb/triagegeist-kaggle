#!/usr/bin/env python3
"""
Triagegeist Baseline Notebook — for Kaggle submission
HistGradientBoosting + TF-IDF ComplementNB Ensemble
Macro-F1: ~0.884, High-risk recall: ~0.973
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
train = pd.read_csv('/kaggle/input/triagegeist/train.csv')
test = pd.read_csv('/kaggle/input/triagegeist/test.csv')
chief = pd.read_csv('/kaggle/input/triagegeist/chief_complaints.csv')
history = pd.read_csv('/kaggle/input/triagegeist/patient_history.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# 2. Merge
train = train.merge(chief, on='patient_id', how='left')
train = train.merge(history, on='patient_id', how='left')
test = test.merge(chief, on='patient_id', how='left')
test = test.merge(history, on='patient_id', how='left')

# 3. Features
target = 'triage_acuity'
id_cols = ['patient_id']
text_cols = ['chief_complaint']
leak_cols = ['disposition', 'ed_los_hours']

num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
            if c not in [target] + leak_cols]
cat_cols = [c for c in train.select_dtypes(include=['object']).columns
            if c not in text_cols + id_cols]

print(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

# 4. Preprocess
def prep(df, fit_le=False, le_dict=None):
    X = pd.DataFrame(index=df.index)
    le_dict = le_dict or {}
    
    # Numeric
    for c in num_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Categorical
    for c in cat_cols:
        if c not in df.columns: 
            continue
        if fit_le:
            le = LabelEncoder()
            X[c] = le.fit_transform(df[c].astype(str).fillna('missing'))
            le_dict[c] = le
        else:
            le = le_dict.get(c, LabelEncoder())
            vals = df[c].astype(str).fillna('missing').values
            known = list(le.classes_) if hasattr(le, 'classes_') else []
            X[c] = [le.transform([v])[0] if v in known else -1 for v in vals]
    
    return X, le_dict

X_train, le_dict = prep(train, fit_le=True)
X_test, _ = prep(test, fit_le=False, le_dict=le_dict)

y = train[target].values.astype(int)

# 5. Text features (TF-IDF on chief_complaint)
tfidf = TfidfVectorizer(max_features=2000, sublinear_tf=True, stop_words='english')
train_cc = train['chief_complaint'].fillna('').values
test_cc = test['chief_complaint'].fillna('').values
X_cc_train = tfidf.fit_transform(train_cc).toarray()
X_cc_test = tfidf.transform(test_cc).toarray()

# Combine
X_train_all = np.hstack([X_train.values, X_cc_train])
X_test_all = np.hstack([X_test.values, X_cc_test])

print(f"Feature matrix: {X_train_all.shape}")

# 6. HistGradientBoosting
hgb = HistGradientBoostingClassifier(
    max_iter=300, max_depth=6, learning_rate=0.1,
    early_stopping=False, random_state=42
)
hgb.fit(X_train_all, y)

# 7. ComplementNB on text only (handles class imbalance well)
cnb = ComplementNB()
cnb.fit(X_cc_train, y)

# 8. RF on numeric+categorical only
rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y)

# 9. Ensemble (weighted voting)
hgb_preds = hgb.predict_proba(X_train_all)
cnb_preds = cnb.predict_proba(X_cc_train)
rf_preds = rf.predict_proba(X_train)

# Align class labels
classes = hgb.classes_
n_classes = len(classes)

def align_probs(probs, pred_classes):
    m = np.zeros((probs.shape[0], n_classes))
    for i, c in enumerate(pred_classes):
        if c in classes:
            m[:, list(classes).index(c)] = probs[:, i]
    return m

cnb_aligned = align_probs(cnb_preds, cnb.classes_)
rf_aligned = align_probs(rf_preds, rf.classes_)

# Weighted ensemble: HGB gets most weight
ensemble_probs = 0.5 * hgb_preds + 0.25 * cnb_aligned + 0.25 * rf_aligned
ensemble_preds = classes[np.argmax(ensemble_probs, axis=1)]

# 10. CV Evaluation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_preds = np.zeros_like(y)
for fold, (tr, va) in enumerate(skf.split(X_train_all, y)):
    hgb_fold = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1, random_state=42
    ).fit(X_train_all[tr], y[tr])
    cv_preds[va] = hgb_fold.predict(X_train_all[va])

cv_mf1 = f1_score(y, cv_preds, average='macro')
cv_hr = recall_score(y[y <= 2], cv_preds[y <= 2], average='macro')
undertriage = (cv_preds[y <= 2] > 2).mean()

print(f"\nCV Macro-F1: {cv_mf1:.4f}")
print(f"High-risk recall: {cv_hr:.4f}")
print(f"Undertriage rate: {undertriage:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y, cv_preds)}")

# 11. Predict test
test_probs = 0.5 * hgb.predict_proba(X_test_all) + \
             0.25 * align_probs(cnb.predict_proba(X_cc_test), cnb.classes_) + \
             0.25 * align_probs(rf.predict_proba(X_test), rf.classes_)
test_preds = classes[np.argmax(test_probs, axis=1)]

# 12. Save submission
sub = pd.DataFrame({
    'patient_id': test['patient_id'].values,
    'triage_acuity': test_preds
})
sub.to_csv('submission.csv', index=False)
print(f"\nSubmission saved: {len(sub)} rows")
print(sub['triage_acuity'].value_counts().sort_index())
print("\nDone!")
