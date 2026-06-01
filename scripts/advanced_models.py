#!/usr/bin/env python3
"""
Advanced Model Comparison for Triagegeist (Kaggle).
Trains LightGBM, CatBoost, XGBoost, and RandomForest with 3-fold CV.
Reports Macro-F1 for each, then tunes the best model.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model imports with error handling
_MODELS_AVAILABLE = {}

try:
    from lightgbm import LGBMClassifier
    _MODELS_AVAILABLE['lightgbm'] = True
except ImportError:
    _MODELS_AVAILABLE['lightgbm'] = False

try:
    from catboost import CatBoostClassifier
    _MODELS_AVAILABLE['catboost'] = True
except ImportError:
    _MODELS_AVAILABLE['catboost'] = False

try:
    from xgboost import XGBClassifier
    _MODELS_AVAILABLE['xgboost'] = True
except ImportError:
    _MODELS_AVAILABLE['xgboost'] = False

try:
    from sklearn.ensemble import RandomForestClassifier
    _MODELS_AVAILABLE['rf'] = True
except ImportError:
    _MODELS_AVAILABLE['rf'] = False

warnings.filterwarnings('ignore')

DATA_DIR = Path('/root/triagegeist_data')
ARTIFACTS_DIR = Path('/root/triagegeist-kaggle/artifacts')
RANDOM_SEED = 42
N_FOLDS = 3


def load_and_merge_data():
    """Load and merge all data sources."""
    print("=" * 60)
    print("Loading and merging data...")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    cc = pd.read_csv(DATA_DIR / 'chief_complaints.csv')
    ph = pd.read_csv(DATA_DIR / 'patient_history.csv')

    print(f"  train: {train.shape}, test: {test.shape}")
    print(f"  chief_complaints: {cc.shape}, patient_history: {ph.shape}")

    # Merge training data
    train = train.merge(cc, on='patient_id', how='left')
    train = train.merge(ph, on='patient_id', how='left')

    print(f"  merged train: {train.shape}")

    return train, test


def add_features(df, is_train=True):
    """Add engineered features."""
    print("\nAdding features...")

    # Pain unrecorded flag
    df['pain_unrecorded'] = (df['pain_score'] == -1).astype(int)

    # Vital sign flags
    # Handle NaN gracefully
    df['low_oxygen'] = (df['spo2'] < 92).fillna(0).astype(int)
    df['fever'] = (df['temperature_c'] >= 38).fillna(0).astype(int)
    df['tachycardia'] = (df['heart_rate'] >= 100).fillna(0).astype(int)
    df['tachypnea'] = (df['respiratory_rate'] >= 22).fillna(0).astype(int)
    df['hypotension'] = (df['systolic_bp'] < 90).fillna(0).astype(int)
    df['gcs_abnormal'] = (df['gcs_total'] < 15).fillna(0).astype(int)
    df['high_shock_index'] = (df['shock_index'] >= 0.9).fillna(0).astype(int)

    print(f"  Added pain_unrecorded, low_oxygen, fever, tachycardia, tachypnea, hypotension, gcs_abnormal, high_shock_index")
    return df


def prepare_features_labels(df):
    """Split into features and labels, handling categoricals."""
    # Target
    y = df['triage_acuity'].values - 1  # make 0-indexed

    # Columns to exclude
    exclude_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity']
    if 'chief_complaint_raw' in df.columns:
        exclude_cols.append('chief_complaint_raw')

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution: {np.bincount(y)}")

    return X, y, feature_cols


def get_preprocessor(X):
    """Create column transformer for numeric + categorical features."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\n  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def cv_score(model, preprocessor, X, y, model_name):
    """Run stratified 3-fold CV and return macro-f1 scores."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]

        # Fit preprocessor and transform
        X_train_processed = preprocessor.fit_transform(X_train_fold)
        X_val_processed = preprocessor.transform(X_val_fold)

        # Handle any remaining NaN (e.g., from catboost)
        if hasattr(X_train_processed, 'toarray'):
            X_train_processed = X_train_processed.toarray()
            X_val_processed = X_val_processed.toarray()

        X_train_processed = np.nan_to_num(X_train_processed, nan=0.0)
        X_val_processed = np.nan_to_num(X_val_processed, nan=0.0)

        # Train model
        model.fit(X_train_processed, y_train_fold)

        # Predict
        y_pred = model.predict(X_val_processed)

        # Calculate macro-f1
        mf1 = f1_score(y_val_fold, y_pred, average='macro')
        fold_scores.append(mf1)
        print(f"    Fold {fold+1}: MF1 = {mf1:.4f}")

    mean_mf1 = np.mean(fold_scores)
    std_mf1 = np.std(fold_scores)
    print(f"  {model_name} CV MF1: {mean_mf1:.4f} ± {std_mf1:.4f}")

    return mean_mf1, std_mf1, fold_scores


def train_and_evaluate_lightgbm(preprocessor, X, y):
    """LightGBM with 3-fold CV."""
    print("\n" + "-" * 50)
    print("Model a) LightGBM")
    print("-" * 50)

    if not _MODELS_AVAILABLE['lightgbm']:
        print("  SKIPPED: lightgbm not installed")
        return None, None, None

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1
    )

    return cv_score(model, preprocessor, X, y, "LightGBM")


def train_and_evaluate_catboost(preprocessor, X, y):
    """CatBoost with 3-fold CV."""
    print("\n" + "-" * 50)
    print("Model b) CatBoost")
    print("-" * 50)

    if not _MODELS_AVAILABLE['catboost']:
        print("  SKIPPED: catboost not installed")
        return None, None, None

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced',
        random_seed=RANDOM_SEED,
        verbose=0,
        thread_count=-1
    )

    return cv_score(model, preprocessor, X, y, "CatBoost")


def train_and_evaluate_xgboost(preprocessor, X, y):
    """XGBoost with 3-fold CV."""
    print("\n" + "-" * 50)
    print("Model c) XGBoost")
    print("-" * 50)

    if not _MODELS_AVAILABLE['xgboost']:
        print("  SKIPPED: xgboost not installed")
        return None, None, None

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=RANDOM_SEED,
        tree_method='hist',
        n_jobs=-1
    )

    return cv_score(model, preprocessor, X, y, "XGBoost")


def train_and_evaluate_rf(preprocessor, X, y):
    """RandomForest with 3-fold CV."""
    print("\n" + "-" * 50)
    print("Model d) RandomForest")
    print("-" * 50)

    if not _MODELS_AVAILABLE['rf']:
        print("  SKIPPED: rf not installed")
        return None, None, None

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    return cv_score(model, preprocessor, X, y, "RandomForest")


def tune_best_model(best_name, preprocessor, X, y):
    """Quick param sweep for the best model with 3-fold CV."""
    print("\n" + "=" * 60)
    print(f"e) BEST MODEL TUNING: {best_name}")
    print("=" * 60)

    if best_name == 'LightGBM':
        param_grid = {
            'num_leaves': [31, 63, 127, 255],
            'learning_rate': [0.01, 0.03, 0.05, 0.1]
        }
    elif best_name == 'CatBoost':
        param_grid = {
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1]
        }
    elif best_name == 'XGBoost':
        param_grid = {
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1]
        }
    else:
        print(f"  No tuning defined for {best_name}, skipping.")
        return None, None

    best_score = -1
    best_params = None
    results = []

    if best_name == 'LightGBM':
        base_params = {
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED,
            'verbose': -1,
            'n_jobs': -1,
        }
        for nl in param_grid['num_leaves']:
            for lr in param_grid['learning_rate']:
                params = {**base_params, 'num_leaves': nl, 'learning_rate': lr}
                model = LGBMClassifier(**params)
                mean_mf1, _, _ = cv_score(model, preprocessor, X, y, f"LGBM(nl={nl},lr={lr})")
                results.append({'num_leaves': nl, 'learning_rate': lr, 'mean_mf1': mean_mf1})
                if mean_mf1 > best_score:
                    best_score = mean_mf1
                    best_params = {'num_leaves': nl, 'learning_rate': lr}

    elif best_name == 'CatBoost':
        base_params = {
            'iterations': 1000,
            'auto_class_weights': 'Balanced',
            'random_seed': RANDOM_SEED,
            'verbose': 0,
            'thread_count': -1,
        }
        for depth in param_grid['depth']:
            for lr in param_grid['learning_rate']:
                params = {**base_params, 'depth': depth, 'learning_rate': lr}
                model = CatBoostClassifier(**params)
                mean_mf1, _, _ = cv_score(model, preprocessor, X, y, f"CatBoost(d={depth},lr={lr})")
                results.append({'depth': depth, 'learning_rate': lr, 'mean_mf1': mean_mf1})
                if mean_mf1 > best_score:
                    best_score = mean_mf1
                    best_params = {'depth': depth, 'learning_rate': lr}

    elif best_name == 'XGBoost':
        base_params = {
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'random_state': RANDOM_SEED,
            'tree_method': 'hist',
            'n_jobs': -1,
        }
        for md in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                params = {**base_params, 'max_depth': md, 'learning_rate': lr}
                model = XGBClassifier(**params)
                mean_mf1, _, _ = cv_score(model, preprocessor, X, y, f"XGB(md={md},lr={lr})")
                results.append({'max_depth': md, 'learning_rate': lr, 'mean_mf1': mean_mf1})
                if mean_mf1 > best_score:
                    best_score = mean_mf1
                    best_params = {'max_depth': md, 'learning_rate': lr}

    print(f"\n  Tuning results for {best_name}:")
    for r in results:
        print(f"    {r}")

    print(f"\n  Best {best_name}: params={best_params}, MF1={best_score:.4f}")
    return best_score, best_params


def main():
    print("=" * 60)
    print("ADVANCED MODEL COMPARISON - Triagegeist")
    print("=" * 60)

    # 1. Load and merge
    train, test = load_and_merge_data()

    # 2. Add features
    train = add_features(train, is_train=True)

    # 3. Prepare features/labels
    X, y, feature_cols = prepare_features_labels(train)

    # 4. Create preprocessor
    preprocessor, numeric_cols, categorical_cols = get_preprocessor(X)

    # 5. Train and evaluate each model
    results = {}

    # LightGBM
    lgbm_mean, lgbm_std, lgbm_scores = train_and_evaluate_lightgbm(preprocessor, X, y)
    if lgbm_mean is not None:
        results['LightGBM'] = {'mean_mf1': float(round(lgbm_mean, 4)), 'std_mf1': float(round(lgbm_std, 4)), 'fold_scores': [float(round(s, 4)) for s in lgbm_scores]}

    # CatBoost
    cb_mean, cb_std, cb_scores = train_and_evaluate_catboost(preprocessor, X, y)
    if cb_mean is not None:
        results['CatBoost'] = {'mean_mf1': float(round(cb_mean, 4)), 'std_mf1': float(round(cb_std, 4)), 'fold_scores': [float(round(s, 4)) for s in cb_scores]}

    # XGBoost
    xgb_mean, xgb_std, xgb_scores = train_and_evaluate_xgboost(preprocessor, X, y)
    if xgb_mean is not None:
        results['XGBoost'] = {'mean_mf1': float(round(xgb_mean, 4)), 'std_mf1': float(round(xgb_std, 4)), 'fold_scores': [float(round(s, 4)) for s in xgb_scores]}

    # RandomForest
    rf_mean, rf_std, rf_scores = train_and_evaluate_rf(preprocessor, X, y)
    if rf_mean is not None:
        results['RandomForest'] = {'mean_mf1': float(round(rf_mean, 4)), 'std_mf1': float(round(rf_std, 4)), 'fold_scores': [float(round(s, 4)) for s in rf_scores]}

    # Determine best model
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_model_name = max(valid_results, key=lambda k: valid_results[k]['mean_mf1'])
        best_model_score = valid_results[best_model_name]['mean_mf1']
        print(f"\n{'=' * 60}")
        print(f"BEST MODEL: {best_model_name} with MF1 = {best_model_score:.4f}")
        print(f"{'=' * 60}")
    else:
        print("\nNo models completed successfully.")
        return

    # 6. Tune best model
    tune_score, tune_params = tune_best_model(best_model_name, preprocessor, X, y)
    if tune_score is not None:
        results[f'{best_model_name}_tuned'] = {'mean_mf1': float(round(tune_score, 4)), 'best_params': tune_params}

    # 7. Save results
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    output_path = ARTIFACTS_DIR / 'model_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # 8. Final report
    lgbm_str = f"LGBM={results.get('LightGBM', {}).get('mean_mf1', 'N/A')}"
    cb_str = f"CatBoost={results.get('CatBoost', {}).get('mean_mf1', 'N/A')}"
    xgb_str = f"XGBoost={results.get('XGBoost', {}).get('mean_mf1', 'N/A')}"
    rf_str = f"RF={results.get('RandomForest', {}).get('mean_mf1', 'N/A')}"
    best_str = f"Best={best_model_name}={best_model_score:.4f}"
    tuned_str = ""
    if tune_params is not None:
        tuned_str = f" with params {tune_params}"

    report = f"MODEL COMPARISON: {lgbm_str}, {cb_str}, {xgb_str}, {rf_str}, {best_str}{tuned_str}"
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)

    return results


if __name__ == '__main__':
    main()
