#!/usr/bin/env python3
"""Final submission: HGB on full data with all engineered features"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

train = pd.read_csv('/root/triagegeist_data/train.csv').merge(
    pd.read_csv('/root/triagegeist_data/chief_complaints.csv'), on='patient_id'
).merge(pd.read_csv('/root/triagegeist_data/patient_history.csv'), on='patient_id')
test = pd.read_csv('/root/triagegeist_data/test.csv').merge(
    pd.read_csv('/root/triagegeist_data/chief_complaints.csv'), on='patient_id'
).merge(pd.read_csv('/root/triagegeist_data/patient_history.csv'), on='patient_id')
y = train['triage_acuity'].values

# Feature engineering
def fe(df):
    d = df.copy()
    txt = d['chief_complaint_raw'].fillna('').astype(str).str.lower()
    
    # Base flags
    d['pain_unrecorded'] = (d['pain_score'] == -1).astype(int)
    d['pain_score'] = d['pain_score'].replace(-1, np.nan)
    for c,th,n in [('spo2',92,'low_o2'),('temperature_c',38,'fever'),('heart_rate',100,'tachy'),
                    ('respiratory_rate',22,'tachyp'),('systolic_bp',90,'hypot'),('gcs_total',15,'gcs_ab'),
                    ('shock_index',0.9,'shock'),('news2_score',5,'high_news2')]:
        d[f'flag_{n}'] = (d[c] >= th if n not in ['low_o2','gcs_ab','hypot'] else (d[c] < th)).astype(int)
    
    # Text features
    d['clen'] = txt.str.len(); d['cword'] = txt.str.split().str.len()
    kw = {'chest':r'chest|thoracic|crushing','stroke':r'stroke|seizure|weakness|aphasia',
          'resp':r'shortness of breath|asthma|hypoxia|wheeze','trauma':r'trauma|fracture|stab|wound|fall',
          'sepsis':r'sepsis|fever|infection','bleed':r'bleed|hemorrhage|melena',
          'od':r'overdose|poison|toxic','preg':r'pregnan|ectopic|miscarriage'}
    for n,p in kw.items(): d[f'kw_{n}'] = txt.str.contains(p, regex=True).astype(int)
    
    # Comorbidity burdens
    for g,cl in [('cardio',['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation','hx_coronary_artery_disease','hx_peripheral_vascular_disease','hx_stroke_prior']),
                 ('resp',['hx_asthma','hx_copd']),('neuro',['hx_dementia','hx_epilepsy','hx_stroke_prior']),
                 ('frail',['hx_dementia','hx_ckd','hx_malignancy','hx_immunosuppressed'])]:
        d[f'{g}_bur'] = d[[c for c in cl if c in d.columns]].sum(axis=1)
    
    # Advanced: vital ratios
    d['ppr'] = d['pulse_pressure'] / (d['systolic_bp'] + 1e-6)
    d['mbr'] = d['mean_arterial_pressure'] / (d['systolic_bp'] + 1e-6)
    d['sf'] = d['spo2'] / 100.0
    d['hrr'] = d['heart_rate'] / (d['respiratory_rate'] + 1e-6)
    d['bo'] = d['bmi'] * d['spo2'] / 100.0
    d['hxt'] = d['heart_rate'] * d['temperature_c'] / 100.0
    
    # NEWS2 subcomponents
    d['n2r'] = (d['respiratory_rate']>20).astype(int)+(d['respiratory_rate']>24).astype(int)
    d['n2o'] = (d['spo2']<96).astype(int)+(d['spo2']<94).astype(int)+(d['spo2']<92).astype(int)
    d['n2c'] = (d['gcs_total']<15).astype(int)
    d['n2h'] = (d['heart_rate']>=91).astype(int)+(d['heart_rate']>=111).astype(int)
    
    # Age interactions
    d['ahr'] = d['age']*d['heart_rate']/100; d['arr'] = d['age']*d['respiratory_rate']/100
    d['asbp'] = d['age']*d['systolic_bp']/100; d['ash'] = d['age']*d['shock_index']
    d['an2'] = d['age']*d['news2_score']/100
    
    # Comorbidity combos
    hx = [c for c in d.columns if c.startswith('hx_')]
    d['thx'] = d[hx].sum(axis=1)
    d['cr'] = ((d['hx_heart_failure']>0)&(d['hx_ckd']>0)).astype(int)
    
    # Temporal
    d['hs'] = np.sin(2*np.pi*d['arrival_hour']/24); d['hc'] = np.cos(2*np.pi*d['arrival_hour']/24)
    d['ms'] = np.sin(2*np.pi*d['arrival_month']/12); d['mc'] = np.cos(2*np.pi*d['arrival_month']/12)
    
    return d

print("Engineering features...", flush=True)
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
print(f"Training HGB on {X.shape[0]} samples × {X.shape[1]} features...", flush=True)

# HGB on full data
from sklearn.ensemble import HistGradientBoostingClassifier
m = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=300,
                                     min_samples_leaf=50, random_state=42)
m.fit(X, y)

preds = m.predict(Xt)
print(f"Predictions: {dict(zip(*np.unique(preds, return_counts=True)))}", flush=True)

sub = pd.DataFrame({'patient_id': test['patient_id'].values, 'triage_acuity': preds})
sub.to_csv('/root/triagegeist-kaggle/submission.csv', index=False)
sub.to_csv('/root/triagegeist-kaggle/artifacts/final_submission.csv', index=False)
print(f"Submission saved: {len(sub)} predictions", flush=True)
print("DONE!", flush=True)
