# Beyond Acuity Prediction: An Interpretable Triage Support Pipeline for Undertriage Detection

Structured vitals, complaint text, and patient history for emergency triage decision support and subgroup auditing.

Public project link: [https://github.com/Rumblingb/triagegeist-kaggle](https://github.com/Rumblingb/triagegeist-kaggle)

## Clinical Problem

Emergency triage is a high-stakes decision point. Clinicians must assign urgency quickly, often with incomplete information and heavy cognitive load. In this setting, the most meaningful failure is not just general classification error, but severe undertriage: patients with clinically important warning signs being assigned a less urgent level than they should receive.

This project frames AI as a second-reader safety layer rather than an autonomous replacement for clinical judgment. The goal is to support more consistent acuity prediction, surface possible undertriage, and expose subgroup-specific failure modes that deserve further review.

## Dataset And Task

The project uses the Triagegeist competition dataset:

- `train.csv` and `test.csv` for structured intake data
- `chief_complaints.csv` for free-text complaint narratives
- `patient_history.csv` for binary comorbidity flags

All records are synthetic. That makes the dataset useful for methodological prototyping and judged comparison, but it also limits external validity. Any real deployment claim would require prospective validation on real clinical data.

The primary task is multiclass prediction of `triage_acuity` (`ESI 1-5`). A secondary goal is to inspect undertriage risk and subgroup-specific behavior, because a judged triage project should show more than a single aggregate score.

## Method

The final pipeline combines three feature families:

- structured triage variables such as vitals, demographics, arrival context, and prior utilization
- raw chief complaint text
- binary patient-history flags

I also add a small set of clinically motivated derived features, including low oxygen, fever, tachycardia, tachypnea, hypotension, abnormal GCS, elevated NEWS2, and keyword flags for presentations such as chest pain, stroke-like symptoms, respiratory distress, trauma, overdose, bleeding, pregnancy-related complaints, and severe infection.

To avoid leakage, the model explicitly excludes `disposition` and `ed_los_hours`. Those fields are post-triage outcomes and would not be available at the moment the triage decision is made.

The current ensemble uses:

- a structured tabular model as the primary learner
- a lightweight complaint-text model as a secondary learner

The structured path carries most of the predictive weight, while the complaint-text path adds targeted signal for clinically meaningful presentations that may not be fully captured by categorical complaint labels alone.

## Validation And Main Results

The local development workflow uses stratified cross-validation and emphasizes metrics that are clinically interpretable.

Latest local snapshot:

- Macro-F1: `0.9317`
- High-risk recall (`triage_acuity <= 2`): `0.9845`
- Severe undertriage rate (`predicted - true >= 2`): `0.0013`

These results suggest that the model is strong overall and especially useful as a safety-oriented support tool. The most important point is not just overall accuracy, but that high-risk recall remains strong while severe undertriage is rare.

## Undertriage Analysis

For a triage-oriented notebook, the most important error pattern is not mild confusion between adjacent ESI levels, but clinically concerning patients being scored too low. For that reason, I exported a dedicated undertriage table and reviewed candidate case studies from the most severe misses.

Current representative examples include:

- `severe malaria, constant`
- `pleuritic chest pain, onset today`
- `palpitations with near-syncope`

These examples are useful because they show the gap between strong average performance and clinically meaningful misses. In the notebook, each case is paired with complaint text, core vitals, predicted vs actual acuity, and a short clinical interpretation.

## Subgroup Audit

This submission also audits subgroup behavior across:

- `age_group`
- `sex`
- `language`
- `site_id`
- `arrival_mode`

The current results are broadly strong, but some slices are weaker than others and deserve honest discussion. Examples from the present snapshot include:

- stronger slices: `language = Estonian`, `arrival_mode = walk-in`, `site_id = SITE-TMP-01`
- weaker slices: `site_id = SITE-OUL-01`, `language = Russian`, `arrival_mode = transfer`

Because the dataset is synthetic, I present these findings as benchmark robustness signals rather than as definitive fairness claims about real patients or hospitals.

## Interpretability

For a judged clinical submission, interpretability matters because it supports credibility and clarifies failure modes. The notebook therefore focuses on clinically plausible drivers such as oxygen saturation, consciousness, respiratory rate, NEWS2 burden, and complaint-specific red flags rather than only reporting an aggregate metric table.

## Limitations

This project has several important limits:

- the dataset is synthetic
- evaluation is retrospective and offline
- real triage workflows vary across sites
- complaint phrasing and documentation style can shift across hospitals
- the model is intended for decision support, not replacement of triage staff

## Reproducibility

The public repository includes:

- the Kaggle-ready notebook
- the local training and submission scripts
- writeup and submission planning documents
- generated analysis artifacts for subgroup review and undertriage inspection

This keeps the submission reproducible and makes it straightforward to understand how the notebook, analysis, and exported submission fit together.

## Conclusion

This work is best understood as a clinically oriented triage support prototype rather than a generic multiclass benchmark. Its value comes from combining strong acuity prediction with undertriage-focused analysis, subgroup auditing, and a transparent discussion of deployment boundaries.
