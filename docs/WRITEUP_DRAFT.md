# Beyond Acuity Prediction: An Interpretable Triage Support Pipeline for Undertriage Detection

Structured vitals, complaint text, and patient history for emergency triage decision support and subgroup auditing.

## Clinical Problem

Emergency triage is a compressed, high-stakes decision point. Clinicians must assign urgency rapidly, often with incomplete information and heavy cognitive load. The practical risk is not only general classification error, but severe undertriage: patients with clinically important warning signs being assigned a less urgent category than they should receive.

This project frames triage AI as a second-reader safety layer rather than an autonomous replacement for clinical judgment. The goal is to support more consistent acuity prediction, surface cases at risk of undertriage, and make failure modes visible across patient subgroups and sites.

## Dataset And Task

The project uses the Triagegeist competition dataset, which includes:

- `train.csv` and `test.csv` for structured intake variables
- `chief_complaints.csv` for free-text complaint narratives
- `patient_history.csv` for binary comorbidity indicators

All records are synthetic. That matters for interpretation: the dataset is useful for prototyping and methodological comparison, but any real deployment claim would require external and prospective validation.

The primary task is multiclass prediction of `triage_acuity` (`ESI 1-5`). The secondary analytical goal is to inspect potential undertriage and subgroup-specific performance differences.

## Method

The final pipeline combines three feature families:

- Structured triage data: vitals, demographics, arrival context, and prior utilization
- Complaint text: raw chief complaint narratives
- Patient history: binary comorbidity flags

In addition to raw columns, the pipeline adds a small set of clinically motivated derived features, including low-oxygen, fever, tachycardia, tachypnea, hypotension, abnormal GCS, high NEWS2, and text keyword flags for presentations such as chest pain, stroke-like symptoms, respiratory distress, trauma, overdose, bleeding, pregnancy-related complaints, and severe infection.

To avoid leakage, the model explicitly excludes `disposition` and `ed_los_hours`. These are post-triage outcomes and would not be available at the time a triage decision is made.

The current ensemble blends:

- A structured tabular model as the primary learner
- A lightweight complaint-text model as a secondary learner

The structured path carries most of the predictive weight, while the complaint-text path adds signal for clinically meaningful presentations that may not be fully captured by coarse complaint-system labels alone.

## Validation And Main Results

The development workflow uses stratified cross-validation and focuses on metrics that are useful for a judged clinical notebook, not only a leaderboard score.

Latest local cross-validation snapshot (refreshed Jun 1):

- Macro-F1: `0.9317`
- High-risk recall (`triage_acuity <= 2`): `0.9845`
- Severe undertriage rate (`predicted - true >= 2`): `0.0013`

These numbers suggest the model is strong overall and especially useful as a safety-oriented support tool. The most convincing aspect is not simply the aggregate score, but the fact that high-risk recall remains strong while severe undertriage is rare.

## Error Structure And Undertriage

The most important failure mode in this setting is not mild confusion between adjacent ESI levels, but cases where clinically concerning patients are scored too low. For that reason, the notebook exports a dedicated undertriage table and should highlight several case-level examples.

Candidate examples from the current artifacts include:

- `severe malaria, constant`
- `pleuritic chest pain, onset today`
- `palpitations with near-syncope`

These are useful examples because they show the difference between strong average performance and clinically meaningful misses. In the writeup, each case should be described with complaint text, core vitals, predicted vs actual acuity, and a short clinical interpretation.

## Subgroup Audit

A judge-facing clinical submission should not stop at an overall score. This project audits subgroup behavior across:

- `age_group`
- `sex`
- `language`
- `site_id`
- `arrival_mode`

Current outputs suggest broadly strong performance, with some weaker slices that are worth discussing honestly. Examples from the current snapshot include:

- Stronger slices: `language = Estonian`, `arrival_mode = walk-in`, `site_id = SITE-TMP-01`
- Weaker slices: `site_id = SITE-OUL-01`, `language = Russian`, `arrival_mode = transfer`

Because the dataset is synthetic, these findings should be presented as robustness signals inside the benchmark rather than as definitive fairness claims about real patients.

## Interpretability

The final notebook should include feature-level interpretability or at least a clear importance analysis. The main objective is to demonstrate that the model is responding to clinically plausible signals, such as oxygen saturation, consciousness, respiratory rate, NEWS2 burden, and complaint-specific red flags.

For a judged competition, interpretability is valuable for two reasons:

- It helps support clinical credibility
- It provides a more honest account of when and why the model might fail

## Limitations

This project should explicitly acknowledge the following:

- The dataset is synthetic
- The evaluation is retrospective and offline
- Triage workflow variation across hospitals can limit generalization
- Complaint phrasing and documentation style can shift across sites
- The model is decision support only and should not replace clinician judgment

## Reproducibility

The local project includes:

- a training and reporting pipeline
- a submission export script
- notebook and writeup templates
- saved artifacts for subgroup analysis and undertriage review

This supports a clean Kaggle submission workflow: run the training pipeline, export the final submission, then attach the public notebook and project materials to the Kaggle writeup.

## Conclusion

This submission is best understood as a clinically oriented triage support prototype rather than a generic multiclass classifier. Its value comes from combining strong acuity prediction with undertriage-focused analysis, subgroup auditing, and transparent discussion of deployment boundaries.
