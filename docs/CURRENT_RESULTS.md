# Current Results

## Latest Cross-Validation Snapshot

Source: `artifacts/metrics.json`

- Macro-F1: `0.9317`
- High-risk recall (`acuity <= 2`): `0.9845`
- Severe undertriage rate (`pred - true >= 2`): `0.0013`

## What This Means

- The structured model is already strong enough to anchor a serious submission.
- The current story is strongest when framed around safety support, not raw classification alone.
- The remaining edge will come from writeup quality, undertriage case analysis, and subgroup audit clarity.

## Current Candidate Submission

- Full-train candidate exported to `artifacts/submission.csv`
- Model bundle saved to `artifacts/models/ensemble.joblib`
- Current structured/text blend weight: `0.8`

Current submission label distribution (fresh run):

- Acuity `1`: `749`
- Acuity `2`: `3386`
- Acuity `3`: `7222`
- Acuity `4`: `5804`
- Acuity `5`: `2839`

## Strong Subgroup Snapshots

- `language = Estonian`: Macro-F1 refreshed
- `arrival_mode = walk-in`: Macro-F1 refreshed
- `site_id = SITE-TMP-01`: Macro-F1 refreshed

## Watch List

- `site_id = SITE-OUL-01`: Macro-F1 refreshed
- `language = Russian`: Macro-F1 refreshed
- `arrival_mode = transfer`: Macro-F1 refreshed

## Candidate Undertriage Cases

Good writeup examples from `artifacts/tables/undertriage_examples.csv`:

- `severe malaria, constant`
- `pleuritic chest pain, onset today`
- `palpitations with near-syncope`
