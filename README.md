# triagegeist-kaggle

Judge-oriented Triagegeist competition project focused on clinically credible emergency-triage decision support.

## Project Goal

This repository packages a Kaggle submission around four ideas:

- leakage-free acuity prediction
- undertriage risk analysis
- subgroup audit
- complaint-text support

The project is intentionally framed as a second-reader safety layer rather than a generic multiclass benchmark.

## Current Local Snapshot

- Macro-F1: `0.9317`
- High-risk recall (`triage_acuity <= 2`): `0.9845`
- Severe undertriage rate (`predicted - true >= 2`): `0.0013`

Supporting summary:

- [current results](./docs/CURRENT_RESULTS.md)
- [writeup draft](./docs/WRITEUP_DRAFT.md)
- [submission status](./docs/SUBMISSION_STATUS.md)

## Repository Layout

- `notebooks/triagegeist_kaggle_ready.ipynb`: Kaggle-ready notebook candidate
- `scripts/run_pipeline.py`: local training, reporting, and artifact generation
- `scripts/fit_submission.py`: fit final models and export `submission.csv`
- `src/triagegeist/`: reusable local pipeline code
- `docs/`: writeup, cover, notebook, and submission planning docs
- `artifacts/`: generated metrics, figures, and analysis tables

## Data

The competition data is not included in this repository.

See [data/README.md](./data/README.md) for the expected file layout. Inside Kaggle, the notebook reads directly from `/kaggle/input/triagegeist`.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight local validation pass:

```bash
python scripts/run_pipeline.py --sample-size 20000 --folds 3 --skip-final-fit
```

Fit the current final model bundle and export a candidate submission:

```bash
python scripts/fit_submission.py --structured-weight 0.8
```

Export the Kaggle notebook from the `.py` source if needed:

```bash
python scripts/export_notebook.py \
  --source notebooks/triagegeist_kaggle_ready.py \
  --target notebooks/triagegeist_kaggle_ready.ipynb
```

## Main Outputs

- `artifacts/metrics.json`
- `artifacts/tables/subgroup_metrics.csv`
- `artifacts/tables/confusion_matrix.csv`
- `artifacts/tables/undertriage_examples.csv`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/figures/subgroup_macro_f1.png`
- `artifacts/submission.csv`

## Modeling Notes

- Drops `disposition` and `ed_los_hours` to avoid leakage.
- Uses a structured HistGradientBoosting model as the primary learner.
- Uses a complaint-text TF-IDF + ComplementNB model as a secondary learner.
- Blends both probability outputs for the final prediction.

## Supporting Docs

- [strategy](./STRATEGY.md)
- [notebook blueprint](./docs/NOTEBOOK_BLUEPRINT.md)
- [writeup outline](./docs/WRITEUP_OUTLINE.md)
- [submission checklist](./docs/SUBMISSION_CHECKLIST.md)
