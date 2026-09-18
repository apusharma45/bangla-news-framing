# Bangla News Framing

An interpretable NLP pipeline for comparing how Bangla news articles covering
the same event frame the government. The primary system uses target-aware
BanglaBERT models for sentence-aspect relevance and stance, followed by an
article-level logistic-regression classifier.

## Colab notebook

The training notebook is developed incrementally in
`notebooks/bangla_news_framing_training.ipynb`.

Before running it, place the completed annotation CSV at:

```text
My Drive/bangla-news-framing/data/raw/banglabias_sentence_annotations_completed.csv
```

Use a Google Colab GPU runtime. Generated models, metrics, predictions, and
figures are written under `My Drive/bangla-news-framing/artifacts` and are not
committed to Git.

The annotations are AI-assisted and are not a fully human-verified gold
standard. Results must be reported with that limitation.

## Local demonstration interface

The repository includes a local Gradio interface with two workflows:

- analyse one Bangla sentence against all six aspects;
- compare two articles about the same event using aspect profiles and the
  saved article classifier.

The interface uses only saved artifacts; the raw training dataset is not
required. Extract the runtime bundle so the repository contains:

```text
artifacts/
├── relevance_model/
├── stance_model/
├── article_classifier.joblib
├── aspect_mapping.json
└── config.json
```

On Windows, install 64-bit Python 3.12, open PowerShell in the repository, and
run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_local.ps1
.\run_local.ps1
```

The first command creates `.venv312` and installs the local requirements. The
second loads the models and opens `http://127.0.0.1:7860`. PyTorch uses a CUDA
GPU automatically when a compatible build is available; otherwise inference
runs on CPU. CPU article comparison is expected to be slower.

Verified examples from the untouched test events appear under the sentence
input when `artifacts/verified_ui_test_examples.csv` is present.
