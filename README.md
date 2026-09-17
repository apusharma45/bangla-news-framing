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
