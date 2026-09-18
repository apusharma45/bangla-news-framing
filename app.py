"""Local Gradio application for Bangla news government-framing analysis."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = APP_ROOT / "artifacts"

FINAL_LABELS = ["NR", "N", "GF", "GC"]
STANCE_LABELS = ["N", "GF", "GC"]
RELEVANCE_THRESHOLD = 0.5

FINAL_LABEL_NAMES = {
    "NR": "No relevant government framing",
    "N": "Neutral / descriptive",
    "GF": "Government-favourable",
    "GC": "Government-critical",
}

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[।!?])\s+|(?<=\.)\s+|[\r\n]+"
)
ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\ufeff")

APP_CSS = """
.gradio-container {
    font-family: "Noto Sans Bengali", "Nirmala UI", "Vrinda", Arial,
                 sans-serif !important;
    max-width: 1450px !important;
}
.result-card {
    border: 1px solid #d6d9df; border-radius: 12px; padding: 16px 20px;
    background: #f8fafc; margin: 8px 0;
}
.small-note { color: #555; font-size: 0.92rem; }
"""


def clean_bangla_text(text: str) -> str:
    """Apply the same conservative normalization used during training."""
    cleaned = html_lib.unescape(str(text))
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = cleaned.translate(ZERO_WIDTH_TRANSLATION)
    cleaned = URL_PATTERN.sub(" [URL] ", cleaned)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return torch.device(requested)


class FramingRuntime:
    """Own the two neural models and the article-level classifier."""

    def __init__(self, artifact_dir: Path, requested_device: str = "auto"):
        self.artifact_dir = artifact_dir.resolve()
        self.relevance_model_dir = self.artifact_dir / "relevance_model"
        self.stance_model_dir = self.artifact_dir / "stance_model"
        self.article_model_path = self.artifact_dir / "article_classifier.joblib"
        self.aspect_mapping_path = self.artifact_dir / "aspect_mapping.json"
        self.config_path = self.artifact_dir / "config.json"

        required = [
            self.relevance_model_dir,
            self.stance_model_dir,
            self.article_model_path,
            self.aspect_mapping_path,
            self.config_path,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Required artifacts are missing:\n{formatted}")

        with self.aspect_mapping_path.open("r", encoding="utf-8") as file:
            self.aspects = json.load(file)
        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        expected_aspect_ids = ["A1", "A2", "A3", "A4", "A5", "A6"]
        actual_aspect_ids = [aspect["aspect_id"] for aspect in self.aspects]
        if actual_aspect_ids != expected_aspect_ids:
            raise ValueError(
                f"Unexpected aspect order: {actual_aspect_ids}; "
                f"expected {expected_aspect_ids}."
            )

        self.feature_columns = [
            feature
            for aspect in self.aspects
            for feature in (
                f"{aspect['aspect_id']}_emphasis",
                f"{aspect['aspect_id']}_stance",
            )
        ]
        self.device = resolve_device(requested_device)
        self.batch_size = 32 if self.device.type == "cuda" else 8

        print(f"Loading models from {self.artifact_dir}")
        print(f"Inference device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.relevance_model_dir,
            local_files_only=True,
        )
        self.relevance_model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.relevance_model_dir,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )
        self.stance_model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.stance_model_dir,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )
        self.article_model = joblib.load(self.article_model_path)

        if self.relevance_model.config.num_labels != 2:
            raise ValueError("The relevance model must have two output labels.")
        if self.stance_model.config.num_labels != 3:
            raise ValueError("The stance model must have three output labels.")

        model_features = getattr(self.article_model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != self.feature_columns:
            raise ValueError("Article-classifier feature order does not match the UI.")

    def infer_pairs(self, sentences: str | Iterable[str]) -> pd.DataFrame:
        """Evaluate every supplied sentence against all six aspects."""
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        cleaned_sentences = [clean_bangla_text(sentence) for sentence in sentences]

        if not cleaned_sentences:
            raise ValueError("No sentences were supplied.")
        if any(not sentence for sentence in cleaned_sentences):
            raise ValueError("A sentence became empty after preprocessing.")

        records = []
        for sentence_index, (raw_sentence, cleaned_sentence) in enumerate(
            zip(sentences, cleaned_sentences)
        ):
            for aspect in self.aspects:
                records.append(
                    {
                        "sentence_index": sentence_index,
                        "sentence_text": str(raw_sentence),
                        "sentence_text_clean": cleaned_sentence,
                        "aspect_id": aspect["aspect_id"],
                        "aspect_name": aspect["aspect_name"],
                        "aspect_text": aspect["aspect_text"],
                    }
                )

        frame = pd.DataFrame(records)
        relevance_batches: list[np.ndarray] = []
        stance_batches: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(frame), self.batch_size):
                batch = frame.iloc[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch["sentence_text_clean"].tolist(),
                    batch["aspect_text"].tolist(),
                    truncation=True,
                    max_length=int(self.config.get("max_length", 128)),
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {
                    name: tensor.to(self.device)
                    for name, tensor in encoded.items()
                }
                relevance_batches.append(
                    torch.softmax(
                        self.relevance_model(**encoded).logits.float(), dim=-1
                    )
                    .cpu()
                    .numpy()
                )
                stance_batches.append(
                    torch.softmax(
                        self.stance_model(**encoded).logits.float(), dim=-1
                    )
                    .cpu()
                    .numpy()
                )

        relevance = np.concatenate(relevance_batches, axis=0)
        stance = np.concatenate(stance_batches, axis=0)
        if relevance.shape != (len(frame), 2) or stance.shape != (len(frame), 3):
            raise RuntimeError("Unexpected neural-model output dimensions.")

        frame["probability_NR"] = relevance[:, 0]
        frame["probability_relevant"] = relevance[:, 1]
        frame["conditional_probability_N"] = stance[:, 0]
        frame["conditional_probability_GF"] = stance[:, 1]
        frame["conditional_probability_GC"] = stance[:, 2]
        frame["probability_N"] = relevance[:, 1] * stance[:, 0]
        frame["probability_GF"] = relevance[:, 1] * stance[:, 1]
        frame["probability_GC"] = relevance[:, 1] * stance[:, 2]

        probability_columns = [
            "probability_NR",
            "probability_N",
            "probability_GF",
            "probability_GC",
        ]
        if not np.allclose(
            frame[probability_columns].sum(axis=1).to_numpy(), 1.0, atol=1e-5
        ):
            raise RuntimeError("The four joint probabilities do not sum to one.")

        frame["predicted_relevant"] = frame["probability_relevant"].ge(
            RELEVANCE_THRESHOLD
        )
        stance_labels = np.asarray(STANCE_LABELS)[stance.argmax(axis=1)]
        frame["predicted_conditional_stance"] = stance_labels
        frame["predicted_final_label"] = np.where(
            frame["predicted_relevant"], stance_labels, "NR"
        )
        frame["predicted_label_name"] = frame["predicted_final_label"].map(
            FINAL_LABEL_NAMES
        )
        return frame

    @staticmethod
    def derive_sentence_result(pair_frame: pd.DataFrame) -> dict[str, object]:
        """Aggregate six pair outputs into a transparent sentence summary."""
        relevant = pair_frame.loc[pair_frame["predicted_relevant"]]
        if relevant.empty:
            return {
                "label": "NR",
                "label_name": FINAL_LABEL_NAMES["NR"],
                "probability_N": 0.0,
                "probability_GF": 0.0,
                "probability_GC": 0.0,
                "relevant_aspect_count": 0,
            }

        masses = relevant[
            ["probability_N", "probability_GF", "probability_GC"]
        ].sum()
        normalized = masses / masses.sum()
        label = {
            "probability_N": "N",
            "probability_GF": "GF",
            "probability_GC": "GC",
        }[normalized.idxmax()]
        return {
            "label": label,
            "label_name": FINAL_LABEL_NAMES[label],
            "probability_N": float(normalized["probability_N"]),
            "probability_GF": float(normalized["probability_GF"]),
            "probability_GC": float(normalized["probability_GC"]),
            "relevant_aspect_count": int(len(relevant)),
        }

    @staticmethod
    def split_article(article_text: str, headline: str = "") -> list[str]:
        headline = clean_bangla_text(headline or "")
        body_sentences = []
        for part in SENTENCE_BOUNDARY_PATTERN.split(str(article_text or "")):
            cleaned = clean_bangla_text(part)
            if cleaned:
                body_sentences.append(cleaned)

        sentences = [headline] if headline else []
        for index, sentence in enumerate(body_sentences):
            if headline and index == 0 and sentence == headline:
                continue
            sentences.append(sentence)

        if not sentences:
            raise ValueError("Enter an article headline or body.")
        if len(sentences) > 500:
            raise ValueError("The article exceeds the 500-sentence demo limit.")
        return sentences

    def build_article_profile(
        self, pair_frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        profile_records = []
        feature_values: dict[str, float] = {}

        for aspect in self.aspects:
            aspect_id = aspect["aspect_id"]
            rows = pair_frame.loc[pair_frame["aspect_id"].eq(aspect_id)]
            relevance_evidence = float(rows["probability_relevant"].sum())
            emphasis = float(rows["probability_relevant"].mean())
            direction_mass = float(
                (
                    rows["probability_relevant"]
                    * (
                        rows["conditional_probability_GF"]
                        - rows["conditional_probability_GC"]
                    )
                ).sum()
            )
            stance = (
                direction_mass / relevance_evidence
                if relevance_evidence > 1e-12
                else 0.0
            )
            profile_records.append(
                {
                    "aspect_id": aspect_id,
                    "aspect_name": aspect["aspect_name"],
                    "sentence_count": int(rows["sentence_index"].nunique()),
                    "emphasis": emphasis,
                    "stance": stance,
                    "relevance_evidence": relevance_evidence,
                    "hard_relevant_count": int(rows["predicted_relevant"].sum()),
                }
            )
            feature_values[f"{aspect_id}_emphasis"] = emphasis
            feature_values[f"{aspect_id}_stance"] = stance

        profile = pd.DataFrame(profile_records)
        features = pd.DataFrame(
            [[feature_values[column] for column in self.feature_columns]],
            columns=self.feature_columns,
        )
        return profile, features

    def analyze_article(self, article_text: str, headline: str = "") -> dict:
        sentences = self.split_article(article_text, headline)
        pairs = self.infer_pairs(sentences)
        profile, features = self.build_article_profile(pairs)
        predicted_label = str(self.article_model.predict(features)[0])
        probabilities = self.article_model.predict_proba(features)[0]
        classes = getattr(self.article_model, "classes_", None)
        if classes is None:
            classes = self.article_model.named_steps["classifier"].classes_
        return {
            "sentences": sentences,
            "pairs": pairs,
            "profile": profile,
            "features": features,
            "predicted_label": predicted_label,
            "probabilities": {
                str(label): float(score)
                for label, score in zip(classes, probabilities)
            },
        }


def sentence_handler(runtime: FramingRuntime, sentence: str):
    if not str(sentence or "").strip():
        raise gr.Error("Please enter a Bangla sentence.")

    pairs = runtime.infer_pairs(sentence)
    overall = runtime.derive_sentence_result(pairs)
    summary = f"""
    <div class="result-card">
      <h3>Overall derived result: {html_lib.escape(str(overall['label_name']))}
          ({overall['label']})</h3>
      <p>Relevant aspects: <strong>{overall['relevant_aspect_count']}/6</strong></p>
      <p>Aggregated model scores — Neutral: <strong>{overall['probability_N']:.3f}</strong>,
         Government-favourable: <strong>{overall['probability_GF']:.3f}</strong>,
         Government-critical: <strong>{overall['probability_GC']:.3f}</strong></p>
      <p class="small-note">This is a transparent aggregation of six aspect-level
         outputs, not a separately trained global sentence label. Scores are raw
         model outputs and are not calibrated real-world confidence.</p>
    </div>
    """
    table = pairs[
        [
            "aspect_id",
            "aspect_name",
            "aspect_text",
            "probability_relevant",
            "probability_NR",
            "probability_N",
            "probability_GF",
            "probability_GC",
            "predicted_label_name",
        ]
    ].copy()
    table.columns = [
        "Aspect",
        "Aspect name",
        "Bangla aspect definition",
        "P(Relevant)",
        "P(NR)",
        "P(Neutral)",
        "P(Govt favourable)",
        "P(Govt critical)",
        "Final pair prediction",
    ]
    probability_columns = [
        "P(Relevant)",
        "P(NR)",
        "P(Neutral)",
        "P(Govt favourable)",
        "P(Govt critical)",
    ]
    table[probability_columns] = table[probability_columns].round(4)
    return summary, table


def comparison_plot(comparison: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Aspect emphasis",
            "Aspect stance: negative = critical, positive = favourable",
        ],
    )
    aspect_ids = comparison["aspect_id"].tolist()
    aspect_names = comparison["aspect_name"].tolist()
    for article, color in (("A", "#2563eb"), ("B", "#f97316")):
        figure.add_trace(
            go.Bar(
                x=aspect_ids,
                y=comparison[f"emphasis_{article}"],
                name=f"Article {article}",
                marker_color=color,
                customdata=aspect_names,
                hovertemplate="%{customdata}<br>Emphasis=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=aspect_ids,
                y=comparison[f"stance_{article}"],
                name=f"Article {article}",
                marker_color=color,
                showlegend=False,
                customdata=aspect_names,
                hovertemplate="%{customdata}<br>Stance=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    figure.update_yaxes(range=[0, 1], title_text="Mean relevance score", row=1, col=1)
    figure.update_yaxes(
        range=[-1, 1], title_text="Conditional stance score", zerolinewidth=2,
        row=1, col=2
    )
    figure.update_layout(
        barmode="group",
        height=480,
        margin=dict(l=40, r=30, t=70, b=40),
        legend_title_text="Article",
    )
    return figure


def article_handler(
    runtime: FramingRuntime,
    event_name: str,
    title_a: str,
    text_a: str,
    title_b: str,
    text_b: str,
):
    try:
        article_a = runtime.analyze_article(text_a, title_a)
        article_b = runtime.analyze_article(text_b, title_b)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    profile_a = article_a["profile"].rename(
        columns={
            "emphasis": "emphasis_A",
            "stance": "stance_A",
            "hard_relevant_count": "relevant_sentences_A",
        }
    )
    profile_b = article_b["profile"].rename(
        columns={
            "emphasis": "emphasis_B",
            "stance": "stance_B",
            "hard_relevant_count": "relevant_sentences_B",
        }
    )
    comparison = profile_a[
        ["aspect_id", "aspect_name", "emphasis_A", "stance_A", "relevant_sentences_A"]
    ].merge(
        profile_b[
            ["aspect_id", "emphasis_B", "stance_B", "relevant_sentences_B"]
        ],
        on="aspect_id",
        validate="one_to_one",
    )
    comparison["emphasis_difference_A_minus_B"] = (
        comparison["emphasis_A"] - comparison["emphasis_B"]
    )
    comparison["stance_difference_A_minus_B"] = (
        comparison["stance_A"] - comparison["stance_B"]
    )

    emphasis_row = comparison.loc[
        comparison["emphasis_difference_A_minus_B"].abs().idxmax()
    ]
    stance_row = comparison.loc[
        comparison["stance_difference_A_minus_B"].abs().idxmax()
    ]

    def probability_text(result: dict) -> str:
        return ", ".join(
            f"{html_lib.escape(label)}: {score:.3f}"
            for label, score in result["probabilities"].items()
        )

    summary = f"""
    <div class="result-card">
      <h3>Same-event article comparison</h3>
      <p><strong>Event:</strong> {html_lib.escape(str(event_name or 'Not specified'))}</p>
      <p><strong>Article A:</strong> {html_lib.escape(str(title_a or 'Article A'))}<br>
         Predicted class: <strong>{html_lib.escape(article_a['predicted_label'])}</strong><br>
         Model scores: {probability_text(article_a)}</p>
      <p><strong>Article B:</strong> {html_lib.escape(str(title_b or 'Article B'))}<br>
         Predicted class: <strong>{html_lib.escape(article_b['predicted_label'])}</strong><br>
         Model scores: {probability_text(article_b)}</p>
      <p>Largest emphasis difference: <strong>{emphasis_row['aspect_id']} —
         {html_lib.escape(emphasis_row['aspect_name'])}</strong>
         ({emphasis_row['emphasis_difference_A_minus_B']:+.3f}, A − B)</p>
      <p>Largest stance difference: <strong>{stance_row['aspect_id']} —
         {html_lib.escape(stance_row['aspect_name'])}</strong>
         ({stance_row['stance_difference_A_minus_B']:+.3f}, A − B)</p>
      <p class="small-note">The article classifier is exploratory: it was trained
         on 118 articles and had zero Neutral recall on the held-out article test.
         Read the aspect evidence together with the hard label.</p>
    </div>
    """

    evidence = []
    for article_name, result in (("Article A", article_a), ("Article B", article_b)):
        for aspect in runtime.aspects:
            candidates = result["pairs"].loc[
                result["pairs"]["aspect_id"].eq(aspect["aspect_id"])
            ]
            row = candidates.loc[candidates["probability_relevant"].idxmax()]
            evidence.append(
                {
                    "Article": article_name,
                    "Aspect": aspect["aspect_id"],
                    "Aspect name": aspect["aspect_name"],
                    "Highest P(Relevant)": round(float(row["probability_relevant"]), 4),
                    "Pair prediction": row["predicted_label_name"],
                    "Representative sentence": row["sentence_text_clean"],
                }
            )

    display_table = comparison.rename(
        columns={
            "aspect_id": "Aspect",
            "aspect_name": "Aspect name",
            "emphasis_A": "Emphasis A",
            "emphasis_B": "Emphasis B",
            "emphasis_difference_A_minus_B": "Emphasis difference A-B",
            "stance_A": "Stance A",
            "stance_B": "Stance B",
            "stance_difference_A_minus_B": "Stance difference A-B",
            "relevant_sentences_A": "Relevant sentences A",
            "relevant_sentences_B": "Relevant sentences B",
        }
    )
    numeric_columns = [
        "Emphasis A",
        "Emphasis B",
        "Emphasis difference A-B",
        "Stance A",
        "Stance B",
        "Stance difference A-B",
    ]
    display_table[numeric_columns] = display_table[numeric_columns].round(4)
    return summary, display_table, comparison_plot(comparison), pd.DataFrame(evidence)


def load_verified_examples(artifact_dir: Path) -> list[list[str]]:
    path = artifact_dir / "verified_ui_test_examples.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Sentence to paste" not in frame.columns:
        return []
    return [[str(sentence)] for sentence in frame["Sentence to paste"].dropna()]


def build_app(runtime: FramingRuntime) -> gr.Blocks:
    with gr.Blocks(title="Bangla News Framing Analysis") as demo:
        gr.Markdown(
            """
            # Bangla News Government-Framing Analysis

            Target-aware BanglaBERT analysis across six government-related
            aspects. The system identifies textual framing signals; it does not
            establish intentional media bias or factual correctness.
            """
        )
        gr.Markdown(
            f"**Runtime:** `{runtime.device}` · **Relevance threshold:** "
            f"`{RELEVANCE_THRESHOLD}` · **Article features:** `12`"
        )

        with gr.Tab("Single Sentence Analysis"):
            sentence_input = gr.Textbox(
                label="Bangla sentence",
                lines=4,
                placeholder="এখানে একটি বাংলা সংবাদ বাক্য লিখুন...",
            )
            sentence_button = gr.Button("Analyze sentence", variant="primary")
            sentence_summary = gr.HTML(label="Overall result")
            sentence_table = gr.Dataframe(
                label="Aspect-level results", interactive=False, wrap=True
            )
            sentence_button.click(
                fn=lambda text: sentence_handler(runtime, text),
                inputs=sentence_input,
                outputs=[sentence_summary, sentence_table],
            )

            examples = load_verified_examples(runtime.artifact_dir)
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=sentence_input,
                    label="Verified examples from untouched test events",
                )

        with gr.Tab("Compare Two Articles"):
            gr.Markdown(
                "Paste two articles that cover the same event. Supply headlines "
                "separately so each is included as one sentence."
            )
            event_input = gr.Textbox(label="Event name")
            with gr.Row():
                with gr.Column():
                    title_a = gr.Textbox(label="Article A headline")
                    text_a = gr.Textbox(label="Article A body", lines=14)
                with gr.Column():
                    title_b = gr.Textbox(label="Article B headline")
                    text_b = gr.Textbox(label="Article B body", lines=14)
            article_button = gr.Button("Compare articles", variant="primary")
            article_summary = gr.HTML(label="Article-level summary")
            article_table = gr.Dataframe(
                label="Aspect comparison", interactive=False, wrap=True
            )
            article_figure = gr.Plot(label="Visual comparison")
            evidence_table = gr.Dataframe(
                label="Representative sentences", interactive=False, wrap=True
            )
            article_button.click(
                fn=lambda event, a_title, a_text, b_title, b_text: article_handler(
                    runtime, event, a_title, a_text, b_title, b_text
                ),
                inputs=[event_input, title_a, text_a, title_b, text_b],
                outputs=[article_summary, article_table, article_figure, evidence_table],
            )

        with gr.Tab("Model Information"):
            gr.Markdown(
                """
                ### Labels
                - **NR:** No relevant government framing
                - **N:** Neutral or descriptive
                - **GF:** Government-favourable
                - **GC:** Government-critical

                ### Limitations
                - Sentence annotations are AI-assisted, not a fully
                  human-verified gold standard.
                - GF and GC are rare training classes; directional predictions
                  are preliminary.
                - The article classifier was trained on 118 articles and had
                  zero Neutral recall on the held-out article test.
                - Raw model scores can be overconfident and should not be read
                  as calibrated certainty.
                - Outputs describe textual framing patterns, not intentional
                  political bias.
                """
            )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Artifact directory (default: ./artifacts).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=os.environ.get("BNF_DEVICE", "auto"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = FramingRuntime(args.artifacts, args.device)
    demo = build_app(runtime)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
