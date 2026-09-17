"""Evaluate both summarizers on the bundled, human-written micro benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rouge_score import rouge_scorer

from backend.summarizers.abstractive import summarize_abstractive
from backend.summarizers.extractive import summarize_extractive


ROOT = Path(__file__).resolve().parent
METRICS = ("rouge1", "rouge2", "rougeL")


@dataclass(frozen=True, slots=True)
class ScoreRow:
    method: str
    rouge1: float
    rouge2: float
    rouge_l: float


def evaluate_method(
    samples: list[dict[str, str]],
    method: str,
    summarizer: Callable[[str, int], str],
    target_words: int,
) -> ScoreRow:
    """Average ROUGE F1 values across one method and the full sample set."""

    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
    totals = {metric: 0.0 for metric in METRICS}
    for index, sample in enumerate(samples, start=1):
        candidate = summarizer(sample["article"], target_words)
        scores = scorer.score(sample["reference"], candidate)
        for metric in METRICS:
            totals[metric] += scores[metric].fmeasure
        print(f"[{method}] {index:02d}/{len(samples):02d} {sample['id']}")
    count = len(samples)
    return ScoreRow(method, totals["rouge1"] / count, totals["rouge2"] / count, totals["rougeL"] / count)


def write_results(rows: list[ScoreRow], output_dir: Path, samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "samples"])
        for row in rows:
            writer.writerow([row.method, f"{row.rouge1:.4f}", f"{row.rouge2:.4f}", f"{row.rouge_l:.4f}", samples])

    lines = [
        "# ROUGE evaluation",
        "",
        f"Macro-averaged F1 on {samples} bundled news-style samples.",
        "",
        "| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "|---|---:|---:|---:|",
        *(f"| {row.method.title()} | {row.rouge1:.4f} | {row.rouge2:.4f} | {row.rouge_l:.4f} |" for row in rows),
        "",
        "> Scores are machine-dependent only in runtime, not in model output when the same dependency and model versions are used.",
    ]
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TextRank and DistilBART with ROUGE.")
    parser.add_argument("--method", choices=("extractive", "abstractive", "both"), default="both")
    parser.add_argument("--length", type=int, default=40, help="Target words per summary")
    parser.add_argument("--data", type=Path, default=ROOT / "sample_data.json")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    samples: list[dict[str, str]] = json.loads(args.data.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError("Evaluation data is empty")

    choices = {
        "extractive": summarize_extractive,
        "abstractive": summarize_abstractive,
    }
    methods = tuple(choices) if args.method == "both" else (args.method,)
    rows = [evaluate_method(samples, method, choices[method], args.length) for method in methods]
    write_results(rows, args.output, len(samples))
    print(f"Wrote {args.output / 'results.md'} and {args.output / 'results.csv'}")


if __name__ == "__main__":
    main()
