"""Lazy, local-only abstractive summarization with Hugging Face Transformers.

DistilBART is the default because it retains the CNN/DailyMail summarization
objective while being noticeably friendlier to CPU-only laptops than BART Large.
Set ``SUMMARIZER_MODEL=facebook/bart-large-cnn`` to use the original model.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


@dataclass(slots=True)
class _ModelBundle:
    tokenizer: Any
    model: Any
    device: Any


_bundle: _ModelBundle | None = None
_load_lock = threading.Lock()


def _load_model() -> _ModelBundle:
    """Load the model once and retain it for subsequent API requests."""

    global _bundle
    if _bundle is not None:
        return _bundle

    with _load_lock:
        if _bundle is not None:
            return _bundle
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised in deployed environment
            raise RuntimeError(
                "Abstractive dependencies are not installed. Run: pip install -r backend/requirements.txt"
            ) from exc

        model_name = os.getenv("SUMMARIZER_MODEL", DEFAULT_MODEL)
        offline = os.getenv("SUMMARIZER_OFFLINE", "0") == "1"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=offline)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _bundle = _ModelBundle(tokenizer=tokenizer, model=model, device=device)
    return _bundle


def _token_chunks(text: str, tokenizer: Any, max_input_tokens: int) -> list[list[int]]:
    # The full sequence is intentionally allowed to exceed the model limit here
    # because it is split immediately below. ``verbose=False`` prevents the
    # tokenizer from emitting a misleading overlength warning before chunking.
    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    return [token_ids[start : start + max_input_tokens] for start in range(0, len(token_ids), max_input_tokens)]


def _normalize_model_input(text: str) -> str:
    """Convert lightweight Markdown notes into clean prose for the news model."""

    normalized: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[*_#`]+", "", raw_line).strip()
        if not line:
            continue
        if raw_line.lstrip().startswith(("*", "#")) and line[-1] not in ".!?;:":
            line += "."
        normalized.append(line)
    return "\n".join(normalized)


def _generate_chunk(
    chunk: list[int],
    max_new_tokens: int,
    min_new_tokens: int,
    bundle: _ModelBundle,
) -> str:
    """Summarize one safely sized token chunk."""

    import torch

    tokenizer = bundle.tokenizer
    # Add the model's boundary tokens and an explicit batch axis. Staying in
    # token space avoids whitespace changes from a decode/re-tokenize round trip.
    prepared = tokenizer.build_inputs_with_special_tokens(chunk)
    input_ids = torch.tensor([prepared], dtype=torch.long, device=bundle.device)
    encoded = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    with torch.inference_mode():
        output = bundle.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            # A value above 1.0 prevents beam search from over-preferring very
            # short candidates when the user asks for a detailed summary.
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    return tokenizer.decode(
        output[0],
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True,
    ).strip()


def _generate(text: str, target_words: int, bundle: _ModelBundle) -> str:
    """Generate directly for short text or hierarchically for long documents."""

    tokenizer = bundle.tokenizer
    # Smaller chunks give each part of a structured long document enough
    # influence; using the absolute 1,024-token ceiling overweights the opening.
    model_limit = min(getattr(tokenizer, "model_max_length", 1024), 1024) - 2
    input_limit = min(model_limit, 768)
    chunks = _token_chunks(text, tokenizer, input_limit)
    if not chunks:
        return ""

    final_min, final_max = _generation_limits(target_words)
    if len(chunks) == 1:
        return _generate_chunk(chunks[0], final_max, final_min, bundle)

    # Map: retain enough detail from every source region. The previous code split
    # the final budget across chunks, starving each one and then concatenating
    # unrelated fragments without a consolidation pass.
    intermediate_max = max(64, min(96, final_max))
    intermediate_min = max(16, min(intermediate_max - 1, int(intermediate_max * 0.3)))
    partials = [
        _generate_chunk(chunk, intermediate_max, intermediate_min, bundle)
        for chunk in chunks
    ]
    combined = " ".join(part for part in partials if part)

    # Reduce: always synthesize multi-chunk candidates into one coherent answer.
    reduced_chunks = _token_chunks(combined, tokenizer, input_limit)
    if len(reduced_chunks) == 1:
        return _generate_chunk(reduced_chunks[0], final_max, final_min, bundle)
    return _generate(combined, target_words, bundle)


def _generation_limits(target_words: int) -> tuple[int, int]:
    """Translate a word target to a firm but safe subword-token interval.

    BART generates subword tokens rather than words. Requiring roughly one token
    per requested word gives normal English prose room for punctuation and split
    words while preventing the model from stopping at half the requested detail.
    """

    maximum = max(32, int(target_words * 1.5))
    minimum = max(12, min(maximum - 1, target_words))
    return minimum, maximum


def _looks_degenerate(summary: str, target_words: int) -> bool:
    """Detect short or repetitive generations that are worse than the baseline."""

    words = re.findall(r"[A-Za-z0-9]+", summary.lower())
    minimum_useful = max(8, min(20, target_words // 3))
    if len(words) < minimum_useful:
        return True
    bigrams = list(zip(words, words[1:]))
    repeated_ratio = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return repeated_ratio > 0.14


def summarize_abstractive(text: str, target_words: int = 90) -> str:
    """Generate a new summary locally with a cached seq2seq transformer."""

    if target_words < 1:
        raise ValueError("target_words must be at least 1")
    if not text.strip():
        return ""
    cleaned = text.strip()
    model_input = _normalize_model_input(cleaned)
    summary = _generate(model_input, target_words, _load_model())
    if _looks_degenerate(summary, target_words):
        # A conservative extractive fallback is preferable to returning broken
        # loops such as "the people ... the people ... the region".
        from .extractive import summarize_extractive

        return summarize_extractive(cleaned, target_words)
    return summary


def model_status() -> dict[str, str | bool]:
    """Expose useful local runtime status without forcing a model download."""

    return {
        "loaded": _bundle is not None,
        "model": os.getenv("SUMMARIZER_MODEL", DEFAULT_MODEL),
        "offline": os.getenv("SUMMARIZER_OFFLINE", "0") == "1",
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Summarize text with a local transformer.")
    parser.add_argument("text", help="Text to summarize")
    parser.add_argument("--length", type=int, default=90, help="Target word count")
    args = parser.parse_args()
    print(summarize_abstractive(args.text, args.length))


if __name__ == "__main__":
    _main()
