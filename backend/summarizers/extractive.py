"""A compact, dependency-free TextRank extractive summarizer.

TextRank is a useful portfolio baseline because it makes no training assumptions:
sentences vote for semantically similar sentences in a graph, and PageRank turns
those votes into an importance score.  The implementation deliberately uses only
the Python standard library so the baseline is transparent and easy to inspect.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(]*[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself yourselves
    """.split()
)


@dataclass(frozen=True, slots=True)
class RankedSentence:
    """A sentence and its TextRank score."""

    index: int
    text: str
    score: float


def sentence_tokenize(text: str) -> list[str]:
    """Split prose into sentences without requiring downloadable tokenizer data."""

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(normalized) if part.strip()]


def _lemma(token: str) -> str:
    """Apply conservative English morphological normalization.

    Full dictionary lemmatizers require a separate language-data download.  These
    rules cover common plural and verb endings while keeping this local MVP
    deterministic and useful offline.
    """

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] and root[-1] not in "lsz" else root
    if len(token) > 4 and token.endswith("ed"):
        root = token[:-2]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] and root[-1] not in "lsz" else root
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def preprocess_sentence(sentence: str) -> list[str]:
    """Lowercase, remove stopwords, and lightly lemmatize one sentence."""

    return [
        _lemma(token)
        for raw in _WORD.findall(sentence.lower())
        if (token := raw.strip("'-")) and token not in _STOPWORDS
    ]


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    shared = left.keys() & right.keys()
    numerator = sum(left[word] * right[word] for word in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _similarity_graph(sentences: list[str]) -> list[list[float]]:
    vectors = [Counter(preprocess_sentence(sentence)) for sentence in sentences]
    graph = [[0.0 for _ in sentences] for _ in sentences]
    for left_index, left in enumerate(vectors):
        for right_index in range(left_index + 1, len(vectors)):
            similarity = _cosine_similarity(left, vectors[right_index])
            graph[left_index][right_index] = similarity
            graph[right_index][left_index] = similarity
    return graph


def _pagerank(
    graph: list[list[float]],
    damping: float = 0.85,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> list[float]:
    """Rank a weighted sentence graph with the PageRank power method."""

    size = len(graph)
    if size == 0:
        return []
    scores = [1.0 / size] * size
    outgoing = [sum(row) for row in graph]

    for _ in range(max_iterations):
        next_scores = [(1.0 - damping) / size for _ in range(size)]
        dangling_mass = sum(scores[index] for index, total in enumerate(outgoing) if total == 0)
        dangling_share = damping * dangling_mass / size
        for target in range(size):
            vote = sum(
                scores[source] * graph[source][target] / outgoing[source]
                for source in range(size)
                if outgoing[source] > 0
            )
            next_scores[target] += damping * vote + dangling_share
        if sum(abs(next_scores[i] - scores[i]) for i in range(size)) < tolerance:
            return next_scores
        scores = next_scores
    return scores


def summarize_extractive(text: str, target_words: int = 90) -> str:
    """Return top-ranked source sentences, restored to their original order.

    ``target_words`` is a soft budget: complete sentences are preferred over an
    abrupt word-level cutoff, but at least one sentence is always returned.
    """

    if target_words < 1:
        raise ValueError("target_words must be at least 1")
    sentences = sentence_tokenize(text)
    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0]

    # A requested budget equal to (or larger than) the source would otherwise
    # select every sentence and look like a no-op. Keep multi-sentence summaries
    # meaningfully compressed while continuing to treat the user's value as the
    # upper bound and preserving complete sentences.
    source_words = len(text.split())
    effective_budget = min(target_words, max(1, int(source_words * 0.6)))

    scores = _pagerank(_similarity_graph(sentences))
    ranked = sorted(
        (RankedSentence(index, sentence, scores[index]) for index, sentence in enumerate(sentences)),
        key=lambda item: (-item.score, item.index),
    )
    selected: list[RankedSentence] = []
    word_count = 0
    for candidate in ranked:
        candidate_words = len(candidate.text.split())
        if selected and word_count + candidate_words > effective_budget:
            continue
        selected.append(candidate)
        word_count += candidate_words
        if word_count >= effective_budget:
            break

    return " ".join(item.text for item in sorted(selected, key=lambda item: item.index))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Summarize text with TextRank.")
    parser.add_argument("text", help="Text to summarize")
    parser.add_argument("--length", type=int, default=90, help="Target word count")
    args = parser.parse_args()
    print(summarize_extractive(args.text, args.length))


if __name__ == "__main__":
    _main()
