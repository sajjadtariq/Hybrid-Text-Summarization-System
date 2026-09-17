from backend.summarizers.extractive import preprocess_sentence, sentence_tokenize, summarize_extractive


ARTICLE = (
    "Solar panels convert sunlight into electricity. "
    "Solar energy is becoming cheaper as manufacturing scales. "
    "Many cities now install solar panels on public buildings. "
    "Wind turbines also generate renewable power. "
    "Renewable energy can reduce dependence on fossil fuels."
)


def test_sentence_tokenizer_preserves_sentences() -> None:
    assert len(sentence_tokenize(ARTICLE)) == 5


def test_preprocessing_removes_stopwords_and_normalizes() -> None:
    assert preprocess_sentence("The cities are installing panels.") == ["city", "install", "panel"]


def test_summary_is_extractive_and_respects_soft_budget() -> None:
    summary = summarize_extractive(ARTICLE, target_words=16)
    source_sentences = sentence_tokenize(ARTICLE)
    assert summary
    assert all(sentence in ARTICLE for sentence in sentence_tokenize(summary))
    assert any(sentence in summary for sentence in source_sentences)


def test_empty_text_returns_empty_summary() -> None:
    assert summarize_extractive("   ") == ""


def test_large_budget_still_compresses_multi_sentence_input() -> None:
    summary = summarize_extractive(ARTICLE, target_words=250)
    assert summary != ARTICLE
    assert len(summary.split()) < len(ARTICLE.split())
