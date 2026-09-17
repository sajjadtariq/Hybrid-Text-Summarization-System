from backend.summarizers.abstractive import (
    _generation_limits,
    _looks_degenerate,
    _normalize_model_input,
)


def test_markdown_notes_are_normalized_for_bart() -> None:
    source = "**Main title**\n\n***1. First cause***\n\n* A supporting point"
    normalized = _normalize_model_input(source)
    assert "*" not in normalized
    assert "Main title." in normalized
    assert "1. First cause." in normalized
    assert "A supporting point." in normalized


def test_repetitive_generation_is_rejected() -> None:
    broken = "The people of the region. The people of the region. The people of the region."
    assert _looks_degenerate(broken, target_words=40)


def test_coherent_generation_is_accepted() -> None:
    summary = (
        "Political, economic, and linguistic disputes increased tensions between East and West Pakistan. "
        "Demands for regional autonomy and unequal representation eventually contributed to separation."
    )
    assert not _looks_degenerate(summary, target_words=40)


def test_generation_cannot_stop_far_below_requested_detail() -> None:
    minimum, maximum = _generation_limits(100)
    assert minimum == 100
    assert maximum == 150
