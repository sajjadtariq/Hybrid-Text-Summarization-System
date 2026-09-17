from typing import Any

import torch

from backend.summarizers.abstractive import _ModelBundle, _generate, model_status, summarize_abstractive


class _FakeTokenizer:
    model_max_length = 8

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        verbose: bool = True,
    ) -> list[int]:
        assert verbose is False
        return [10, 11, 12]

    def build_inputs_with_special_tokens(self, chunk: list[int]) -> list[int]:
        return [0, *chunk, 2]

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        assert kwargs["clean_up_tokenization_spaces"] is True
        return "A short summary."


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        assert kwargs["input_ids"].ndim == 2
        assert kwargs["input_ids"].shape[0] == 1
        assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
        return torch.tensor([[20, 21]])


def test_empty_text_does_not_load_model() -> None:
    assert summarize_abstractive("   ") == ""
    assert model_status()["loaded"] is False


def test_model_status_has_local_configuration() -> None:
    status = model_status()
    assert isinstance(status["model"], str)
    assert isinstance(status["offline"], bool)


def test_generation_adds_required_batch_dimension() -> None:
    bundle = _ModelBundle(tokenizer=_FakeTokenizer(), model=_FakeModel(), device=torch.device("cpu"))
    assert _generate("Source text.", 20, bundle) == "A short summary."


def test_long_text_is_summarized_then_consolidated() -> None:
    class LongTokenizer(_FakeTokenizer):
        def encode(
            self,
            text: str,
            add_special_tokens: bool = False,
            verbose: bool = True,
        ) -> list[int]:
            assert verbose is False
            return list(range(13)) if text == "long source" else [10, 11, 12]

    model = _FakeModel()
    bundle = _ModelBundle(tokenizer=LongTokenizer(), model=model, device=torch.device("cpu"))
    assert _generate("long source", 20, bundle) == "A short summary."
    assert model.calls == 4  # three map summaries plus one final reduce pass
