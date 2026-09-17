"""Summarization strategies exposed by the API."""

from .abstractive import summarize_abstractive
from .extractive import summarize_extractive

__all__ = ["summarize_abstractive", "summarize_extractive"]
