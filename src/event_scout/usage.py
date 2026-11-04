"""Token counting and cost estimates.

Every API response reports how many tokens it used. Adding them up across the steps
of the loop shows something important: because the whole conversation is re-sent
on every step, input tokens grow with each step. Prompt caching is what keeps that
affordable: the repeated part is read from the cache at a fraction of the price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Pricing


@dataclass
class TokenUsage:
    input_tokens: int = 0  # prompt tokens NOT served from the cache
    cache_read_tokens: int = 0  # prompt tokens served from the cache (cheap)
    cache_write_tokens: int = 0  # prompt tokens written to the cache this time
    output_tokens: int = 0  # generated tokens, thinking included
    requests: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> TokenUsage:
        total = cls()
        total.add(usage)
        return total

    def add(self, usage: Any) -> None:
        """Add one API response's `usage` (or another TokenUsage) to the running total."""
        if isinstance(usage, TokenUsage):
            self.input_tokens += usage.input_tokens
            self.cache_read_tokens += usage.cache_read_tokens
            self.cache_write_tokens += usage.cache_write_tokens
            self.output_tokens += usage.output_tokens
            self.requests += usage.requests
            return
        # The cache fields can be missing or None, so default them to 0.
        self.input_tokens += usage.input_tokens or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", None) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", None) or 0
        self.output_tokens += usage.output_tokens or 0
        self.requests += 1

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def cost(self, pricing: Pricing | None) -> float | None:
        """Estimated cost in USD, or None if we don't know the model's prices."""
        if pricing is None:
            return None
        return (
            self.input_tokens * pricing.input
            + self.cache_read_tokens * pricing.cache_read
            + self.cache_write_tokens * pricing.cache_write
            + self.output_tokens * pricing.output
        ) / 1_000_000
