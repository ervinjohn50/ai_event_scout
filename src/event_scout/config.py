"""Settings for the agent: which model, how hard it thinks, and the safety limits."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "claude-sonnet-5-5"

# Effort controls how much the model thinks before answering, and therefore its speed
# and cost. The API's default varies by model, so it's set explicitly here, which keeps
# the choice visible and easy to change.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"

# The most tokens one reply may use, thinking included. If a reply hits this limit
# it is cut off (stop_reason "max_tokens").
DEFAULT_MAX_TOKENS = 16_000

# The most model calls one question may take. An agent decides for itself when it's
# done, so a hard limit is needed in case it never decides (a bug, or a confusing tool).
DEFAULT_MAX_STEPS = 10

# Opt-in beta: if a safety filter wrongly declines a request, the API retries it on
# another model instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_steps: int = DEFAULT_MAX_STEPS
    use_fallbacks: bool = True


@dataclass(frozen=True)
class Pricing:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_write: float  # writing a prompt prefix into the cache costs a bit more than input
    cache_read: float  # reading it back later is much cheaper


# Used only to show cost estimates in the trace. Check the official pricing page for
# current numbers before relying on these.
PRICING: dict[str, Pricing] = {
    "claude-opus-5-5": Pricing(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    "claude-sonnet-5-5": Pricing(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-haiku-5-5": Pricing(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
}
