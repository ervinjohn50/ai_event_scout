"""A record of each question asked through the web API, and reports built from it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import DEFAULT_DB_PATH, connect, transaction
from .usage import TokenUsage

# The stop_reason recorded when a question failed with an error and has no reply.
ERROR = "error"

# COUNT(*) includes every question. The averages leave out failed questions, which have no
# steps or tokens to measure and would pull the averages down. The total cost has the same
# gap: a question that fails partway may have used tokens that aren't counted here.
_STATS_BY_ENGINE = """
    SELECT
        engine,
        COUNT(*) AS questions,
        COUNT(*) FILTER (WHERE stop_reason = :error) AS errors,
        AVG(steps) FILTER (WHERE stop_reason <> :error) AS avg_steps,
        AVG(tool_calls) FILTER (WHERE stop_reason <> :error) AS avg_tool_calls,
        AVG(seconds) FILTER (WHERE stop_reason <> :error) AS avg_seconds,
        AVG(input_tokens + cache_read_tokens + cache_write_tokens)
            FILTER (WHERE stop_reason <> :error) AS avg_prompt_tokens,
        AVG(output_tokens) FILTER (WHERE stop_reason <> :error) AS avg_output_tokens,
        COALESCE(SUM(cost_usd), 0) AS total_cost_usd
    FROM question_runs
    GROUP BY engine
    ORDER BY engine
"""


@dataclass(frozen=True)
class QuestionRun:
    session_id: str
    engine: str
    model: str
    stop_reason: str
    steps: int
    tool_calls: int
    usage: TokenUsage
    cost_usd: float | None
    seconds: float


class RunLog:
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)

    def record(self, run: QuestionRun) -> None:
        with transaction(self.path) as connection:
            connection.execute(
                """
                INSERT INTO question_runs (
                    session_id, engine, model, stop_reason, steps, tool_calls, input_tokens,
                    cache_read_tokens, cache_write_tokens, output_tokens, cost_usd, seconds, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.session_id,
                    run.engine,
                    run.model,
                    run.stop_reason,
                    run.steps,
                    run.tool_calls,
                    run.usage.input_tokens,
                    run.usage.cache_read_tokens,
                    run.usage.cache_write_tokens,
                    run.usage.output_tokens,
                    run.cost_usd,
                    run.seconds,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def stats(self) -> list[dict[str, Any]]:
        """Totals and averages for each engine that has answered at least one question."""
        with connect(self.path) as connection:
            rows = connection.execute(_STATS_BY_ENGINE, {"error": ERROR}).fetchall()
        return [dict(row) for row in rows]
