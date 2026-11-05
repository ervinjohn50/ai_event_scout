"""An evaluation set: run the same questions through an engine and score the results.

    uv run event-scout-eval                  run every case through both engines
    uv run event-scout-eval --cases budget-filter,prompt-injection --engine langgraph
    uv run event-scout-eval --langsmith      also record experiments in LangSmith

How the results stay trustworthy:
- Every case runs against the offline sample data on a pinned date, with its own empty
  shortlist, so results are reproducible and one case can't affect another.
- Grading is plain code, with no AI judge, so the same answer always gets the same grade.
- Where possible, checks look at the outcome (what's in the shortlist afterward) rather
  than the path (which calls were made), so any valid approach can pass.
- API errors, cut-off replies, and refusals are counted separately from the pass rate.
  They aren't evidence that an answer was wrong.
- Both engines send identical requests, so whichever runs second reuses the first one's
  prompt cache and looks cheaper and faster. The order alternates per case to balance
  that, and the report adds a cost that ignores the cache, for comparing engines.
- Each run's full conversation is saved to evals/results/ so a surprising grade can be
  checked by reading what actually happened.

The test suite grades a known-good answer for every case (which must pass) and an empty
answer (which must fail), so a bug in a check doesn't go unnoticed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentObserver
from .cli import ENGINES, build_agent, has_credentials_for, load_dotenv, LANGCHAIN_MISSING_TEXT, NO_CREDENTIALS_TEXT
from .config import DEFAULT_MODEL, EFFORT_LEVELS, PRICING, AgentConfig
from .events.sample import DEFAULT_SAMPLE_PATH, SampleEventSource
from .prompts import build_system_prompt
from .shortlist import ShortlistStore
from .tools import ToolOutcome, build_tools
from .tracing import configure_tracing, flush_traces
from .usage import TokenUsage

# Every run pretends it's this day, so "this weekend" always means the same dates.
EVAL_TODAY = date(2026, 11, 3)
DEFAULT_DATASET = Path("evals") / "dataset.json"
DEFAULT_RESULTS_DIR = Path("evals") / "results"
DEFAULT_EVAL_EFFORT = "low"  # cheaper, and applied equally to both engines


# --- The dataset -------------------------------------------------------------------------------


class ShortlistExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    note_contains: str | None = None


class Case(BaseModel):
    """One test: what to ask, and what must be true afterward. Misspelled fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tags: list[str] = Field(default_factory=list)
    turns: list[str] = Field(min_length=1)  # more than one tests a follow-up question
    setup_shortlist: list[str] = Field(default_factory=list)

    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    must_mention: list[str] = Field(default_factory=list)  # all of these
    must_mention_any: list[str] = Field(default_factory=list)  # at least one of these
    must_not_mention: list[str] = Field(default_factory=list)
    # {name: words}: a name may appear only if one of the words is close by. "Marigold Engine
    # is over budget" is helpful, but recommending it without saying so is wrong.
    flagged_mentions: dict[str, list[str]] = Field(default_factory=dict)
    asks_question: bool = False
    shortlist_contains: list[ShortlistExpectation] = Field(default_factory=list)
    shortlist_empty: bool = False


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extra_events: list[dict[str, Any]] = Field(default_factory=list)
    cases: list[Case]

    def version(self) -> str:
        """Changes whenever the questions or expectations change."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:8]


def load_dataset(path: Path = DEFAULT_DATASET) -> Dataset:
    return Dataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def build_source(dataset: Dataset, directory: Path) -> SampleEventSource:
    """The offline sample events plus the dataset's extras, on the pinned date."""
    events = json.loads(DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8")) + dataset.extra_events
    path = Path(directory) / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")
    return SampleEventSource(path=path, today=EVAL_TODAY)


# --- Running a case ------------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RunRecord:
    """Everything that happened in one run of one case."""

    case_id: str
    engine: str
    rep: int = 1
    status: str = "ok"  # "ok", or "cut_off" / "refused" / "error", which aren't graded
    error: str | None = None
    answers: list[str] = field(default_factory=list)  # one per turn
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    shortlist: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    served_models: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "ok" and all(check.passed for check in self.checks)


class _RecordingObserver(AgentObserver):
    def __init__(self, record: RunRecord) -> None:
        self.record = record

    def on_model_response(self, step: int, response: Any) -> None:
        model = getattr(response, "model", None)
        if model and model not in self.record.served_models:
            self.record.served_models.append(model)

    def on_tool_result(self, tool_use: Any, outcome: ToolOutcome) -> None:
        self.record.tool_calls.append(
            ToolCall(tool_use.name, dict(tool_use.input), outcome.content, outcome.is_error)
        )


# Given tools, a system prompt, and an observer, returns an engine to run.
EngineFactory = Callable[[Any, str, AgentObserver], Any]

_NOT_GRADED = {"max_tokens": "cut_off", "model_context_window_exceeded": "cut_off", "refusal": "refused"}


def run_case(case: Case, dataset: Dataset, engine_name: str, make_engine: EngineFactory, rep: int = 1) -> RunRecord:
    """Run one case in a fresh temporary folder, then grade it."""
    record = RunRecord(case.id, engine_name, rep)
    with tempfile.TemporaryDirectory() as directory:
        source = build_source(dataset, Path(directory))
        store = ShortlistStore(Path(directory) / "shortlist.json")
        for event_id in case.setup_shortlist:
            store.add(source.get(event_id))

        agent = make_engine(
            build_tools(source, store),
            build_system_prompt(EVAL_TODAY, source.describe()),
            _RecordingObserver(record),
        )
        started = time.perf_counter()
        try:
            for turn in case.turns:
                reply = agent.ask(turn)
                record.answers.append(reply.text)
                record.stop_reason = reply.stop_reason
                record.steps += reply.steps
                record.usage.add(reply.usage)
                if reply.stop_reason in _NOT_GRADED:
                    record.status = _NOT_GRADED[reply.stop_reason]
                    break
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError):
            raise  # affects every case, so stop now instead of repeating the failure
        except anthropic.APIError as err:
            record.status = "error"
            record.error = f"{type(err).__name__}: {err}"
        record.seconds = time.perf_counter() - started
        record.shortlist = store.all()
        record.transcript = agent.export_messages()

    if record.status == "ok":
        record.checks = grade(case, record, known_event_names(dataset))
    return record


def known_event_names(dataset: Dataset) -> list[str]:
    events = json.loads(DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8")) + dataset.extra_events
    return [event["name"] for event in events]


# --- Grading -------------------------------------------------------------------------------------


def grade(case: Case, record: RunRecord, known_events: Iterable[str] = ()) -> list[Check]:
    """Each property gets its own check, so a failure says exactly what went wrong."""
    answer = _normalize(record.answers[-1] if record.answers else "")
    called = [call.name for call in record.tool_calls]
    saved = {entry["event_id"]: entry for entry in record.shortlist}
    checks = [Check("finished", record.stop_reason == "end_turn", f"stop_reason={record.stop_reason}")]

    for tool in case.tools_called:
        checks.append(Check(f"called {tool}", tool in called, f"called: {called}"))
    for tool in case.tools_not_called:
        checks.append(Check(f"did not call {tool}", tool not in called, f"called: {called}"))
    if case.max_tool_calls is not None:
        checks.append(Check(f"at most {case.max_tool_calls} tool calls", len(called) <= case.max_tool_calls, f"called: {called}"))

    for text in case.must_mention:
        checks.append(Check(f"mentions '{text}'", _normalize(text) in answer))
    if case.must_mention_any:
        found = any(_normalize(text) in answer for text in case.must_mention_any)
        checks.append(Check(f"mentions one of {case.must_mention_any}", found))
    for text in case.must_not_mention:
        checks.append(Check(f"doesn't mention '{text}'", _normalize(text) not in answer))
    for name, words in case.flagged_mentions.items():
        checks.append(Check(f"flags '{name}' if mentioned", _is_flagged(answer, name, words)))

    if case.asks_question:
        checks.append(Check("asks a question", "?" in answer))
    for expected in case.shortlist_contains:
        entry = saved.get(expected.event_id)
        note_ok = entry is not None and (
            not expected.note_contains or _normalize(expected.note_contains) in _normalize(entry.get("note") or "")
        )
        checks.append(Check(f"saved {expected.event_id}", note_ok, f"shortlist: {list(saved)}"))
    if case.shortlist_empty:
        checks.append(Check("saved nothing", not saved, f"shortlist: {list(saved)}"))

    checks.append(_check_grounded(record, case, known_events))
    return checks


def _normalize(text: str) -> str:
    """Ignore differences that don't change the meaning: case, bold markers, curly quotes."""
    return text.replace("*", "").replace("’", "'").casefold()


def _is_flagged(answer: str, name: str, words: list[str], window: int = 150) -> bool:
    """True if `name` isn't in the answer, or every mention has a flag word close by."""
    name = _normalize(name)
    for match in re.finditer(re.escape(name), answer):
        nearby = answer[max(0, match.start() - window) : match.end() + window]
        if not any(_normalize(word) in nearby for word in words):
            return False
    return True


def _check_grounded(record: RunRecord, case: Case, known_events: Iterable[str]) -> Check:
    """Prices, links, and event names in the answers must come from tool results.

    This catches invented details, but not a fully invented event with no price or link.
    """
    results_text = "\n".join(call.result for call in record.tool_calls)
    answers = "\n".join(record.answers)
    allowed_prices = set(re.findall(r"\d+(?:\.\d+)?", " ".join(_prices_in(record) + case.turns)))
    problems = []

    for price in re.findall(r"[$£]\s?(\d+(?:\.\d+)?)", answers):
        if price not in allowed_prices:
            problems.append(f"price {price}")
    for url in re.findall(r"https?://[^\s)\]>\"'`]+", answers):
        if url.rstrip(".,;:!*") not in results_text:
            problems.append(f"link {url}")
    normalized_answers, normalized_results = _normalize(answers), _normalize(results_text)
    for name in known_events:
        if _normalize(name) in normalized_answers and _normalize(name) not in normalized_results:
            problems.append(f"event '{name}'")

    return Check("only states facts from tools", not problems, ", ".join(problems))


def _prices_in(record: RunRecord) -> list[str]:
    prices: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "price" and isinstance(item, str):
                    prices.append(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for call in record.tool_calls:
        try:
            collect(json.loads(call.result))
        except json.JSONDecodeError:
            pass  # error messages aren't JSON
    return prices


# --- Reporting -------------------------------------------------------------------------------------


@dataclass
class Summary:
    engine: str
    runs: int
    graded: int
    passed: int
    errors: int
    cut_off: int
    refused: int
    avg_steps: float
    avg_prompt_tokens: float
    avg_output_tokens: float
    avg_cost: float | None  # what was actually paid, which depends on cache hits
    avg_no_cache_cost: float | None  # the same tokens at full price, for comparing engines
    avg_seconds: float

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.graded if self.graded else None


def summarize(engine: str, records: list[RunRecord], model: str = DEFAULT_MODEL) -> Summary:
    """Recompute every number from the individual runs. Failed API calls don't count toward averages."""
    graded = [r for r in records if r.status == "ok"]
    completed = [r for r in records if r.status != "error"]
    pricing = PRICING.get(model)

    def average(values: list[float]) -> float:
        return mean(values) if values else 0.0

    costs = [r.usage.cost(pricing) for r in completed]
    no_cache_costs = [
        (r.usage.prompt_tokens * pricing.input + r.usage.output_tokens * pricing.output) / 1_000_000
        for r in completed
        if pricing
    ]
    return Summary(
        engine=engine,
        runs=len(records),
        graded=len(graded),
        passed=sum(r.passed for r in graded),
        errors=sum(r.status == "error" for r in records),
        cut_off=sum(r.status == "cut_off" for r in records),
        refused=sum(r.status == "refused" for r in records),
        avg_steps=average([r.steps for r in completed]),
        avg_prompt_tokens=average([r.usage.prompt_tokens for r in completed]),
        avg_output_tokens=average([r.usage.output_tokens for r in completed]),
        avg_cost=average(costs) if pricing else None,
        avg_no_cache_cost=average(no_cache_costs) if pricing else None,
        avg_seconds=average([r.seconds for r in completed]),
    )


def format_report(summaries: list[Summary]) -> str:
    header = (
        f"{'Engine':<13}{'Passed':>9}{'Rate':>7}{'Steps':>7}{'Prompt tok':>12}{'Output tok':>12}"
        f"{'Paid/q':>9}{'No-cache/q':>12}{'Time/q':>8}"
    )
    lines = [header, "-" * len(header)]

    def dollars(value: float | None) -> str:
        return f"${value:.4f}" if value is not None else "n/a"

    for s in summaries:
        rate = f"{s.pass_rate:.0%}" if s.pass_rate is not None else "n/a"
        lines.append(
            f"{s.engine:<13}{f'{s.passed}/{s.graded}':>9}{rate:>7}{s.avg_steps:>7.1f}"
            f"{s.avg_prompt_tokens:>12,.0f}{s.avg_output_tokens:>12,.0f}"
            f"{dollars(s.avg_cost):>9}{dollars(s.avg_no_cache_cost):>12}{f'{s.avg_seconds:.1f}s':>8}"
        )
    for s in summaries:
        if s.errors or s.cut_off or s.refused:
            lines.append(f"{s.engine}: not graded: {s.errors} API errors, {s.cut_off} cut off, {s.refused} refused")

    # For a pass rate, the 95% margin on a paired difference is roughly 1/sqrt(runs).
    graded = min((s.graded for s in summaries), default=0)
    if graded:
        lines.append(
            f"\nWith {graded} graded runs per engine, pass-rate differences under about "
            f"{100 / graded**0.5:.0f} points are within normal run-to-run variation. Use --reps for more precision."
        )
    if len(summaries) > 1:
        lines.append(
            "The engines share a prompt cache, so Paid/q and Time/q depend on run order. "
            "Compare engines with No-cache/q."
        )
    return "\n".join(lines)


def format_run(record: RunRecord, model: str = DEFAULT_MODEL) -> str:
    label = {"ok": "PASS" if record.passed else "FAIL"}.get(record.status, record.status.upper())
    cost = record.usage.cost(PRICING.get(model))
    cost_text = f"  ${cost:.4f}" if cost is not None else ""
    lines = [f"  {label:<8}{record.engine:<13}{record.case_id:<30}{record.steps:>2} steps  {record.seconds:>5.1f}s{cost_text}"]
    if record.error:
        lines.append(f"            {record.error}")
    for check in record.checks:
        if not check.passed:
            lines.append(f"            failed: {check.name}" + (f" ({check.detail})" if check.detail else ""))
    return "\n".join(lines)


def save_results(records: list[RunRecord], version: str, directory: Path = DEFAULT_RESULTS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d-%H%M%S}-{version}.json"
    rows = [{**asdict(record), "passed": record.passed} for record in records]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# --- Recording experiments in LangSmith ----------------------------------------------------------


def upload_dataset(client: Any, dataset: Dataset) -> str:
    """Create the LangSmith dataset once per version, so old experiments stay comparable."""
    name = f"event-scout-evals-{dataset.version()}"
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description="Event Scout evaluation questions")
        client.create_examples(
            dataset_name=name,
            examples=[
                {"inputs": {"case_id": case.id, "turns": case.turns}, "metadata": {"tags": case.tags}}
                for case in dataset.cases
            ],
        )
    return name


def langsmith_target(
    dataset: Dataset, engine: str, make_engine: EngineFactory, records: list[RunRecord]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The function LangSmith runs for each example. It also keeps records for the local report."""
    cases = {case.id: case for case in dataset.cases}

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        record = run_case(cases[inputs["case_id"]], dataset, engine, make_engine)
        records.append(record)
        return {
            "answer": record.answers[-1] if record.answers else "",
            "status": record.status,
            "steps": record.steps,
            "tools": [call.name for call in record.tool_calls],
            "checks": [asdict(check) for check in record.checks],
        }

    return target


def langsmith_evaluator(outputs: dict[str, Any]) -> dict[str, Any]:
    """Send the code checks as LangSmith scores. Ungraded runs get no score, not a zero."""
    if outputs["status"] != "ok":
        return {"results": []}
    checks = outputs["checks"]
    return {
        "results": [
            {"key": "passed", "score": int(all(check["passed"] for check in checks))},
            *[{"key": check["name"], "score": int(check["passed"]), "comment": check["detail"]} for check in checks],
        ]
    }


def run_langsmith_experiment(
    dataset: Dataset, engine: str, make_engine: EngineFactory, config: AgentConfig, reps: int, client: Any = None
) -> list[RunRecord]:
    from langsmith import Client, evaluate

    client = client or Client()
    records: list[RunRecord] = []
    evaluate(
        langsmith_target(dataset, engine, make_engine, records),
        data=upload_dataset(client, dataset),
        evaluators=[langsmith_evaluator],
        experiment_prefix=f"{engine}-{config.effort}",
        metadata={"engine": engine, "model": config.model, "effort": config.effort, "dataset_version": dataset.version()},
        num_repetitions=reps,
        client=client,
    )
    return records


# --- Command line ----------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="event-scout-eval", description="Score the agent on a set of test questions.")
    parser.add_argument("--engine", choices=[*ENGINES, "both"], default="both", help="which engine to test (default: both)")
    parser.add_argument("--effort", choices=EFFORT_LEVELS, default=DEFAULT_EVAL_EFFORT, help=f"default: {DEFAULT_EVAL_EFFORT}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--reps", type=int, default=1, help="runs per case; more costs more but reduces noise (default: 1)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help=f"default: {DEFAULT_DATASET}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help=f"default: {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--langsmith", action="store_true", help="record experiments in LangSmith (needs LANGSMITH_API_KEY)")
    parser.add_argument("--list", action="store_true", help="list the cases and exit, with no API calls")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, client: Any = None, chat_model_factory: Callable[[], Any] | None = None) -> int:
    """Entry point. `client` and `chat_model_factory` let tests pass in fakes."""
    load_dotenv(Path(".env"))
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)

    if args.cases:
        wanted = args.cases.split(",")
        unknown = set(wanted) - {case.id for case in dataset.cases}
        if unknown:
            print(f"Unknown case ids: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        dataset = dataset.model_copy(update={"cases": [case for case in dataset.cases if case.id in wanted]})

    if args.list:
        for case in dataset.cases:
            print(f"{case.id:<30}{case.turns[0]}")
        return 0

    engines = list(ENGINES) if args.engine == "both" else [args.engine]
    using_fakes = client is not None or chat_model_factory is not None
    if not using_fakes and not all(has_credentials_for(engine, os.environ) for engine in engines):
        print(NO_CREDENTIALS_TEXT, file=sys.stderr)
        return 2
    if args.langsmith:
        if not os.environ.get("LANGSMITH_API_KEY"):
            print("--langsmith needs LANGSMITH_API_KEY. Get a free key at https://smith.langchain.com", file=sys.stderr)
            return 2
        os.environ["LANGSMITH_TRACING"] = "true"
    tracing = configure_tracing()

    config = AgentConfig(model=args.model, effort=args.effort)
    print(
        f"Dataset {dataset.version()}: {len(dataset.cases)} cases x {args.reps} rep(s) x {len(engines)} engine(s) = "
        f"{len(dataset.cases) * args.reps * len(engines)} runs | {config.model}, effort {config.effort}"
    )

    def factory_for(engine: str) -> EngineFactory:
        def make_engine(tools: Any, system_prompt: str, observer: AgentObserver) -> Any:
            chat_model = chat_model_factory() if chat_model_factory else None
            return build_agent(engine, tools, system_prompt, config, observer, client=client, chat_model=chat_model, tracing=tracing)

        return make_engine

    records_by_engine: dict[str, list[RunRecord]] = {engine: [] for engine in engines}
    try:
        if args.langsmith:
            # LangSmith runs each experiment as a whole, so the order can't alternate here.
            for engine in engines:
                records_by_engine[engine] = run_langsmith_experiment(dataset, engine, factory_for(engine), config, args.reps)
                for record in records_by_engine[engine]:
                    print(format_run(record, config.model), flush=True)
        else:
            for rep in range(1, args.reps + 1):
                for index, case in enumerate(dataset.cases):
                    # Alternate who goes first, so neither engine always gets the warm cache.
                    order = engines if (index + rep) % 2 else list(reversed(engines))
                    for engine in order:
                        record = run_case(case, dataset, engine, factory_for(engine), rep)
                        records_by_engine[engine].append(record)
                        print(format_run(record, config.model), flush=True)
    except ImportError:
        print(LANGCHAIN_MISSING_TEXT, file=sys.stderr)
        return 2
    except anthropic.APIError as err:
        print(f"\nError: stopping the eval. {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    finally:
        if tracing:
            flush_traces()

    all_records = [record for records in records_by_engine.values() for record in records]
    summaries = [summarize(engine, records, config.model) for engine, records in records_by_engine.items()]
    path = save_results(all_records, dataset.version(), args.results_dir)
    print(f"\n{format_report(summaries)}\n\nFull conversations: {path}")
    return 0
