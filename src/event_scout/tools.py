"""The tools the agent can use, and the code that runs them.

How tool use works:
  1. Each request sends Claude a list of tool *definitions*: a name, a description,
     and a JSON Schema describing the inputs. That is ALL Claude knows about a tool.
     It never sees this Python code.
  2. When Claude wants to use a tool, its reply contains a `tool_use` block, e.g.
     {"name": "search_events", "input": {"city": "Chicago"}}.
  3. The app runs the matching function (`ToolRegistry.run`) and sends the result
     back as a `tool_result` block. Claude reads it and decides what to do next.

So a "tool" is three things: a Pydantic model for its input (which provides both
the JSON Schema and the validation), a description that tells the model when to
use it, and an ordinary Python function that does the work.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .events import EventSource, EventSourceError
from .shortlist import ShortlistError, ShortlistStore


class ToolError(Exception):
    """An expected tool failure (e.g. unknown event ID). Its message is shown to the model."""


# --- Input models ----------------------------------------------------------------------
# Each field's `description` ends up in the JSON Schema Claude sees, so they're
# written for the model: each states the expected format and gives an example.
# `extra="forbid"` rejects fields the model invents (like "town" instead of "city").


class SearchEventsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str | None = Field(
        None,
        description="Artist, team, venue, or genre to match, e.g. 'indie rock' or 'Velvet Static'. "
        "Keep it short: a name or genre, not a sentence.",
    )
    city: str | None = Field(None, description="City name, e.g. 'Chicago'. Use the common name: 'New York', not 'NYC'.")
    country_code: str | None = Field(
        None,
        description="Two-letter ISO country code, e.g. 'US' or 'GB'.",
        pattern=r"^[A-Za-z]{2}$",
    )
    start_date: date | None = Field(
        None, description="Earliest event date (YYYY-MM-DD). Defaults to today."
    )
    end_date: date | None = Field(None, description="Latest event date (YYYY-MM-DD), inclusive.")
    category: Literal["music", "sports", "arts", "film", "misc"] | None = Field(
        None, description="Broad event type. 'arts' includes theatre; 'misc' includes comedy."
    )
    max_results: int = Field(10, ge=1, le=25, description="How many events to return (1-25).")

    @model_validator(mode="after")
    def check_date_order(self) -> SearchEventsInput:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class GetEventDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(description="The event's `id` from search_events results.")


class SaveToShortlistInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(description="The event's `id` from search_events results.")
    note: str | None = Field(
        None, description="Optional short note from the user, e.g. 'go with Sam'.", max_length=200
    )


class ListShortlistInput(BaseModel):
    model_config = ConfigDict(extra="forbid")  # takes no input


# --- Tool and registry -------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Any]  # receives a validated input model; returns JSON-able data

    def definition(self) -> dict[str, Any]:
        """The exact dict sent to the API in the `tools` list."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _strip_titles(self.input_model.model_json_schema()),
        }


@dataclass(frozen=True)
class ToolOutcome:
    """What goes back to Claude in a `tool_result` block."""

    content: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def definitions(self) -> list[dict[str, Any]]:
        # Always the same tools in the same order. The tool list is part of every
        # request, and keeping it byte-identical lets the API reuse its prompt cache.
        return [tool.definition() for tool in self._tools.values()]

    def run(self, name: str, raw_input: dict[str, Any]) -> ToolOutcome:
        """Run one tool call from the model. Never raises.

        Every failure becomes an error result instead of a crash. The model reads the
        error message and usually fixes its call on the next step. That self-correction
        is a big part of what makes agents robust.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(f"Unknown tool '{name}'. Available: {', '.join(self._tools)}.", True)

        # Never trust model output: validate before running anything.
        try:
            args = tool.input_model.model_validate(raw_input)
        except ValidationError as err:
            return ToolOutcome(f"Invalid input for {name}: {_describe_validation_error(err)}", True)

        try:
            result = tool.handler(args)
        except (ToolError, EventSourceError, ShortlistError) as err:
            return ToolOutcome(str(err), True)
        except Exception as err:  # a bug in this code: report it, don't crash the agent
            return ToolOutcome(f"Internal error in {name}: {type(err).__name__}: {err}", True)

        return ToolOutcome(json.dumps(result, ensure_ascii=False, default=str))


def build_tools(source: EventSource, shortlist: ShortlistStore) -> ToolRegistry:
    """Create the agent's tools, wired to a data source and a shortlist file."""

    def search_events(args: SearchEventsInput) -> dict[str, Any]:
        events = source.search(**args.model_dump())
        result: dict[str, Any] = {"count": len(events), "events": [e.summary() for e in events]}
        # Tool results can steer the model, not just inform it.
        if not events:
            result["hint"] = (
                "No events matched. Try fewer filters: drop the keyword, widen the dates, "
                "or check the city spelling."
            )
        elif len(events) == args.max_results:
            result["note"] = (
                "Only the earliest matches are shown, because results are sorted by date. Later events "
                "may exist: raise max_results (up to 25), or narrow the dates or keyword, before "
                "saying what's available."
            )
        return result

    def get_event_details(args: GetEventDetailsInput) -> dict[str, Any]:
        event = source.get(args.event_id)
        if event is None:
            raise ToolError(f"No event with id '{args.event_id}'. Use an id from search_events.")
        return event.details()

    def save_to_shortlist(args: SaveToShortlistInput) -> dict[str, Any]:
        # A tool with a side effect (it writes a file). Saving is harmless, so it runs
        # without asking. A riskier action (buying tickets, sending an email) should
        # stop and ask the user to confirm before doing anything.
        event = source.get(args.event_id)
        if event is None:
            raise ToolError(f"No event with id '{args.event_id}'. Use an id from search_events.")
        entry, created = shortlist.add(event, args.note)
        return {
            "status": "saved" if created else "already_saved",
            "entry": entry,
            "shortlist_size": len(shortlist.all()),
        }

    def list_shortlist(args: ListShortlistInput) -> dict[str, Any]:
        entries = shortlist.all()
        return {"count": len(entries), "entries": entries}

    return ToolRegistry(
        [
            Tool(
                name="search_events",
                description=(
                    "Search upcoming concerts and live events (music, sports, arts, film, comedy). "
                    "Call this whenever the user wants to find or browse events, e.g. 'what's on in "
                    "Chicago this weekend' or 'any jazz next month'. All filters are optional and "
                    "combined with AND. Returns short summaries with an `id` for each event. Use "
                    "get_event_details for lineup, address, and ticket link. If nothing matches, "
                    "retry with fewer filters before telling the user nothing was found."
                ),
                input_model=SearchEventsInput,
                handler=search_events,
            ),
            Tool(
                name="get_event_details",
                description=(
                    "Get full details for one event: lineup, venue address, price range, on-sale "
                    "status, and ticket link. Call this when the user asks about a specific event, "
                    "or before giving details that search summaries don't include."
                ),
                input_model=GetEventDetailsInput,
                handler=get_event_details,
            ),
            Tool(
                name="save_to_shortlist",
                description=(
                    "Save an event to the user's personal shortlist, which is kept on their "
                    "computer between sessions. Only call this when the user asks to save, "
                    "bookmark, or remember an event. Saving an event again updates its note."
                ),
                input_model=SaveToShortlistInput,
                handler=save_to_shortlist,
            ),
            Tool(
                name="list_shortlist",
                description=(
                    "List the events the user has saved. Call this when the user asks what they've "
                    "saved, or mentions 'my list' or 'my shortlist'."
                ),
                input_model=ListShortlistInput,
                handler=list_shortlist,
            ),
        ]
    )


# --- Helpers -----------------------------------------------------------------------------


def _strip_titles(schema: Any) -> Any:
    """Remove the auto-generated "title" keys Pydantic adds to JSON Schemas.

    They repeat the field names ("Keyword", "City", ...), so they're noise that costs
    tokens. A property that is literally named "title" is kept: only schema keywords
    are removed, never keys inside a "properties" mapping.
    """
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "properties":
            cleaned[key] = {name: _strip_titles(sub) for name, sub in value.items()}
        else:
            cleaned[key] = _strip_titles(value)
    return cleaned


def _describe_validation_error(err: ValidationError) -> str:
    """Turn a Pydantic error into one short line the model can act on."""
    problems = []
    for e in err.errors():
        field = ".".join(str(part) for part in e["loc"]) or "input"
        problems.append(f"{field}: {e['msg']} (got {e.get('input')!r})")
    return "; ".join(problems)
