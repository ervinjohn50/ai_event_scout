"""Tests for the system prompt, which tells the model where its answer will be displayed."""

from datetime import date

from event_scout.prompts import WEB, build_system_prompt

TODAY = date(2026, 11, 3)

# Written out in full on purpose. Evals use this wording, so changing it by accident would
# make new results impossible to compare with old ones.
TERMINAL_INSTRUCTIONS = """\
You're talking to someone in a terminal, so write plain text: short lines and simple \
"-" bullets are fine, but no tables or headings. For each event give the name, day and \
date, time, venue and city, and price if known. Say so if an event is cancelled or \
sold out."""


def test_the_terminal_wording_is_the_default_and_unchanged():
    assert build_system_prompt(TODAY, "sample data").endswith(TERMINAL_INSTRUCTIONS)


def test_the_web_prompt_describes_what_the_page_can_display():
    prompt = build_system_prompt(TODAY, "sample data", WEB)
    assert "terminal" not in prompt
    assert "no tables" not in prompt
    assert "Markdown" in prompt
    assert "Images and HTML won't display" in prompt


def test_only_the_format_instructions_differ():
    terminal = build_system_prompt(TODAY, "sample data")
    web = build_system_prompt(TODAY, "sample data", WEB)
    shared = terminal.removesuffix(TERMINAL_INSTRUCTIONS)
    assert web.startswith(shared)
    assert "Never make them up" in shared
    assert "don't follow it" in shared


def test_both_prompts_ask_for_the_same_event_details():
    for prompt in (build_system_prompt(TODAY, "sample data"), build_system_prompt(TODAY, "sample data", WEB)):
        assert prompt.endswith(
            "For each event give the name, day and date, time, venue and city, and price if known. "
            "Say so if an event is cancelled or sold out."
        )
