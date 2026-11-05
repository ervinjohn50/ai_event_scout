import { describe, expect, it } from "vitest";

import { describeRunningStep, describeStep, formatCost, stopNotice } from "./steps";

const step = (tool: string, input: Record<string, unknown> = {}) => ({ tool, input, is_error: false });

describe("describeStep", () => {
  it.each([
    [step("search_events", { keyword: "jazz", city: "Chicago" }), "Searched for jazz in Chicago"],
    [step("search_events", { city: "Austin" }), "Searched for events in Austin"],
    [step("search_events"), "Searched for events"],
    [step("search_events", { city: 42 }), "Searched for events"],
    [step("get_event_details", { event_id: "x" }), "Looked up event details"],
    [step("save_to_shortlist", { event_id: "x" }), "Saved an event to the shortlist"],
    [step("list_shortlist"), "Checked the shortlist"],
    [step("new_tool"), "Ran new_tool"],
  ])("describes %j", (input, expected) => {
    expect(describeStep(input)).toBe(expected);
  });
});

describe("describeRunningStep", () => {
  it.each([
    [step("search_events", { keyword: "jazz", city: "Chicago" }), "Searching for jazz in Chicago"],
    [step("search_events"), "Searching for events"],
    [step("get_event_details"), "Looking up event details"],
    [step("save_to_shortlist"), "Saving an event to the shortlist"],
    [step("list_shortlist"), "Checking the shortlist"],
    [step("new_tool"), "Running new_tool"],
  ])("describes %j", (input, expected) => {
    expect(describeRunningStep(input)).toBe(expected);
  });
});

describe("stopNotice", () => {
  it("says nothing about a reply that finished normally", () => {
    expect(stopNotice("end_turn")).toBeNull();
  });

  it.each([
    ["max_tokens", "cut off"],
    ["model_context_window_exceeded", "cut off"],
    ["refusal", "declined"],
    ["step_limit", "step limit"],
    ["something_new", "something_new"],
  ])("explains %s", (reason, expected) => {
    expect(stopNotice(reason)).toContain(expected);
  });
});

describe("formatCost", () => {
  it.each([
    [null, null],
    [0.004, "under 1¢"],
    [0.02, "$0.02"],
    [1.5, "$1.50"],
  ])("formats %s", (cost, expected) => {
    expect(formatCost(cost)).toBe(expected);
  });
});
