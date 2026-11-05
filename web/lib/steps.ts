import type { ToolStep } from "./api";

function text(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

/** A short description of a tool call, such as "Searched for jazz in Chicago". */
export function describeStep({ tool, input }: ToolStep): string {
  switch (tool) {
    case "search_events": {
      const keyword = text(input.keyword);
      const city = text(input.city);
      return `Searched for ${keyword ?? "events"}${city ? ` in ${city}` : ""}`;
    }
    case "get_event_details":
      return "Looked up event details";
    case "save_to_shortlist":
      return "Saved an event to the shortlist";
    case "list_shortlist":
      return "Checked the shortlist";
    default:
      return `Ran ${tool}`;
  }
}

/** The same description while the tool is still running, such as "Searching for jazz in Chicago". */
export function describeRunningStep({ tool, input }: Pick<ToolStep, "tool" | "input">): string {
  switch (tool) {
    case "search_events": {
      const keyword = text(input.keyword);
      const city = text(input.city);
      return `Searching for ${keyword ?? "events"}${city ? ` in ${city}` : ""}`;
    }
    case "get_event_details":
      return "Looking up event details";
    case "save_to_shortlist":
      return "Saving an event to the shortlist";
    case "list_shortlist":
      return "Checking the shortlist";
    default:
      return `Running ${tool}`;
  }
}

/** A label for replies that didn't finish normally. Null means the reply is complete. */
export function stopNotice(stopReason: string): string | null {
  switch (stopReason) {
    case "end_turn":
      return null;
    case "max_tokens":
    case "model_context_window_exceeded":
      return "This reply was cut off.";
    case "refusal":
      return "Claude declined this request.";
    case "step_limit":
      return "The agent hit its step limit before finishing.";
    default:
      return `This reply stopped early (${stopReason}).`;
  }
}

export function formatCost(cost: number | null): string | null {
  if (cost === null) return null;
  return cost < 0.01 ? "under 1¢" : `$${cost.toFixed(2)}`;
}
