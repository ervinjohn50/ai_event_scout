import type { EngineName } from "./api";

export const ENGINE_LABELS: Record<EngineName, string> = {
  custom: "Custom loop",
  langgraph: "LangGraph",
};

export function engineLabel(engine: string): string {
  return ENGINE_LABELS[engine as EngineName] ?? engine;
}

const DATE_FORMAT = new Intl.DateTimeFormat("en-US", {
  weekday: "short",
  month: "short",
  day: "numeric",
  year: "numeric",
});

/**
 * Formats a date like "2026-11-10". `new Date("2026-11-10")` would read it as midnight UTC,
 * which is still the day before in the Americas, so the parts are passed separately to get
 * that date in the viewer's own time zone.
 */
export function formatEventDate(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return value; // for example "TBA"
  const [, year, month, day] = match;
  return DATE_FORMAT.format(new Date(Number(year), Number(month) - 1, Number(day)));
}

/**
 * Returns the address only if it's a normal web link. Event data comes from outside this app,
 * and a "javascript:" address in a link would run code when clicked.
 */
export function safeUrl(value: string | null): string | null {
  if (!value) return null;
  try {
    const { protocol } = new URL(value);
    return protocol === "https:" || protocol === "http:" ? value : null;
  } catch {
    return null;
  }
}

/** A number for the stats table. Null means there's nothing to average yet. */
export function formatNumber(value: number | null, digits = 1): string {
  if (value === null) return "–";
  return value.toLocaleString("en-US", { maximumFractionDigits: digits });
}

/** A total in dollars, which keeps small amounts visible. */
export function formatDollars(value: number): string {
  if (value === 0) return "$0.00";
  return value < 0.01 ? "under $0.01" : `$${value.toFixed(2)}`;
}

/** A total cost that doesn't look complete, or free, when some questions have no known price. */
export function formatTotalCost(total: number, unpricedQuestions: number): string {
  if (unpricedQuestions === 0) return formatDollars(total);
  return total === 0 ? "Unknown" : `${formatDollars(total)} + ${unpricedQuestions} unpriced`;
}
