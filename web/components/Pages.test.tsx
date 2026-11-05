import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { EngineStats, ShortlistEntry } from "@/lib/api";

import { Nav } from "./Nav";
import { Shortlist } from "./Shortlist";
import { Stats } from "./Stats";

vi.mock("next/navigation", () => ({ usePathname: () => "/shortlist" }));

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function entry(overrides: Partial<ShortlistEntry> = {}): ShortlistEntry {
  return {
    event_id: "sample-002",
    name: "Lake Effect Brass Band",
    date: "2026-11-10",
    venue: "Green Door Jazz Club",
    city: "Chicago",
    price: "20-35 USD",
    ticket_url: "https://example.com/tickets/sample-002",
    source: "sample",
    note: "date night",
    saved_at: "2026-11-05T12:00:00+00:00",
    ...overrides,
  };
}

function stubFetch(...responses: (Response | Error)[]) {
  const queue = [...responses];
  const fetchMock = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(() => {
    const next = queue.shift();
    if (!next) throw new Error("No scripted response left");
    return next instanceof Response ? Promise.resolve(next) : Promise.reject(next);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("Shortlist", () => {
  it("shows saved events with their details", async () => {
    stubFetch(json({ entries: [entry()] }));
    render(<Shortlist />);

    expect(screen.getByRole("status")).toHaveTextContent("Loading");
    expect(await screen.findByRole("heading", { name: "Lake Effect Brass Band" })).toBeInTheDocument();
    expect(screen.getByText("Tue, Nov 10, 2026 · Green Door Jazz Club, Chicago · 20-35 USD")).toBeInTheDocument();
    expect(screen.getByText("date night")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Tickets" });
    expect(link).toHaveAttribute("href", "https://example.com/tickets/sample-002");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("handles events with missing details", async () => {
    stubFetch(json({ entries: [entry({ date: "TBA", venue: null, city: null, price: null, ticket_url: null, note: null })] }));
    render(<Shortlist />);

    expect(await screen.findByText("TBA")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Tickets" })).not.toBeInTheDocument();
  });

  it("doesn't link to an unsafe ticket address from outside data", async () => {
    stubFetch(json({ entries: [entry({ ticket_url: "javascript:alert(1)" })] }));
    render(<Shortlist />);

    await screen.findByRole("heading", { name: "Lake Effect Brass Band" });
    expect(screen.queryByRole("link", { name: "Tickets" })).not.toBeInTheDocument();
  });

  it("shows event text as text", async () => {
    stubFetch(json({ entries: [entry({ name: "<img src=x onerror=alert(1)>" })] }));
    const { container } = render(<Shortlist />);

    expect(await screen.findByRole("heading", { name: "<img src=x onerror=alert(1)>" })).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("points to the chat when nothing is saved", async () => {
    stubFetch(json({ entries: [] }));
    render(<Shortlist />);
    expect(await screen.findByText(/Nothing saved yet/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Ask the scout" })).toHaveAttribute("href", "/");
  });

  it("removes an event", async () => {
    const fetchMock = stubFetch(
      json({ entries: [entry(), entry({ event_id: "keep", name: "Keep Me" })] }),
      new Response(null, { status: 204 }),
    );
    render(<Shortlist />);
    await userEvent.click(await screen.findByRole("button", { name: "Remove Lake Effect Brass Band" }));

    await waitFor(() => expect(screen.queryByText("Lake Effect Brass Band")).not.toBeInTheDocument());
    expect(screen.getByText("Keep Me")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenLastCalledWith("/api/shortlist/sample-002", { method: "DELETE" });
  });

  it("treats an event that's already gone as removed", async () => {
    stubFetch(json({ entries: [entry()] }), json({ detail: "'sample-002' isn't on the shortlist." }, 404));
    render(<Shortlist />);
    await userEvent.click(await screen.findByRole("button", { name: /Remove/ }));

    expect(await screen.findByText(/Nothing saved yet/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps the event and explains if removing fails", async () => {
    stubFetch(json({ entries: [entry()] }), json({ detail: "The database can't be used." }, 500));
    render(<Shortlist />);
    await userEvent.click(await screen.findByRole("button", { name: /Remove/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The database can't be used.");
    expect(screen.getByRole("heading", { name: "Lake Effect Brass Band" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Remove/ })).toBeEnabled();
  });

  it("offers a retry when loading fails", async () => {
    stubFetch(new TypeError("offline"), json({ entries: [entry()] }));
    render(<Shortlist />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Can't reach the API server");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Lake Effect Brass Band" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("Stats", () => {
  const stats: EngineStats = {
    engine: "custom",
    questions: 3,
    errors: 1,
    avg_steps: 2.34,
    avg_tool_calls: 1.5,
    avg_seconds: 3.72,
    avg_prompt_tokens: 4874.2,
    avg_output_tokens: 278.4,
    total_cost_usd: 0.0195,
  };

  it("shows a row for each engine", async () => {
    stubFetch(json({ engines: [stats, { ...stats, engine: "langgraph", questions: 1, errors: 0 }] }));
    render(<Stats />);

    const row = (await screen.findByRole("rowheader", { name: "Custom loop" })).closest("tr")!;
    const cells = within(row).getAllByRole("cell").map((cell) => cell.textContent);
    expect(cells).toEqual(["3", "1", "2.3", "1.5", "3.7", "4,874", "278", "$0.02"]);
    expect(screen.getByRole("rowheader", { name: "LangGraph" })).toBeInTheDocument();
  });

  it("shows dashes for an engine with only failed questions", async () => {
    stubFetch(
      json({
        engines: [
          {
            ...stats,
            errors: 3,
            avg_steps: null,
            avg_tool_calls: null,
            avg_seconds: null,
            avg_prompt_tokens: null,
            avg_output_tokens: null,
            total_cost_usd: 0,
          },
        ],
      }),
    );
    render(<Stats />);

    const row = (await screen.findByRole("rowheader", { name: "Custom loop" })).closest("tr")!;
    const cells = within(row).getAllByRole("cell").map((cell) => cell.textContent);
    expect(cells).toEqual(["3", "3", "–", "–", "–", "–", "–", "$0.00"]);
  });

  it("explains what the numbers leave out", async () => {
    stubFetch(json({ engines: [stats] }));
    render(<Stats />);
    expect(await screen.findByText(/Failed questions are left out/)).toBeInTheDocument();
  });

  it("has a clear empty state", async () => {
    stubFetch(json({ engines: [] }));
    render(<Stats />);
    expect(await screen.findByText(/No questions yet/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("offers a retry when loading fails", async () => {
    stubFetch(json({ detail: "The database can't be used." }, 500), json({ engines: [stats] }));
    render(<Stats />);

    expect(await screen.findByRole("alert")).toHaveTextContent("The database can't be used.");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("table")).toBeInTheDocument();
  });
});

describe("Nav", () => {
  it("marks the current page", () => {
    render(<Nav />);
    expect(screen.getByRole("link", { name: "Shortlist" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Chat" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("link", { name: "Stats" })).toHaveAttribute("href", "/stats");
  });
});
