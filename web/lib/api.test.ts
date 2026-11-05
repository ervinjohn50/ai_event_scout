import { describe, expect, it, vi } from "vitest";

import { ApiError, endConversation, getHealth, sendMessage } from "./api";

function stubFetch(result: Response | Error | DOMException) {
  const fetchMock = vi.fn(() => (result instanceof Response ? Promise.resolve(result) : Promise.reject(result)));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const conversation = { sessionId: null, engine: "custom" } as const;

function stream(...events: [string, unknown][]) {
  const text = events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join("");
  return new Response(text, { headers: { "Content-Type": "text/event-stream" } });
}

const ignoreProgress = () => {};

describe("sendMessage", () => {
  it("posts JSON, because the server refuses anything else", async () => {
    const fetchMock = stubFetch(stream(["done", { answer: "Hi" }]));
    await sendMessage("hello", { sessionId: "abc", engine: "langgraph" }, ignoreProgress);

    expect(fetchMock).toHaveBeenCalledWith("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello", engine: "langgraph", session_id: "abc" }),
    });
  });

  it("reports progress in order and returns the final answer", async () => {
    stubFetch(
      stream(
        ["model_call", { step: 1 }],
        ["tool_start", { id: "t1", tool: "search_events", input: {} }],
        ["tool_result", { id: "t1", tool: "search_events", input: {}, is_error: false }],
        ["done", { answer: "Found it." }],
      ),
    );
    const progress: string[] = [];
    const reply = await sendMessage("hi", conversation, (event) => progress.push(event.type));

    expect(progress).toEqual(["model_call", "tool_start", "tool_result"]);
    expect(reply.answer).toBe("Found it.");
  });

  it("turns an error event into an error, with its status", async () => {
    stubFetch(stream(["model_call", { step: 1 }], ["error", { status: 429, detail: "Rate limited by the API." }]));
    const error = await sendMessage("hi", conversation, ignoreProgress).catch((err: unknown) => err);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ message: "Rate limited by the API.", status: 429 });
  });

  it("says when the stream ends without an answer", async () => {
    stubFetch(stream(["model_call", { step: 1 }]));
    await expect(sendMessage("hi", conversation, ignoreProgress)).rejects.toThrow("connection to the server was lost");
  });

  it("says when the connection drops partway through", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('event: model_call\ndata: {"step":1}\n\n'));
        controller.error(new TypeError("network error"));
      },
    });
    stubFetch(new Response(body));
    await expect(sendMessage("hi", conversation, ignoreProgress)).rejects.toThrow("connection to the server was lost");
  });

  it("doesn't fail on a garbled event", async () => {
    stubFetch(new Response("event: done\ndata: {not json\n\n"));
    await expect(sendMessage("hi", conversation, ignoreProgress)).rejects.toBeInstanceOf(ApiError);
  });

  it("uses the server's message for errors raised on purpose", async () => {
    stubFetch(json({ detail: "Rate limited by the API." }, 429));
    const error = await sendMessage("hi", conversation, ignoreProgress).catch((err: unknown) => err);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ message: "Rate limited by the API.", status: 429 });
  });

  it("uses a general message for validation errors, which aren't written for people", async () => {
    stubFetch(json({ detail: [{ loc: ["body", "message"], msg: "too long" }] }, 422));
    await expect(sendMessage("hi", conversation, ignoreProgress)).rejects.toThrow("couldn't accept");
  });

  it("explains when the forwarding server can't reach FastAPI", async () => {
    stubFetch(new Response("Internal Server Error", { status: 500 }));
    await expect(sendMessage("hi", conversation, ignoreProgress)).rejects.toThrow("Can't reach the API server");
  });

  it("explains when the network request fails", async () => {
    stubFetch(new TypeError("Failed to fetch"));
    const error = await sendMessage("hi", conversation, ignoreProgress).catch((err: unknown) => err);
    expect(error).toMatchObject({ message: expect.stringContaining("Can't reach"), status: 0 });
  });
});

describe("other calls", () => {
  it("doesn't report a cancelled request as a server problem", async () => {
    stubFetch(new DOMException("Aborted", "AbortError"));
    const error = await getHealth().catch((err: unknown) => err);
    expect(error).not.toBeInstanceOf(ApiError);
  });

  it("handles an empty response when ending a conversation", async () => {
    const fetchMock = stubFetch(new Response(null, { status: 204 }));
    await expect(endConversation("a/b")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("/api/chat/a%2Fb", { method: "DELETE" });
  });
});
