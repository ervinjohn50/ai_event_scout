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

describe("sendMessage", () => {
  it("posts JSON, because the server refuses anything else", async () => {
    const fetchMock = stubFetch(json({ answer: "Hi" }));
    await sendMessage("hello", { sessionId: "abc", engine: "langgraph" });

    expect(fetchMock).toHaveBeenCalledWith("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello", engine: "langgraph", session_id: "abc" }),
    });
  });

  it("uses the server's message for errors raised on purpose", async () => {
    stubFetch(json({ detail: "Rate limited by the API." }, 429));
    const error = await sendMessage("hi", conversation).catch((err: unknown) => err);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ message: "Rate limited by the API.", status: 429 });
  });

  it("uses a general message for validation errors, which aren't written for people", async () => {
    stubFetch(json({ detail: [{ loc: ["body", "message"], msg: "too long" }] }, 422));
    await expect(sendMessage("hi", conversation)).rejects.toThrow("couldn't accept");
  });

  it("explains when the forwarding server can't reach FastAPI", async () => {
    stubFetch(new Response("Internal Server Error", { status: 500 }));
    await expect(sendMessage("hi", conversation)).rejects.toThrow("Can't reach the API server");
  });

  it("explains when the network request fails", async () => {
    stubFetch(new TypeError("Failed to fetch"));
    const error = await sendMessage("hi", conversation).catch((err: unknown) => err);
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
