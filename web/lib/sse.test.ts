import { describe, expect, it } from "vitest";

import { readServerSentEvents } from "./sse";

function streamOf(chunks: (string | Uint8Array)[]) {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      controller.close();
    },
  });
}

async function collect(chunks: (string | Uint8Array)[]) {
  const events = [];
  for await (const event of readServerSentEvents(streamOf(chunks))) events.push(event);
  return events;
}

describe("readServerSentEvents", () => {
  it("reads several events from one chunk", async () => {
    expect(await collect(['event: a\ndata: {"n":1}\n\nevent: b\ndata: {"n":2}\n\n'])).toEqual([
      { event: "a", data: '{"n":1}' },
      { event: "b", data: '{"n":2}' },
    ]);
  });

  it("reassembles an event split across chunks, at any point", async () => {
    const text = 'event: done\ndata: {"answer":"Hi"}\n\n';
    for (let split = 1; split < text.length; split++) {
      expect(await collect([text.slice(0, split), text.slice(split)])).toEqual([
        { event: "done", data: '{"answer":"Hi"}' },
      ]);
    }
  });

  it("handles a chunk ending in the middle of a multi-byte character", async () => {
    const bytes = new TextEncoder().encode('event: done\ndata: {"answer":"Café 🎷"}\n\n');
    const middleOfEmoji = bytes.length - 6;
    expect(await collect([bytes.slice(0, middleOfEmoji), bytes.slice(middleOfEmoji)])).toEqual([
      { event: "done", data: '{"answer":"Café 🎷"}' },
    ]);
  });

  it("skips keep-alive comments", async () => {
    expect(await collect([": keep-alive\n\nevent: a\ndata: {}\n\n: keep-alive\n\n"])).toEqual([
      { event: "a", data: "{}" },
    ]);
  });

  it("ignores an incomplete event at the end of the stream", async () => {
    expect(await collect(["event: a\ndata: {}\n\nevent: b\ndata: {"])).toEqual([{ event: "a", data: "{}" }]);
  });

  it("uses the standard default name for an event without one", async () => {
    expect(await collect(["data: hello\n\n"])).toEqual([{ event: "message", data: "hello" }]);
  });
});
