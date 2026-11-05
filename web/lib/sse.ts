export interface ServerSentEvent {
  event: string;
  data: string;
}

/**
 * Reads Server-Sent Events from a response. The browser's own EventSource only supports GET,
 * and sending a question needs POST, so this reads the stream directly.
 *
 * Network chunks can end anywhere, even in the middle of a letter, so text is held in a buffer
 * until a complete event (ending in a blank line) has arrived.
 */
export async function* readServerSentEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<ServerSentEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;
      // stream: true holds back a partial multi-byte character until the rest of it arrives.
      buffer += decoder.decode(value, { stream: true });

      let end = buffer.indexOf("\n\n");
      while (end !== -1) {
        const parsed = parseEvent(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (parsed) yield parsed;
        end = buffer.indexOf("\n\n");
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}

function parseEvent(block: string): ServerSentEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice("event: ".length);
    else if (line.startsWith("data: ")) data.push(line.slice("data: ".length));
    // Anything else, such as a ": keep-alive" comment, carries no data.
  }
  return data.length > 0 ? { event, data: data.join("\n") } : null;
}
