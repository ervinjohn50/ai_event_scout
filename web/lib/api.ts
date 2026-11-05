// Every call to the backend goes through this file. The types match the response models in
// src/event_scout/api.py and need to be changed together with them.

export const MAX_MESSAGE_LENGTH = 4000;

export type EngineName = "custom" | "langgraph";

export interface ToolStep {
  tool: string;
  input: Record<string, unknown>;
  is_error: boolean;
}

export interface Usage {
  input_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
}

export interface ChatResponse {
  session_id: string;
  engine: EngineName;
  answer: string;
  stop_reason: string;
  steps: number;
  tool_calls: ToolStep[];
  usage: Usage;
  cost_usd: number | null;
  seconds: number;
}

export interface Health {
  status: "ok";
  data_source: string;
  model: string;
  available_engines: EngineName[];
  tracing: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const UNREACHABLE = "Can't reach the API server. Check that it's running, then try again.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(UNREACHABLE, 0);
  }
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  let detail: unknown;
  try {
    detail = ((await response.json()) as { detail?: unknown }).detail;
  } catch {
    // Not JSON, so FastAPI didn't send it. The forwarding server couldn't reach FastAPI.
    return UNREACHABLE;
  }
  // FastAPI sends a string for errors raised on purpose, and a list for invalid requests.
  return typeof detail === "string" ? detail : "The server couldn't accept that request.";
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/api/health", { signal });
}

export function sendMessage(
  message: string,
  conversation: { sessionId: string | null; engine: EngineName },
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      engine: conversation.engine,
      session_id: conversation.sessionId,
    }),
  });
}

export function endConversation(sessionId: string): Promise<void> {
  return request<void>(`/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}
