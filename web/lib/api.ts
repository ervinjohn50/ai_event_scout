// Every call to the backend goes through this file. The types match the response models in
// src/event_scout/api.py and need to be changed together with them.

import { readServerSentEvents } from "./sse";

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

/** Progress reported while the agent works. Thinking and tool results stay on the server. */
export type ProgressEvent =
  | { type: "model_call"; step: number }
  | { type: "tool_start"; id: string; tool: string; input: Record<string, unknown> }
  | { type: "tool_result"; id: string; tool: string; input: Record<string, unknown>; is_error: boolean };

export interface Health {
  status: "ok";
  data_source: string;
  model: string;
  available_engines: EngineName[];
  tracing: boolean;
}

export interface ShortlistEntry {
  event_id: string;
  name: string;
  date: string;
  venue: string | null;
  city: string | null;
  price: string | null;
  ticket_url: string | null;
  source: string;
  note: string | null;
  saved_at: string;
}

export interface ShortlistResponse {
  entries: ShortlistEntry[];
}

export interface EngineStats {
  engine: string;
  questions: number;
  errors: number;
  avg_steps: number | null;
  avg_tool_calls: number | null;
  avg_seconds: number | null;
  avg_prompt_tokens: number | null;
  avg_output_tokens: number | null;
  total_cost_usd: number;
  unpriced_questions: number;
}

export interface StatsResponse {
  engines: EngineStats[];
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
const CONNECTION_LOST = "The connection to the server was lost before the answer arrived.";

async function fetchOk(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(UNREACHABLE, 0);
  }
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return response;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetchOk(path, init);
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

/** Asks a question, calling `onProgress` as the agent works, and returns the final answer. */
export async function sendMessage(
  message: string,
  conversation: { sessionId: string | null; engine: EngineName },
  onProgress: (event: ProgressEvent) => void,
): Promise<ChatResponse> {
  const response = await fetchOk("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      engine: conversation.engine,
      session_id: conversation.sessionId,
    }),
  });
  if (!response.body) throw new ApiError(CONNECTION_LOST, 0);

  try {
    for await (const { event, data } of readServerSentEvents(response.body)) {
      const payload = JSON.parse(data);
      if (event === "done") return payload as ChatResponse;
      if (event === "error") throw new ApiError(payload.detail, payload.status);
      onProgress({ type: event, ...payload } as ProgressEvent);
    }
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(CONNECTION_LOST, 0);
  }
  throw new ApiError(CONNECTION_LOST, 0); // the stream ended without an answer
}

export function endConversation(sessionId: string): Promise<void> {
  return request<void>(`/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

export function getShortlist(signal?: AbortSignal): Promise<ShortlistResponse> {
  return request<ShortlistResponse>("/api/shortlist", { signal });
}

export function removeFromShortlist(eventId: string): Promise<void> {
  return request<void>(`/api/shortlist/${encodeURIComponent(eventId)}`, { method: "DELETE" });
}

export function getStats(signal?: AbortSignal): Promise<StatsResponse> {
  return request<StatsResponse>("/api/stats", { signal });
}
