"use client";

import { createContext, type ReactNode, useContext, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type ChatResponse,
  type EngineName,
  endConversation,
  getHealth,
  type Health,
  type ProgressEvent,
  sendMessage,
} from "@/lib/api";

export type ChatItem =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "assistant"; reply: ChatResponse }
  | { id: number; kind: "error"; text: string };

export interface LiveStep {
  id: string;
  tool: string;
  input: Record<string, unknown>;
  status: "running" | "done" | "failed";
}

type NewChatItem = { kind: "user"; text: string } | { kind: "assistant"; reply: ChatResponse } | { kind: "error"; text: string };

interface ChatState {
  items: ChatItem[];
  draft: string;
  setDraft: (draft: string) => void;
  engine: EngineName;
  sending: boolean;
  liveSteps: LiveStep[];
  health: Health | null;
  healthError: string | null;
  send: () => Promise<void>;
  startNewChat: (engine?: EngineName) => void;
}

const ChatContext = createContext<ChatState | null>(null);

const HEALTH_RETRY_MS = 3000;

/**
 * Holds the conversation. React clears a component's state when it leaves the screen, so if
 * this state lived in the chat page, visiting another page would lose the conversation. This
 * provider sits in the layout, which stays mounted while moving between pages. A question
 * still in progress also finishes normally while another page is showing.
 */
export function ChatProvider({
  children,
  healthRetryMs = HEALTH_RETRY_MS,
}: {
  children: ReactNode;
  healthRetryMs?: number;
}) {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [draft, setDraft] = useState("");
  const [engine, setEngine] = useState<EngineName>("custom");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [liveSteps, setLiveSteps] = useState<LiveStep[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const nextId = useRef(1);

  useEffect(() => {
    // FastAPI can take a moment longer to start than this page, so a failed check is retried
    // until it succeeds. Otherwise the warning would stay up after the server is ready.
    const controller = new AbortController();
    let retry: ReturnType<typeof setTimeout> | undefined;
    function check() {
      getHealth(controller.signal).then(
        (result) => {
          setHealth(result);
          setHealthError(null);
        },
        (err: unknown) => {
          if (!(err instanceof ApiError)) return; // the check was cancelled
          setHealthError(err.message);
          retry = setTimeout(check, healthRetryMs);
        },
      );
    }
    check();
    return () => {
      controller.abort();
      clearTimeout(retry);
    };
  }, [healthRetryMs]);

  function addItem(item: NewChatItem) {
    const id = nextId.current++;
    setItems((current) => [...current, { ...item, id }]);
  }

  function startNewChat(nextEngine: EngineName = engine) {
    // Freeing the conversation on the server is only tidying up. If this fails, the server
    // drops it later anyway, so there's nothing to tell the user.
    if (sessionId) endConversation(sessionId).catch(() => {});
    setSessionId(null);
    setItems([]);
    setEngine(nextEngine);
  }

  function handleProgress(event: ProgressEvent) {
    if (event.type === "tool_start") {
      const { id, tool, input } = event;
      setLiveSteps((steps) => [...steps, { id, tool, input, status: "running" }]);
    } else if (event.type === "tool_result") {
      const status = event.is_error ? "failed" : "done";
      setLiveSteps((steps) => steps.map((step) => (step.id === event.id ? { ...step, status } : step)));
    }
  }

  async function send() {
    const message = draft.trim();
    if (!message || sending) return;

    addItem({ kind: "user", text: message });
    setDraft("");
    setSending(true);
    setLiveSteps([]);
    try {
      const reply = await sendMessage(message, { sessionId, engine }, handleProgress);
      setSessionId(reply.session_id);
      addItem({ kind: "assistant", reply });
    } catch (err) {
      // A failed question ends its conversation on the server, so the next message starts a
      // new one. The text goes back in the box, so it can be sent again.
      setSessionId(null);
      setDraft(message);
      const reason = err instanceof ApiError ? err.message : "Something went wrong.";
      addItem({ kind: "error", text: `${reason} Your next message starts a new conversation.` });
    } finally {
      setSending(false);
      setLiveSteps([]);
    }
  }

  return (
    <ChatContext.Provider
      value={{ items, draft, setDraft, engine, sending, liveSteps, health, healthError, send, startNewChat }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat(): ChatState {
  const state = useContext(ChatContext);
  if (!state) throw new Error("useChat must be used inside ChatProvider");
  return state;
}
