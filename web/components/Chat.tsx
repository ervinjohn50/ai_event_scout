"use client";

import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type ChatResponse,
  type EngineName,
  endConversation,
  getHealth,
  type Health,
  MAX_MESSAGE_LENGTH,
  sendMessage,
} from "@/lib/api";
import { describeStep, formatCost, stopNotice } from "@/lib/steps";

import { Answer } from "./Answer";
import styles from "./Chat.module.css";

type ChatItem =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "assistant"; reply: ChatResponse }
  | { id: number; kind: "error"; text: string };

const ENGINE_LABELS: Record<EngineName, string> = {
  custom: "Custom loop",
  langgraph: "LangGraph",
};

const EXAMPLES = [
  "What concerts are on in Chicago this weekend?",
  "Any comedy in New York this month?",
  "Show me what's on my shortlist.",
];

const HEALTH_RETRY_MS = 3000;

export function Chat({ healthRetryMs = HEALTH_RETRY_MS }: { healthRetryMs?: number }) {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [draft, setDraft] = useState("");
  const [engine, setEngine] = useState<EngineName>("custom");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const nextId = useRef(1);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [items, sending]);

  function addItem(item: Omit<Extract<ChatItem, { kind: "user" }>, "id"> | Omit<Extract<ChatItem, { kind: "assistant" }>, "id"> | Omit<Extract<ChatItem, { kind: "error" }>, "id">) {
    const id = nextId.current++;
    setItems((current) => [...current, { ...item, id } as ChatItem]);
  }

  function startNewChat(nextEngine: EngineName = engine) {
    // Freeing the conversation on the server is only tidying up. If this fails, the server
    // drops it later anyway, so there's nothing to tell the user.
    if (sessionId) endConversation(sessionId).catch(() => {});
    setSessionId(null);
    setItems([]);
    setEngine(nextEngine);
    inputRef.current?.focus();
  }

  async function send() {
    const message = draft.trim();
    if (!message || sending) return;

    addItem({ kind: "user", text: message });
    setDraft("");
    setSending(true);
    try {
      const reply = await sendMessage(message, { sessionId, engine });
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
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void send();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // isComposing is true while typing with an input method, where Enter confirms a word.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send();
    }
  }

  const empty = items.length === 0 && !sending;

  return (
    <div className={styles.chat}>
      <div className={styles.toolbar}>
        <label className={styles.engine}>
          <span>Engine</span>
          <select
            value={engine}
            disabled={sending}
            title="Changing the engine starts a new chat"
            onChange={(event) => startNewChat(event.target.value as EngineName)}
          >
            {(Object.keys(ENGINE_LABELS) as EngineName[]).map((name) => (
              <option key={name} value={name} disabled={health ? !health.available_engines.includes(name) : false}>
                {ENGINE_LABELS[name]}
              </option>
            ))}
          </select>
        </label>
        {health && (
          <span className={styles.status}>
            {health.data_source === "ticketmaster" ? "Live Ticketmaster data" : "Sample data"}
          </span>
        )}
        <button type="button" className={styles.newChat} disabled={sending || empty} onClick={() => startNewChat()}>
          New chat
        </button>
      </div>

      {healthError && (
        <p className={styles.banner} role="alert">
          {healthError}
        </p>
      )}

      <div className={styles.messages} role="log" aria-label="Conversation">
        {empty && (
          <div className={styles.welcome}>
            <h2>Find something to do.</h2>
            <p>Ask about concerts and live events, then save the ones you like.</p>
            <div className={styles.examples}>
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => {
                    setDraft(example);
                    inputRef.current?.focus();
                  }}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {items.map((item) => {
          if (item.kind === "user") {
            return (
              <div key={item.id} className={styles.user}>
                {item.text}
              </div>
            );
          }
          if (item.kind === "error") {
            return (
              <p key={item.id} className={styles.error} role="alert">
                {item.text}
              </p>
            );
          }
          return <AssistantMessage key={item.id} reply={item.reply} />;
        })}

        {sending && (
          <p className={styles.working} role="status">
            Working on it…
          </p>
        )}
        <div ref={endRef} />
      </div>

      <form className={styles.composer} onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          aria-label="Your message"
          placeholder="Ask about events…"
          rows={2}
          maxLength={MAX_MESSAGE_LENGTH}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button type="submit" disabled={sending || !draft.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

function AssistantMessage({ reply }: { reply: ChatResponse }) {
  const notice = stopNotice(reply.stop_reason);
  const cost = formatCost(reply.cost_usd);
  const calls = reply.tool_calls.length;
  const details = [
    `${reply.seconds.toFixed(1)}s`,
    `${reply.steps} ${reply.steps === 1 ? "step" : "steps"}`,
    ...(cost ? [cost] : []),
  ].join(" · ");

  return (
    <article className={styles.assistant}>
      <Answer text={reply.answer} />
      {notice && <p className={styles.notice}>{notice}</p>}
      <div className={styles.meta}>
        {calls > 0 ? (
          <details>
            <summary>
              {calls} tool {calls === 1 ? "call" : "calls"} · {details}
            </summary>
            <ul>
              {reply.tool_calls.map((step, index) => (
                <li key={index} className={step.is_error ? styles.stepFailed : undefined}>
                  {describeStep(step)}
                  {step.is_error && " (failed)"}
                </li>
              ))}
            </ul>
          </details>
        ) : (
          details
        )}
      </div>
    </article>
  );
}
