"use client";

import { type FormEvent, type KeyboardEvent, useEffect, useRef } from "react";

import { type ChatResponse, type EngineName, MAX_MESSAGE_LENGTH } from "@/lib/api";
import { ENGINE_LABELS } from "@/lib/format";
import { describeRunningStep, describeStep, formatCost, stopNotice } from "@/lib/steps";

import { Answer } from "./Answer";
import styles from "./Chat.module.css";
import { type LiveStep, useChat } from "./ChatProvider";

const EXAMPLES = [
  "What concerts are on in Chicago this weekend?",
  "Any comedy in New York this month?",
  "Show me what's on my shortlist.",
];

export function Chat() {
  const { items, draft, setDraft, engine, sending, liveSteps, health, healthError, send, startNewChat } = useChat();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [items, sending, liveSteps]);

  function focusInput() {
    inputRef.current?.focus();
  }

  function submit() {
    void send().finally(focusInput);
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // isComposing is true while typing with an input method, where Enter confirms a word.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
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
            onChange={(event) => {
              startNewChat(event.target.value as EngineName);
              focusInput();
            }}
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
        <button
          type="button"
          className={styles.newChat}
          disabled={sending || empty}
          onClick={() => {
            startNewChat();
            focusInput();
          }}
        >
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
                    focusInput();
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

        {sending && <LiveProgress steps={liveSteps} />}
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

function LiveProgress({ steps }: { steps: LiveStep[] }) {
  const toolsRunning = steps.some((step) => step.status === "running");
  return (
    <div className={styles.progress} role="status">
      {steps.length > 0 && (
        <ul>
          {steps.map((step) => (
            <li key={step.id} className={step.status === "failed" ? styles.stepFailed : undefined}>
              <span className={styles[step.status]} aria-hidden="true" />
              {step.status === "running" && `${describeRunningStep(step)}…`}
              {step.status === "done" && describeStep({ ...step, is_error: false })}
              {step.status === "failed" && `${describeStep({ ...step, is_error: true })} (failed)`}
            </li>
          ))}
        </ul>
      )}
      {/* Between tools, the agent is deciding what to do next or writing its answer. */}
      {!toolsRunning && <p className={styles.working}>Thinking…</p>}
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
