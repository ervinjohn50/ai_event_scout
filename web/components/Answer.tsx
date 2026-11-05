import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import styles from "./Answer.module.css";

/**
 * Shows an answer written in Markdown. Answers can repeat text from outside sources, such as
 * event names, so they're treated as untrusted:
 * - react-markdown shows any HTML as plain text, so it can't run in the page.
 * - Images are removed, because loading one sends a request to whatever address it names,
 *   which an injected instruction could use to send data elsewhere.
 * - Links only work for safe address types, and open in a new tab that can't reach this one.
 */
export function Answer({ text }: { text: string }) {
  return (
    <div className={styles.answer}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]} // adds tables, which Claude often uses to compare events
        disallowedElements={["img"]}
        unwrapDisallowed
        components={{
          a: ({ href, children }) =>
            href ? (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ) : (
              <>{children}</> // an unsafe address was removed, so only the text remains
            ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
