"use client";

import Link from "next/link";

import { getStats } from "@/lib/api";
import { engineLabel, formatNumber, formatTotalCost } from "@/lib/format";
import { useApiData } from "@/lib/useApiData";

import styles from "./Pages.module.css";

export function Stats() {
  const { data, error, retry } = useApiData(getStats);

  return (
    <div className={styles.page}>
      <h2 className={styles.heading}>Usage by engine</h2>

      {error && (
        <div className={styles.errorBox} role="alert">
          <p>{error}</p>
          <button type="button" onClick={retry}>
            Try again
          </button>
        </div>
      )}
      {!data && !error && <p role="status">Loading…</p>}

      {data?.engines.length === 0 && (
        <p className={styles.empty}>
          No questions yet. <Link href="/">Ask the scout</Link> something, and the numbers will show up here.
        </p>
      )}

      {data && data.engines.length > 0 && (
        <>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Engine</th>
                  <th scope="col">Questions</th>
                  <th scope="col">Errors</th>
                  <th scope="col">Avg steps</th>
                  <th scope="col">Avg tool calls</th>
                  <th scope="col">Avg seconds</th>
                  <th scope="col">Avg prompt tokens</th>
                  <th scope="col">Avg output tokens</th>
                  <th scope="col">Total cost</th>
                </tr>
              </thead>
              <tbody>
                {data.engines.map((row) => (
                  <tr key={row.engine}>
                    <th scope="row">{engineLabel(row.engine)}</th>
                    <td>{row.questions}</td>
                    <td>{row.errors}</td>
                    <td>{formatNumber(row.avg_steps)}</td>
                    <td>{formatNumber(row.avg_tool_calls)}</td>
                    <td>{formatNumber(row.avg_seconds)}</td>
                    <td>{formatNumber(row.avg_prompt_tokens, 0)}</td>
                    <td>{formatNumber(row.avg_output_tokens, 0)}</td>
                    <td>{formatTotalCost(row.total_cost_usd, row.unpriced_questions)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.footnote}>
            Covers questions asked through this web app. Costs are estimates. Failed questions are left out of the
            averages, and any tokens they used aren&rsquo;t in the total, so the real cost can be a little higher.
            Questions on a model with no known price aren&rsquo;t in the total either. With a small number of questions,
            differences between engines can easily be chance.
          </p>
        </>
      )}
    </div>
  );
}
