"use client";

import Link from "next/link";
import { useState } from "react";

import { ApiError, getShortlist, removeFromShortlist, type ShortlistEntry } from "@/lib/api";
import { formatEventDate, safeUrl } from "@/lib/format";
import { useApiData } from "@/lib/useApiData";

import styles from "./Pages.module.css";

export function Shortlist() {
  const { data, error, retry, setData } = useApiData(getShortlist);
  const [removing, setRemoving] = useState<string | null>(null);
  const [removeError, setRemoveError] = useState<string | null>(null);

  async function remove(eventId: string) {
    setRemoving(eventId);
    setRemoveError(null);
    try {
      await removeFromShortlist(eventId);
      dropEntry(eventId);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        dropEntry(eventId); // already gone, which is what was wanted
      } else {
        setRemoveError(err instanceof ApiError ? err.message : "Something went wrong.");
      }
    } finally {
      setRemoving(null);
    }
  }

  function dropEntry(eventId: string) {
    setData((current) => current && { entries: current.entries.filter((entry) => entry.event_id !== eventId) });
  }

  return (
    <div className={styles.page}>
      <h2 className={styles.heading}>Your shortlist</h2>

      {error && (
        <div className={styles.errorBox} role="alert">
          <p>{error}</p>
          <button type="button" onClick={retry}>
            Try again
          </button>
        </div>
      )}
      {!data && !error && <p role="status">Loading…</p>}

      {removeError && (
        <p className={styles.errorBox} role="alert">
          {removeError}
        </p>
      )}

      {data?.entries.length === 0 && (
        <p className={styles.empty}>
          Nothing saved yet. <Link href="/">Ask the scout</Link> to find an event and save it.
        </p>
      )}

      {data && data.entries.length > 0 && (
        <ul className={styles.list}>
          {data.entries.map((entry) => (
            <EventCard
              key={entry.event_id}
              entry={entry}
              removing={removing === entry.event_id}
              onRemove={() => void remove(entry.event_id)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function EventCard({
  entry,
  removing,
  onRemove,
}: {
  entry: ShortlistEntry;
  removing: boolean;
  onRemove: () => void;
}) {
  const place = [entry.venue, entry.city].filter(Boolean).join(", ");
  const tickets = safeUrl(entry.ticket_url);
  return (
    <li className={styles.card}>
      <div className={styles.cardBody}>
        <h3>{entry.name}</h3>
        <p className={styles.cardMeta}>
          {formatEventDate(entry.date)}
          {place && ` · ${place}`}
          {entry.price && ` · ${entry.price}`}
        </p>
        {entry.note && <p className={styles.note}>{entry.note}</p>}
        {tickets && (
          <a href={tickets} target="_blank" rel="noopener noreferrer">
            Tickets
          </a>
        )}
      </div>
      <button type="button" disabled={removing} aria-label={`Remove ${entry.name}`} onClick={onRemove}>
        {removing ? "Removing…" : "Remove"}
      </button>
    </li>
  );
}
