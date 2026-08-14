import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import type { TimelineBadge, TimelineEventRecord, TimelineField } from "../types";
import { DetailModal } from "./DetailModal";

type EventTimelineProps = {
  events: TimelineEventRecord[];
  className?: string;
};

export function EventTimeline({ events, className }: EventTimelineProps) {
  const orderedEvents = useMemo(() => events.slice().reverse(), [events]);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEventRecord | null>(null);

  return (
    <>
      <div className={className ? `panel ${className}` : "panel"}>
        <div className="panel-header">
          <h3>Event Timeline</h3>
          <span className="muted">{orderedEvents.length} items</span>
        </div>
        <div className="timeline">
          {orderedEvents.map((event, index) => (
            <button
              key={`${event.type}-${event.step}-${index}`}
              type="button"
              className={`timeline-item timeline-trigger timeline-tone-${event.tone ?? "neutral"}`}
              onClick={() => setSelectedEvent(event)}
            >
              <div className={`timeline-dot timeline-dot-${event.tone ?? "neutral"}`} />
              <div className="timeline-copy">
                <div className="timeline-topline">
                  <strong>{event.title ?? formatEventTitle(event.type)}</strong>
                  <span className="timeline-step">
                    Step {String(event.step ?? "-")}
                    {event.timestamp ? ` · ${formatTimestamp(String(event.timestamp))}` : ""}
                  </span>
                </div>
                <div className="timeline-detail">{event.summary ?? "Open for details"}</div>
                {event.badges?.length ? (
                  <div className="timeline-badges">
                    {event.badges.map((badge, badgeIndex) => (
                      <span key={`${badge.label}-${badgeIndex}`} className="timeline-badge">
                        <span>{badge.label}</span>
                        <strong>{formatBadgeValue(badge)}</strong>
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            </button>
          ))}
        </div>
      </div>
      {selectedEvent ? (
        <DetailModal
          title={selectedEvent.title ?? formatEventTitle(selectedEvent.type)}
          subtitle={`${formatEventTitle(selectedEvent.type)} · Step ${String(selectedEvent.step ?? "-")}`}
          onClose={() => setSelectedEvent(null)}
        >
          {selectedEvent.summary ? (
            <section className="modal-section">
              <h4>Summary</h4>
              <p className="modal-copy">{selectedEvent.summary}</p>
            </section>
          ) : null}
          {selectedEvent.badges?.length ? (
            <section className="modal-section">
              <h4>Overview</h4>
              <div className="structured-grid">
                {selectedEvent.badges.map((badge, index) => (
                  <div key={`${badge.label}-${index}`} className="structured-card">
                    <span>{badge.label}</span>
                    <strong>{formatBadgeValue(badge)}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
          {selectedEvent.fields?.length ? (
            <section className="modal-section">
              <h4>Details</h4>
              <div className="structured-grid">
                {selectedEvent.fields.map((field, index) => (
                  <div key={`${field.label}-${index}`} className="structured-card">
                    <span>{field.label}</span>
                    <strong>{renderFieldValue(field)}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
          <details className="modal-section">
            <summary>Show Raw JSON</summary>
            <pre>{JSON.stringify(selectedEvent, null, 2)}</pre>
          </details>
        </DetailModal>
      ) : null}
    </>
  );
}

function formatEventTitle(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatBadgeValue(badge: TimelineBadge): string {
  return formatValue(badge.value);
}

function renderFieldValue(field: TimelineField) {
  if (field.kind === "candidate" && typeof field.value === "string" && field.value.trim()) {
    return <Link to={`/candidates/${field.value}`}>{field.value}</Link>;
  }
  if (field.kind === "candidate_list" && Array.isArray(field.value)) {
    const values = field.value;
    if (!values.length) {
      return "-";
    }
    return (
      <span className="timeline-link-list">
        {values.map((item, index) =>
          typeof item === "string" && item.trim() ? (
            <span key={`${item}-${index}`}>
              <Link to={`/candidates/${item}`}>{item}</Link>
              {index < values.length - 1 ? ", " : ""}
            </span>
          ) : null,
        )}
      </span>
    );
  }
  if (field.kind === "json") {
    return <code>{formatJson(field.value)}</code>;
  }
  return formatValue(field.value);
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (Array.isArray(value)) {
    if (!value.length) {
      return "[]";
    }
    return value.map((item) => formatValue(item)).join(", ");
  }
  if (typeof value === "object") {
    return formatJson(value);
  }
  return String(value);
}

function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return ts;
  }
}
