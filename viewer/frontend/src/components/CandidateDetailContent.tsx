import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { hasSecondaryMetric, metricLabel } from "../metricLabels";
import type { CandidateDetailResponse } from "../types";

type CandidateDetailContentProps = {
  detail: CandidateDetailResponse;
};

export function CandidateDetailContent({ detail }: CandidateDetailContentProps) {
  const candidate = detail.candidate;
  const llmCall = detail.llm_call ?? null;
  const isRandomInit = llmCall?.phase === "random_init";
  const primaryMetric = metricLabel(detail.primary_metric_label, "Primary");
  const secondaryMetric = metricLabel(detail.secondary_metric_label, "Secondary");
  const showSecondary = hasSecondaryMetric(detail.secondary_metric_label);
  const trainMetrics = extractTrainMetrics(candidate.metadata);
  const directParent = detail.parents[0] ?? null;
  const targetCandidate = detail.target_candidate ?? directParent;
  const ancestorCandidates = detail.ancestor_candidates ?? [];
  const inspirationCandidates = detail.inspiration_candidates ?? [];
  return (
    <>
      {isRandomInit && (
        <div style={{ marginBottom: "0.75rem" }}>
          <span className="stage-badge" style={{ fontSize: "0.75rem", padding: "3px 10px" }}>
            Random Init
          </span>
          <span style={{ marginLeft: "0.5rem", color: "var(--color-text-muted, #888)", fontSize: "0.8rem" }}>
            Generated during initialisation — not evolved from a parent
          </span>
        </div>
      )}
      <div className="metric-grid compact">
        <Metric title={primaryMetric} value={formatNumber(candidate.primary_fitness)} />
        {showSecondary ? <Metric title={secondaryMetric} value={formatNumber(candidate.secondary_fitness)} /> : null}
        {trainMetrics ? <Metric title="Detected Events Mean" value={formatNumber(trainMetrics.mean_detected_bounce_count)} /> : null}
        {trainMetrics ? <Metric title="Detected Events Std" value={formatNumber(trainMetrics.std_detected_bounce_count)} /> : null}
        <Metric title="Status" value={candidate.is_active ? "Active" : "Inactive"} />
        {candidate.stage != null && candidate.stage > 0 ? <Metric title="Stage" value={String(candidate.stage)} /> : null}
      </div>

      <FitnessComponentsTable stats={candidate.stats} primaryMetric={primaryMetric} />

      <section className="modal-section">
        <h4>Feature Values</h4>
        <div className="structured-table">
          <div className="structured-row structured-head">
            <span>Feature</span>
            <span>Raw</span>
            <span>Normalized</span>
          </div>
          {detail.descriptor_labels.map((label, index) => (
            <div key={label} className="structured-row">
              <span>{label}</span>
              <strong>{formatNumber(candidate.descriptor_raw?.[index])}</strong>
              <strong>{formatNumber(candidate.descriptor_norm?.[index])}</strong>
            </div>
          ))}
        </div>
      </section>

      <div className="two-column modal-columns">
        <section className="modal-section">
          <h4>Parent</h4>
          <div className="lineage-list">
            {directParent ? (
              <Link className="lineage-card" to={`/candidates/${directParent.id}`}>
                <strong>{directParent.id}</strong>
                <span>{primaryMetric} {formatNumber(directParent.primary_fitness)}</span>
                {showSecondary ? <span>{secondaryMetric} {formatNumber(directParent.secondary_fitness)}</span> : null}
              </Link>
            ) : (
              <div className="inset-card">
                <strong>None</strong>
              </div>
            )}
          </div>
        </section>
        <section className="modal-section lineage-section-compact">
          <h4>Children</h4>
          <div className="lineage-list lineage-list-scroll">
            {detail.children.length ? (
              detail.children.map((child) => (
                <div key={child.id} className="lineage-card">
                  <strong>{child.id}</strong>
                  <span>{primaryMetric} {formatNumber(child.primary_fitness)}</span>
                  {showSecondary ? <span>{secondaryMetric} {formatNumber(child.secondary_fitness)}</span> : null}
                </div>
              ))
            ) : (
              <div className="inset-card">
                <strong>None</strong>
              </div>
            )}
          </div>
        </section>
      </div>

      <div className="two-column modal-columns">
        <section className="modal-section">
          <h4>Target Elite</h4>
          <div className="lineage-list">
            {targetCandidate ? (
              <Link className="lineage-card" to={`/candidates/${targetCandidate.id}`}>
                <strong>{targetCandidate.id}</strong>
                <span>{primaryMetric} {formatNumber(targetCandidate.primary_fitness)}</span>
                {showSecondary ? <span>{secondaryMetric} {formatNumber(targetCandidate.secondary_fitness)}</span> : null}
              </Link>
            ) : (
              <div className="inset-card">
                <strong>Unavailable</strong>
              </div>
            )}
          </div>
        </section>
        <section className="modal-section">
          <h4>Ancestors</h4>
          <div className="lineage-list">
            {ancestorCandidates.length ? (
              ancestorCandidates.map((ancestor) => (
                <Link key={ancestor.id} className="lineage-card" to={`/candidates/${ancestor.id}`}>
                  <strong>{ancestor.id}</strong>
                  <span>{primaryMetric} {formatNumber(ancestor.primary_fitness)}</span>
                  {showSecondary ? <span>{secondaryMetric} {formatNumber(ancestor.secondary_fitness)}</span> : null}
                </Link>
              ))
            ) : (
              <div className="inset-card">
                <strong>None</strong>
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="modal-section">
        <h4>Inspirational Elites</h4>
        <div className="lineage-list">
          {inspirationCandidates.length ? (
            inspirationCandidates.map((inspiration) => (
              <Link key={inspiration.id} className="lineage-card" to={`/candidates/${inspiration.id}`}>
                <strong>{inspiration.id}</strong>
                <span>{primaryMetric} {formatNumber(inspiration.primary_fitness)}</span>
                {showSecondary ? <span>{secondaryMetric} {formatNumber(inspiration.secondary_fitness)}</span> : null}
              </Link>
            ))
          ) : (
            <div className="inset-card">
              <strong>None</strong>
            </div>
          )}
        </div>
      </section>

      <section className="modal-section">
        <h4>History</h4>
        <div className="structured-table">
          <div className="structured-row structured-head">
            <span>Version</span>
            <span>{primaryMetric}</span>
            {showSecondary ? <span>{secondaryMetric}</span> : null}
          </div>
          {detail.history.map((record) => (
            <div key={`${record.id}-${record.record_version}`} className="structured-row">
              <span>{record.record_version}</span>
              <strong>{formatNumber(record.primary_fitness)}</strong>
              {showSecondary ? <strong>{formatNumber(record.secondary_fitness)}</strong> : null}
            </div>
          ))}
        </div>
      </section>

      <section className="modal-section">
        <h4>Code</h4>
        <pre className="code-viewer">{candidate.code}</pre>
      </section>

      <details id="llm-call-section" className="modal-section" open={Boolean(llmCall)}>
        <summary>{llmCall ? "LLM Call" : "LLM Call Unavailable"}</summary>
        {llmCall ? (
          <>
            <div className="structured-grid modal-subsection">
              <div className="structured-card">
                <span>Call ID</span>
                <strong>{formatUnknown(llmCall.llm_call_id)}</strong>
              </div>
              <div className="structured-card">
                <span>Model</span>
                <strong>{formatUnknown(llmCall.model)}</strong>
              </div>
              <div className="structured-card">
                <span>Status</span>
                <strong>{formatUnknown(llmCall.status)}</strong>
              </div>
              <div className="structured-card">
                <span>Latency</span>
                <strong>{formatLatency(llmCall.latency_seconds)}</strong>
              </div>
              <div className="structured-card">
                <span>Step</span>
                <strong>{formatUnknown(llmCall.step)}</strong>
              </div>
              <div className="structured-card">
                <span>Created</span>
                <strong>{formatDateTime(llmCall.created_at)}</strong>
              </div>
            </div>
            <div className="two-column modal-columns">
              <section className="modal-section">
                <h4>Parent At Call Time</h4>
                <div className="lineage-list">
                  {Array.isArray(llmCall.parent_ids) && llmCall.parent_ids.length ? (
                    <Link className="lineage-card" to={`/candidates/${String(llmCall.parent_ids[0])}`}>
                      <strong>{String(llmCall.parent_ids[0])}</strong>
                    </Link>
                  ) : (
                    <div className="inset-card">
                      <strong>None</strong>
                    </div>
                  )}
                </div>
              </section>
              <section className="modal-section">
                <h4>Children Produced</h4>
                <div className="lineage-list">
                  {Array.isArray(llmCall.child_ids) && llmCall.child_ids.length ? (
                    llmCall.child_ids.map((childId) => (
                      <Link key={String(childId)} className="lineage-card" to={`/candidates/${String(childId)}`}>
                        <strong>{String(childId)}</strong>
                      </Link>
                    ))
                  ) : (
                    <div className="inset-card">
                      <strong>None</strong>
                    </div>
                  )}
                </div>
              </section>
            </div>
            <details className="modal-section">
              <summary>Prompt</summary>
              <pre>{formatJsonOrText(llmCall.prompt)}</pre>
            </details>
            {llmCall.reasoning && (
              <details className="modal-section">
                <summary>Reasoning</summary>
                <pre>{formatJsonOrText(llmCall.reasoning)}</pre>
              </details>
            )}
            <details className="modal-section">
              <summary>Response</summary>
              <pre>{formatJsonOrText(llmCall.response)}</pre>
            </details>
          </>
        ) : (
          <div className="structured-card modal-subsection">
            <span>llm call</span>
            <strong>-</strong>
          </div>
        )}
      </details>

      <details className="modal-section">
        <summary>Metadata</summary>
        <div className="structured-grid modal-subsection">
          {Object.entries(candidate.metadata ?? {}).length ? (
            Object.entries(candidate.metadata).map(([key, value]) => (
              <div key={key} className="structured-card">
                <span>{key}</span>
                <strong>{formatUnknown(value)}</strong>
              </div>
            ))
          ) : (
            <div className="structured-card">
              <span>metadata</span>
              <strong>-</strong>
            </div>
          )}
        </div>
      </details>
    </>
  );
}

function extractTrainMetrics(metadata: Record<string, unknown> | null | undefined) {
  const raw = metadata?.train_metrics;
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const metrics = raw as Record<string, unknown>;
  return {
    mean_detected_bounce_count: typeof metrics.mean_detected_bounce_count === "number" ? metrics.mean_detected_bounce_count : null,
    std_detected_bounce_count: typeof metrics.std_detected_bounce_count === "number" ? metrics.std_detected_bounce_count : null,
  };
}

function Metric({ title, value }: { title: string; value: ReactNode }) {
  return (
    <div className="metric-card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatNumber(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return value.toFixed(4);
}

function formatUnknown(value: unknown): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => formatUnknown(item)).join(", ") : "[]";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function formatLatency(value: unknown): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${value.toFixed(2)}s`;
}

function formatDateTime(value: unknown): string {
  if (typeof value !== "string" || !value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatJsonOrText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value ?? null, null, 2);
}

// Ordered display config for fitness components
const COMPONENT_CONFIG: Record<string, { label: string; weight?: number; isDelta?: boolean; isBaseline?: boolean; isInfo?: boolean }> = {
  log_loss_delta:        { label: "Δ Log-Loss (vs uniform)", weight: 1.00, isDelta: true },
  log_loss:              { label: "Log-Loss (raw)",          isInfo: true },
  f1:                    { label: "F1 (reference)",          isInfo: true },
  n_root_nodes:          { label: "Root Nodes",              isInfo: true },
  n_intermediate_nodes:  { label: "Intermediate Nodes",      isInfo: true },
  n_edges:               { label: "Edges",                   isInfo: true },
  run_error_rate:        { label: "Episode Error Rate",      isInfo: true },
};

function FitnessComponentsTable({ stats, primaryMetric }: { stats?: Record<string, unknown> | null; primaryMetric: string }) {
  if (!stats) return null;
  const entry = stats["components"] as { value?: Record<string, unknown>; explanation?: string } | undefined;
  if (!entry?.value || typeof entry.value !== "object") return null;
  const components = entry.value as Record<string, unknown>;
  if (!Object.keys(components).length) return null;

  // Split into scored (weighted) rows, info rows, and baseline rows
  const scoredKeys = Object.keys(COMPONENT_CONFIG).filter(k => COMPONENT_CONFIG[k].weight != null && k in components);
  const infoKeys = Object.keys(COMPONENT_CONFIG).filter(k => COMPONENT_CONFIG[k].isInfo && k in components);
  const baselineKeys = Object.keys(COMPONENT_CONFIG).filter(k => COMPONENT_CONFIG[k].isBaseline && k in components);
  // Any unknown keys not in config
  const knownKeys = new Set(Object.keys(COMPONENT_CONFIG));
  const unknownKeys = Object.keys(components).filter(k => !knownKeys.has(k) && typeof components[k] === "number");

  const renderRow = (key: string) => {
    const v = components[key];
    const cfg = COMPONENT_CONFIG[key] ?? {};
    const num = typeof v === "number" ? v : null;
    const bool = typeof v === "boolean" ? v : null;
    const { label = key.replace(/_/g, " "), weight, isDelta, isBaseline, isInfo } = cfg;
    const muted = isBaseline || isInfo;
    const color = isBaseline ? "#7a8f8d" : isInfo ? "#9aada9" : isDelta && num != null && num > 0 ? "#10b981" : isDelta && num != null && num < 0 ? "#ef4444" : undefined;
    const displayVal = bool != null ? (bool ? "yes" : "no")
      : num != null ? `${num.toFixed(4)}${isDelta && num > 0 ? " ↑" : isDelta && num < 0 ? " ↓" : ""}` : String(v);
    const weightLabel = weight != null ? <span style={{ color: "#5a7572", fontSize: "0.78em", marginLeft: "0.4em" }}>×{weight.toFixed(2)}</span> : null;
    return (
      <div key={key} className="structured-row" style={{ opacity: muted ? 0.65 : 1 }}>
        <span style={{ color }}>{label}{weightLabel}</span>
        <span style={{ fontVariantNumeric: "tabular-nums", color }}>{displayVal}</span>
      </div>
    );
  };

  return (
    <section className="modal-section">
      <h4>{primaryMetric} Components</h4>
      <div className="structured-table">
        <div className="structured-row structured-head">
          <span>Component</span>
          <span>Value</span>
        </div>
        {scoredKeys.length > 0 && scoredKeys.map(renderRow)}
        {(infoKeys.length > 0 || baselineKeys.length > 0 || unknownKeys.length > 0) && (
          <div className="structured-row" style={{ borderTop: "1px solid rgba(255,255,255,0.08)", marginTop: "4px", paddingTop: "4px" }}>
            <span style={{ color: "#5a7572", fontSize: "0.78em" }}>Reference</span>
            <span />
          </div>
        )}
        {baselineKeys.map(renderRow)}
        {infoKeys.map(renderRow)}
        {unknownKeys.map(k => renderRow(k))}
      </div>
    </section>
  );
}
