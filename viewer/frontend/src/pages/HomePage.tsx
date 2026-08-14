import { useMemo, useState, useCallback } from "react";

import { api } from "../api";
import { metricLabel } from "../metricLabels";
import type { RunSummary } from "../types";
import { RunImporter } from "../components/RunImporter";

type HomePageProps = {
  runs: RunSummary[];
  onImport: (path: string) => Promise<void>;
  onRefreshAll: () => Promise<void>;
  onRefresh: (runId: string) => Promise<void>;
  onRemove: (runId: string) => Promise<void>;
  onCancel: (runId: string) => Promise<void>;
  onPause: (runId: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
};

export function HomePage({ runs, onImport, onRefreshAll, onRefresh, onRemove, onCancel, onPause, onResume, selectedRunId, onSelectRun }: HomePageProps) {
  const [projectFilter, setProjectFilter] = useState("all");
  const [showOlder, setShowOlder] = useState(true);
  const [showOnlyRunning, setShowOnlyRunning] = useState(false);
  const [refreshingProject, setRefreshingProject] = useState<string | null>(null);

  const handleRefreshProject = useCallback(async (projectId: string) => {
    setRefreshingProject(projectId);
    try {
      await api.refreshProject(projectId);
      await onRefreshAll();
    } finally {
      setRefreshingProject(null);
    }
  }, [onRefreshAll]);

  const cutoff = useMemo(() => Date.now() - 24 * 60 * 60 * 1000, []);

  const projectIds = useMemo(
    () => ["all", ...new Set(runs.map((run) => run.project_id).sort())],
    [runs],
  );

  const selectedRunIsOld = useMemo(() => {
    if (!selectedRunId) return false;
    const r = runs.find((r) => r.run_id === selectedRunId || r.island_run_ids?.includes(selectedRunId));
    if (!r?.created_at) return false;
    return Date.parse(r.created_at) < cutoff;
  }, [runs, selectedRunId, cutoff]);

  const filteredRuns = useMemo(() => {
    let projectFiltered = runs.filter((run) => projectFilter === "all" || run.project_id === projectFilter);
    if (showOnlyRunning) {
      projectFiltered = projectFiltered.filter((run) => run.status === "running");
    }
    if (showOlder || selectedRunIsOld) return projectFiltered;
    return projectFiltered.filter((run) => {
      const t = run.created_at ? Date.parse(run.created_at) : 0;
      const isRecent = t >= cutoff;
      const isSelected = run.run_id === selectedRunId || run.island_run_ids?.includes(selectedRunId ?? "");
      return isRecent || isSelected;
    });
  }, [projectFilter, runs, showOnlyRunning, showOlder, selectedRunIsOld, cutoff, selectedRunId]);
  const groupedRuns = useMemo(() => {
    const projectGroups = new Map<string, RunSummary[]>();
    for (const run of filteredRuns) {
      const existing = projectGroups.get(run.project_id) ?? [];
      existing.push(run);
      projectGroups.set(run.project_id, existing);
    }
    return [...projectGroups.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([projectId, projectRuns]) => [
        projectId,
        collapseRunsToJobs(projectRuns, selectedRunId),
      ] as const);
  }, [filteredRuns, selectedRunId]);

  return (
    <div className="page-grid">
      <RunImporter onImport={onImport} onRefreshAll={onRefreshAll} />
      <section className="panel">
        <div className="panel-header">
          <h2>Project Filter</h2>
        </div>
        <div className="controls-grid">
          <label className="field">
            <span>Project</span>
            <select value={projectFilter} onChange={(event) => setProjectFilter(event.target.value)}>
              {projectIds.map((projectId) => (
                <option key={projectId} value={projectId}>
                  {projectId === "all" ? "All Projects" : projectId}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Time range</span>
            <button
              type="button"
              className={showOlder ? "" : "ghost"}
              onClick={() => setShowOlder((v) => !v)}
              style={{ width: "100%" }}
            >
              {showOlder ? "All time" : "Last 24 hours"}
            </button>
          </label>
          <label className="field">
            <span>Only running</span>
            <button
              type="button"
              className={showOnlyRunning ? "" : "ghost"}
              onClick={() => setShowOnlyRunning((v) => !v)}
              style={{ width: "100%" }}
            >
              {showOnlyRunning ? "Running only" : "Show all"}
            </button>
          </label>
        </div>
      </section>
      {groupedRuns.map(([projectId, projectRuns]) => (
        <section key={projectId} className="page-grid">
          <div className="section-heading">
            <h2>{projectId}</h2>
            <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span>{projectRuns.length} job{projectRuns.length === 1 ? "" : "s"}</span>
              <button
                type="button"
                className="ghost"
                style={{ fontSize: "0.75rem", padding: "0.2rem 0.6rem" }}
                disabled={refreshingProject === projectId}
                onClick={() => void handleRefreshProject(projectId)}
              >
                {refreshingProject === projectId ? "Refreshing…" : "Refresh"}
              </button>
            </span>
          </div>
          <div className="card-grid">
            {projectRuns.map((job) => (
              <article key={job.key} className={`panel run-card${job.isSelected ? " selected" : ""}`}>
                <div className="panel-header run-card-header">
                  <div>
                    <h3>{formatTimestamp(job.primaryRun.created_at)}</h3>
                    <p>{job.isSelected ? "Current analysis target" : formatRunSubtitle(job.primaryRun)}</p>
                  </div>
                  <span className={`status-pill status-${(job.status ?? "unknown").toLowerCase()}`}>{job.status}</span>
                </div>
                <div className="run-stat-strip">
                  <span className="run-stat"><strong>Step</strong> {job.primaryRun.current_step}</span>
                  {(job.primaryRun.total_stages ?? 1) > 1 ? (
                    <span className="run-stat stage-badge" style={{ border: "none" }}><strong>Stage</strong> {job.primaryRun.current_stage ?? "?"}/{job.primaryRun.total_stages}</span>
                  ) : null}
                  {(job.primaryRun.island_count ?? 1) > 1 ? (
                    <span className="run-stat"><strong>Islands</strong> {job.primaryRun.island_count}</span>
                  ) : null}
                  <span className="run-stat"><strong>Candidates</strong> {job.primaryRun.candidate_count}</span>
                  <span className="run-stat"><strong>Cells</strong> {job.primaryRun.archive_occupancy}</span>
                  <span className="run-stat"><strong>{metricLabel(job.primaryRun.primary_metric_label, "Primary")}</strong> {format(job.primaryRun.best_primary_fitness)}</span>
                </div>
                {(() => {
                  const actionId = job.primaryRun.island_run_ids?.[0] ?? job.primaryRun.run_id;
                  const allIds = job.primaryRun.island_run_ids ?? [job.primaryRun.run_id];
                  return (
                    <div className="button-row run-actions">
                      <button type="button" onClick={() => onSelectRun(actionId)}>
                        {job.isSelected ? "Current" : "Set"}
                      </button>
                      <button type="button" onClick={() => { void Promise.all(allIds.map((id) => onRefresh(id))); }}>Refresh</button>
                      {job.status === "running" && (
                        <button type="button" onClick={() => { void onPause(actionId); }}>Pause</button>
                      )}
                      {job.status === "paused" && (
                        <button type="button" onClick={() => { void onResume(actionId); }}>Resume</button>
                      )}
                      {job.status === "running" && (
                        <button type="button" className="ghost danger" onClick={() => { void onCancel(actionId); }}>Cancel</button>
                      )}
                      <button type="button" className="ghost" onClick={() => { void Promise.all(allIds.map((id) => onRemove(id))); }}>Remove</button>
                    </div>
                  );
                })()}
              </article>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function format(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return value.toFixed(4);
}

function formatTimestamp(value: string | undefined) {
  if (!value) {
    return "Unknown time";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

type JobGroup = {
  key: string;
  primaryRun: RunSummary;
  runs: RunSummary[];
  isSelected: boolean;
  status: string;
};

function collapseRunsToJobs(runs: RunSummary[], selectedRunId: string | null): JobGroup[] {
  return runs
    .map((run) => {
      const relatedIds = run.island_run_ids ?? [run.run_id];
      const isSelected = relatedIds.includes(selectedRunId ?? "") || run.run_id === selectedRunId;
      return {
        key: run.run_id,
        primaryRun: run,
        runs: [run],
        isSelected,
        status: run.status ?? "unknown",
      };
    })
    .sort((left, right) => compareRuns(right.primaryRun, left.primaryRun));
}

function formatRunSubtitle(run: RunSummary, _runCount?: number) {
  if (run.status === "stalled" && run.last_updated_at) {
    return `Last update ${formatTimestamp(run.last_updated_at)}`;
  }
  return run.project_id;
}

function deriveGroupStatus(runs: RunSummary[]) {
  if (runs.some((run) => run.status === "running")) {
    return "running";
  }
  if (runs.some((run) => run.status === "stalled")) {
    return "stalled";
  }
  return runs[0]?.status ?? "unknown";
}

function compareRuns(left: RunSummary, right: RunSummary) {
  const leftTime = left.created_at ? Date.parse(left.created_at) : Number.NEGATIVE_INFINITY;
  const rightTime = right.created_at ? Date.parse(right.created_at) : Number.NEGATIVE_INFINITY;
  if (leftTime !== rightTime) {
    return leftTime - rightTime;
  }
  return left.run_id.localeCompare(right.run_id);
}
