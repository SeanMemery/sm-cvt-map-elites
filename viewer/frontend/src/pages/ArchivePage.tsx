import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { api } from "../api";
import { ArchiveMap } from "../components/ArchiveMap";
import { CandidateDetailModal } from "../components/CandidateDetailModal";
import { CandidatePanel } from "../components/CandidatePanel";
import { HelpBadge } from "../components/HelpBadge";
import { LoadingBar } from "../components/LoadingBar";
import { MetricSelector } from "../components/MetricSelector";
import { hasSecondaryMetric } from "../metricLabels";
import { metricLabel } from "../metricLabels";
import type { CandidateDetailResponse, CandidateRecord, EventRecord, IslandInfoResponse, RunSummaryResponse, SnapshotResponse } from "../types";

type ArchivePageProps = {
  runId: string | null;
  islandMode: string;
};

export function ArchivePage({ runId, islandMode }: ArchivePageProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [stepToStage, setStepToStage] = useState<Map<number, number>>(new Map());
  const [candidates, setCandidates] = useState<CandidateRecord[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateDetailResponse | null>(null);
  const [showFullCandidate, setShowFullCandidate] = useState(false);
  const [loading, setLoading] = useState(false);
  const [candidateLoading, setCandidateLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [islandInfo, setIslandInfo] = useState<IslandInfoResponse | null>(null);
  const [selectedIsland, setSelectedIsland] = useState<string>("all");
  const [mergedSnapshot, setMergedSnapshot] = useState<SnapshotResponse | null>(null);
  const snapshotCacheRef = useRef<Map<string, SnapshotResponse>>(new Map());

  const descriptorLabels = summary?.descriptor_labels ?? [];
  const showSecondary = hasSecondaryMetric(summary?.secondary_metric_label);
  const showPrimaryValidation = candidates.some((candidate) => candidate.primary_validation_fitness != null);
  const showSecondaryValidation = candidates.some((candidate) => candidate.secondary_validation_fitness != null);
  const defaultX = "raw:0";
  const defaultY = "primary_fitness";
  const xMetric = searchParams.get("xMetric") ?? defaultX;
  const yMetric = searchParams.get("yMetric") ?? defaultY;
  const colorMetric = searchParams.get("colorMetric") ?? "primary_fitness";
  const sizeMetric = searchParams.get("sizeMetric") ?? "none";
  const fixedSize = clampFixedSize(searchParams.get("fixedSize"));
  const selectedCandidateId = searchParams.get("candidateId");
  useEffect(() => {
    if (!runId) {
      setSummary(null);
      setCandidates([]);
      setStepToStage(new Map());
      setSelectedCandidate(null);
      setError(null);
      return;
    }
    const activeRunId = runId;
    const activeIslandMode = islandMode;
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [summaryResponse, timeseriesResponse, islandInfoResponse] = await Promise.all([
          api.getRunSummary(activeRunId),
          api.getTimeseries(activeRunId),
          api.getIslandInfo(activeRunId).catch(() => null),
        ]);
        const hasSiblings = (islandInfoResponse?.sibling_run_ids.length ?? 0) > 0;
        const candidateResponse = (activeIslandMode === "all" && hasSiblings)
          ? await api.getCombinedCandidates(activeRunId)
          : await api.getCandidates(activeRunId, buildPopulationParams(summaryResponse.summary.candidate_count));
        if (!cancelled) {
          setSummary(summaryResponse);
          setCandidates(candidateResponse.items);
          const allEvents: EventRecord[] = [
            ...(timeseriesResponse.events ?? []),
            ...timeseriesResponse.timeline_events,
          ];
          const stageMap = new Map<number, number>();
          for (const event of allEvents) {
            if (event.type === "stage_start" && typeof event.step === "number" && typeof (event as Record<string, unknown>).stage === "number") {
              stageMap.set(event.step, (event as Record<string, unknown>).stage as number);
            }
          }
          setStepToStage(stageMap);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [runId, islandMode]);

  // Load island info and reset island selection when run changes
  useEffect(() => {
    if (!runId) { setIslandInfo(null); setMergedSnapshot(null); return; }
    snapshotCacheRef.current = new Map();
    api.getIslandInfo(runId).then((info) => {
      setIslandInfo(info);
      if (info.sibling_run_ids.length > 0) {
        setSelectedIsland(islandMode === "all" ? "all" : (islandMode || runId!));
      } else {
        setSelectedIsland(runId);
      }
    }).catch(() => {
      setIslandInfo(null);
      setSelectedIsland(runId);
    });
  }, [runId, islandMode]);

  useEffect(() => {
    if (!islandInfo) return;
    if (islandInfo.sibling_run_ids.length > 0) {
      setSelectedIsland(islandMode === "all" ? "all" : (islandMode || (runId ?? "")));
    }
  }, [islandMode, islandInfo, runId]);

  // Load merged snapshot when "all" is selected
  useEffect(() => {
    if (!runId || selectedIsland !== "all") { setMergedSnapshot(null); return; }
    const cacheKey = `merged:${runId}`;
    const cached = snapshotCacheRef.current.get(cacheKey);
    if (cached) { setMergedSnapshot(cached); return; }
    api.getMergedSnapshot(runId).then((snap) => {
      snapshotCacheRef.current.set(cacheKey, snap);
      setMergedSnapshot(snap);
    }).catch(() => setMergedSnapshot(null));
  }, [runId, selectedIsland]);

  useEffect(() => {
    if (!runId || !candidates.length) return;
    const selectedId = selectedCandidateId;
    const first = candidates[0];
    if (selectedId || first) {
      const next = new URLSearchParams(searchParams);
      next.set("candidateId", selectedId ?? first?.id ?? "");
      setSearchParams(next, { replace: true });
    }
  }, [candidates, searchParams, selectedCandidateId, setSearchParams, runId]);

  useEffect(() => {
    const candidateId = selectedCandidateId;
    if (!runId || !candidateId) {
      setSelectedCandidate(null);
      setShowFullCandidate(false);
      return;
    }
    const activeRunId = runId;
    const activeCandidateId = candidateId;
    let cancelled = false;
    async function loadCandidate() {
      setCandidateLoading(true);
      try {
        const response = await api.getCandidate(activeRunId, activeCandidateId);
        if (!cancelled) setSelectedCandidate(response);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setCandidateLoading(false);
      }
    }
    void loadCandidate();
    return () => { cancelled = true; };
  }, [runId, selectedCandidateId]);

  if (!runId) {
    return <NoRunSelected message="Choose a run on Home, then use the left navigation to inspect its archive." />;
  }

  const metricOptions = useMemo(
    () => [
      { value: "primary_fitness", label: metricLabel(summary?.primary_metric_label, "Primary") },
      ...(showPrimaryValidation
        ? [{ value: "primary_validation_fitness", label: metricLabel(summary?.primary_validation_metric_label, "Primary Validation") }]
        : []),
      ...(showSecondary ? [{ value: "secondary_fitness", label: metricLabel(summary?.secondary_metric_label, "Secondary") }] : []),
      ...(showSecondaryValidation
        ? [{ value: "secondary_validation_fitness", label: metricLabel(summary?.secondary_validation_metric_label, "Secondary Validation") }]
        : []),
      ...descriptorLabels.map((label, index) => ({ value: `raw:${index}`, label })),
    ],
    [
      descriptorLabels,
      showPrimaryValidation,
      showSecondary,
      showSecondaryValidation,
      summary?.primary_metric_label,
      summary?.primary_validation_metric_label,
      summary?.secondary_metric_label,
      summary?.secondary_validation_metric_label,
    ],
  );
  const xAxisLabel = metricOptions.find((option) => option.value === xMetric)?.label ?? xMetric;
  const yAxisLabel = metricOptions.find((option) => option.value === yMetric)?.label ?? yMetric;

  function setMetric(key: string, value: string) {
    const next = new URLSearchParams(searchParams);
    next.set(key, value);
    setSearchParams(next);
  }

  function handleCandidateClick(candidateId: string) {
    const next = new URLSearchParams(searchParams);
    next.set("candidateId", candidateId);
    setSearchParams(next);
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }
  if (!summary) {
    return (
      <>
        <LoadingBar active={loading} label="Loading archive explorer" />
        <div className="loading-panel">Loading archive explorer...</div>
      </>
    );
  }

  return (
    <div className="page-grid">
      <LoadingBar active={loading || candidateLoading} label="Loading archive data" />
      <section className="panel">
        <div className="panel-header">
          <div>
            <h1>Archive Explorer</h1>
            <p>{formatRunStamp(summary.summary.project_id, summary.summary.created_at)}</p>
          </div>
        </div>
        {summary.snapshot_steps.length > 0 ? (
          <div className="muted" style={{ fontSize: "0.82rem", marginBottom: "0.35rem" }}>
            Snapshots: {summary.snapshot_steps.map((s) => {
              const stage = stepToStage.get(s);
              return stage != null ? `S${stage}·${s}` : String(s);
            }).join(", ")}
          </div>
        ) : null}
        <div className="controls-grid">
          <MetricSelector label="X Metric" value={xMetric} onChange={(value) => setMetric("xMetric", value)} options={metricOptions} />
          <MetricSelector label="Y Metric" value={yMetric} onChange={(value) => setMetric("yMetric", value)} options={metricOptions} />
          <MetricSelector
            label="Color By"
            value={colorMetric}
            onChange={(value) => setMetric("colorMetric", value)}
            options={[
              { value: "solid", label: "Solid" },
              { value: "primary_fitness", label: metricLabel(summary.primary_metric_label, "Primary") },
              ...(showPrimaryValidation
                ? [{ value: "primary_validation_fitness", label: metricLabel(summary.primary_validation_metric_label, "Primary Validation") }]
                : []),
              ...(showSecondary ? [{ value: "secondary_fitness", label: metricLabel(summary.secondary_metric_label, "Secondary") }] : []),
              ...(showSecondaryValidation
                ? [{ value: "secondary_validation_fitness", label: metricLabel(summary.secondary_validation_metric_label, "Secondary Validation") }]
                : []),
              { value: "age", label: "Candidate Age" },
            ]}
          />
          <MetricSelector
            label="Size By"
            value={sizeMetric}
            onChange={(value) => setMetric("sizeMetric", value)}
            options={[
              { value: "none", label: "Fixed" },
              { value: "primary_fitness", label: metricLabel(summary.primary_metric_label, "Primary") },
              ...(showPrimaryValidation
                ? [{ value: "primary_validation_fitness", label: metricLabel(summary.primary_validation_metric_label, "Primary Validation") }]
                : []),
              ...(showSecondary ? [{ value: "secondary_fitness", label: metricLabel(summary.secondary_metric_label, "Secondary") }] : []),
              ...(showSecondaryValidation
                ? [{ value: "secondary_validation_fitness", label: metricLabel(summary.secondary_validation_metric_label, "Secondary Validation") }]
                : []),
              { value: "age", label: "Age" },
              ...descriptorLabels.map((label, index) => ({ value: `raw:${index}`, label })),
            ]}
          />
        </div>
      </section>
      <div className="archive-layout">
        <div className="panel">
          <div className="panel-header">
            <div className="title-with-help">
              <h3>Archive Map</h3>
              <HelpBadge text="Plots archive elites in feature space. Choose raw, normalized, centroid, or projected Voronoi views to understand how solutions occupy the archive." />
            </div>
            {sizeMetric === "none" ? (
              <label className="archive-slider-field archive-slider-field-compact">
                <span>Size</span>
                <input
                  type="range"
                  min="4"
                  max="20"
                  step="1"
                  value={fixedSize}
                  onChange={(event) => setMetric("fixedSize", event.target.value)}
                />
                <strong>{fixedSize}</strong>
              </label>
            ) : null}
          </div>
          <ArchiveMap
            snapshot={
              selectedIsland === "all" && mergedSnapshot
                ? mergedSnapshot
                : buildPopulationSnapshot(candidates, summary.summary.current_step)
            }
            candidates={candidates}
            xMetric={xMetric}
            yMetric={yMetric}
            colorMetric={colorMetric}
            sizeMetric={sizeMetric}
            fixedSize={fixedSize}
            primaryMetricLabel={summary.primary_metric_label}
            primaryValidationMetricLabel={summary.primary_validation_metric_label}
            secondaryMetricLabel={summary.secondary_metric_label}
            secondaryValidationMetricLabel={summary.secondary_validation_metric_label}
            onCandidateClick={handleCandidateClick}
          />
        </div>
        <CandidatePanel
          candidate={selectedCandidate?.candidate}
          descriptorLabels={descriptorLabels}
          primaryMetricLabel={summary.primary_metric_label}
          primaryValidationMetricLabel={summary.primary_validation_metric_label}
          secondaryMetricLabel={summary.secondary_metric_label}
          secondaryValidationMetricLabel={summary.secondary_validation_metric_label}
          onOpenFull={selectedCandidate ? () => setShowFullCandidate(true) : undefined}
        />
      </div>
      {selectedCandidate && showFullCandidate ? (
        <CandidateDetailModal detail={selectedCandidate} onClose={() => setShowFullCandidate(false)} />
      ) : null}
    </div>
  );
}

function buildPopulationParams(candidateCount: number) {
  const params = new URLSearchParams();
  params.set("limit", String(Math.max(candidateCount, 1)));
  params.set("sort", "created_at_step:desc");
  params.set("include_code", "false");
  return params;
}

function buildPopulationSnapshot(candidates: CandidateRecord[], step: number): SnapshotResponse {
  return {
    step,
    normalizer_bounds: { lower: [], upper: [] },
    occupied_cells: candidates.map((candidate) => ({
      cell_id: candidate.cell_id ?? -1,
      candidate_id: candidate.id,
      primary_fitness: candidate.primary_fitness,
      primary_validation_fitness: candidate.primary_validation_fitness,
      secondary_fitness: candidate.secondary_fitness,
      secondary_validation_fitness: candidate.secondary_validation_fitness,
      descriptor_raw: candidate.descriptor_raw,
      descriptor_norm: candidate.descriptor_norm,
    })),
  };
}

function formatRunStamp(projectId: string, createdAt?: string) {
  if (!createdAt) {
    return projectId;
  }
  const date = new Date(createdAt);
  return `${projectId} · ${Number.isNaN(date.getTime()) ? createdAt : date.toLocaleString()}`;
}

function NoRunSelected({ message }: { message: string }) {
  return (
    <section className="panel empty-state">
      <h2>No Run Selected</h2>
      <p>{message}</p>
    </section>
  );
}

function clampFixedSize(rawValue: string | null) {
  const parsed = Number(rawValue ?? "8");
  if (!Number.isFinite(parsed)) {
    return 8;
  }
  return Math.max(4, Math.min(20, Math.round(parsed)));
}
