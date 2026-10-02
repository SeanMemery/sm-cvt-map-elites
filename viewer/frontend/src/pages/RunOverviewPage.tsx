import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { CandidateDetailModal } from "../components/CandidateDetailModal";
import { EChart } from "../components/EChart";
import { EventTimeline } from "../components/EventTimeline";
import { FitnessChart } from "../components/FitnessChart";
import { HelpBadge } from "../components/HelpBadge";
import { IslandFitnessChart } from "../components/IslandFitnessChart";
import { LoadingBar } from "../components/LoadingBar";
import { MetricSelector } from "../components/MetricSelector";
import { bestMetricLabel, hasSecondaryMetric, metricLabel } from "../metricLabels";
import type {
  CandidateDetailResponse,
  CandidateRecord,
  IslandInfoResponse,
  RunCompareResponse,
  RunSummaryResponse,
  StageMarker,
  TimingRecord,
  TimeseriesResponse,
} from "../types";

const ISLAND_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

/** Derive a stable color from the island NUMBER embedded in the run ID.
 *  e.g. "causal_...island_0_..." → index 0 → always same color. */
function islandColor(id: string): string {
  const match = id.match(/island[_\s]?(\d+)/i);
  const idx = match ? parseInt(match[1], 10) : 0;
  return ISLAND_COLORS[idx % ISLAND_COLORS.length];
}

function islandShortLabel(id: string, idx: number): string {
  const match = id.match(/island[_\s]?(\d+)/i);
  return match ? `Island ${match[1]}` : `Island ${idx + 1}`;
}

/** Keep every nth entry so plots stay readable as step counts grow.
 *  ≤25 steps → every step, ≤50 → every 2, ≤75 → every 3, etc. */
function thinStats<T extends { step?: number | null }>(entries: T[]): T[] {
  if (entries.length === 0) return entries;
  const maxStep = Math.max(...entries.map((e) => Number(e.step ?? 0)));
  const n = Math.max(1, Math.ceil(maxStep / 25));
  if (n <= 1) return entries;
  // Always keep the last entry so the line extends to the current step
  const filtered = entries.filter((e) => Number(e.step ?? 0) % n === 0);
  const last = entries[entries.length - 1];
  if (filtered.length === 0 || filtered[filtered.length - 1] !== last) {
    filtered.push(last);
  }
  return filtered;
}

type RunOverviewPageProps = {
  runId: string | null;
  islandMode?: string;
};

export function RunOverviewPage({ runId, islandMode = "all" }: RunOverviewPageProps) {
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesResponse | null>(null);
  const [compare, setCompare] = useState<RunCompareResponse | null>(null);
  const [timing, setTiming] = useState<TimingRecord[] | null>(null);
  const [bestPrimaryCandidate, setBestPrimaryCandidate] = useState<CandidateRecord | null>(null);
  const [bestSecondaryCandidate, setBestSecondaryCandidate] = useState<CandidateRecord | null>(null);
  const [selectedCandidateDetail, setSelectedCandidateDetail] = useState<CandidateDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [occupancyMode, setOccupancyMode] = useState<"count" | "percent">("count");
  const [islandInfo, setIslandInfo] = useState<IslandInfoResponse | null>(null);
  const [siblingStats, setSiblingStats] = useState<Record<string, Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>>>({});
  const [siblingSummaries, setSiblingSummaries] = useState<RunSummaryResponse[]>([]);
  const requestVersionRef = useRef(0);
  async function loadOverview(activeRunId: string, refreshFirst = false) {
    const requestVersion = requestVersionRef.current + 1;
    requestVersionRef.current = requestVersion;
    setLoading(true);
    setError(null);
    try {
      if (refreshFirst) {
        await api.refreshRun(activeRunId);
      }
      const [
        summaryResponse,
        timeseriesResponse,
        compareResponse,
        bestPrimaryResponse,
        bestSecondaryResponse,
        timingResponse,
        islandInfoResponse,
      ] = await Promise.all([
        api.getRunSummary(activeRunId),
        api.getTimeseries(activeRunId),
        api.getRunCompare(activeRunId),
        api.getCandidates(activeRunId, buildTopCandidateParams("primary_fitness")),
        api.getCandidates(activeRunId, buildTopCandidateParams("secondary_fitness", true)).catch(() => ({ items: [] })),
        api.getTiming(activeRunId).catch(() => ({ timing: [] })),
        api.getIslandInfo(activeRunId).catch(() => null),
      ]);
      if (requestVersion !== requestVersionRef.current) {
        return;
      }
      setSummary(summaryResponse);
      setTimeseries(timeseriesResponse);
      setCompare(compareResponse);
      setBestPrimaryCandidate(bestPrimaryResponse.items[0] ?? null);
      setBestSecondaryCandidate(bestSecondaryResponse.items[0] ?? null);
      // Sibling top candidates are resolved later once sibling run IDs are known
      setTiming(timingResponse.timing ?? []);

      // Refresh sibling data (timeseries + summaries) on every poll
      if (islandInfoResponse && islandInfoResponse.sibling_run_ids.length > 0) {
        if (refreshFirst) {
          await Promise.all(islandInfoResponse.sibling_run_ids.map((id) => api.refreshRun(id).catch(() => null)));
        }
        const newStats: Record<string, Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>> = {};
        const newSummaries: RunSummaryResponse[] = [];
        if (islandInfoResponse.island_id) {
          newStats[islandInfoResponse.island_id] = timeseriesResponse.stats as Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>;
        }
        const allRunIds = [activeRunId, ...islandInfoResponse.sibling_run_ids];
        await Promise.all(
          islandInfoResponse.sibling_run_ids.map(async (siblingId) => {
            try {
              const [ts, summ] = await Promise.all([api.getTimeseries(siblingId), api.getRunSummary(siblingId)]);
              newStats[siblingId] = ts.stats as Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>;
              newSummaries.push(summ);
            } catch { /* ignore */ }
          }),
        );
        if (requestVersion !== requestVersionRef.current) return;
        // Find the best primary and secondary candidates across ALL islands
        const [allPrimary, allSecondary] = await Promise.all([
          Promise.all(allRunIds.map((id) => api.getCandidates(id, buildTopCandidateParams("primary_fitness")).catch(() => ({ items: [] })))),
          Promise.all(allRunIds.map((id) => api.getCandidates(id, buildTopCandidateParams("secondary_fitness", true)).catch(() => ({ items: [] })))),
        ]);
        // Tag each candidate with the island run it came from, then find the best
        const taggedPrimary = allPrimary.flatMap((r, i) => r.items.map((c) => ({ ...c, island_run_id: allRunIds[i] })));
        const taggedSecondary = allSecondary.flatMap((r, i) => r.items.map((c) => ({ ...c, island_run_id: allRunIds[i] })));
        const bestPrimary = taggedPrimary.sort((a, b) => (b.primary_fitness ?? 0) - (a.primary_fitness ?? 0))[0] ?? null;
        const bestSecondary = taggedSecondary.sort((a, b) => (b.secondary_fitness ?? 0) - (a.secondary_fitness ?? 0))[0] ?? null;
        if (bestPrimary) setBestPrimaryCandidate(bestPrimary);
        if (bestSecondary) setBestSecondaryCandidate(bestSecondary);
        setIslandInfo(islandInfoResponse);
        setSiblingStats(newStats);
        setSiblingSummaries(newSummaries);
      }
    } catch (err) {
      if (requestVersion !== requestVersionRef.current) {
        return;
      }
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (requestVersion === requestVersionRef.current) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    if (!runId) {
      setSummary(null);
      setTimeseries(null);
      setCompare(null);
      setBestPrimaryCandidate(null);
      setBestSecondaryCandidate(null);
      setSelectedCandidateDetail(null);
      setTiming(null);
      setError(null);
      return;
    }
    const activeRunId = runId;
    let cancelled = false;
    async function load() {
      await loadOverview(activeRunId, false);
      if (cancelled) {
        return;
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  // Clear island state when run changes (data is now loaded inside loadOverview)
  useEffect(() => {
    if (!runId) {
      setIslandInfo(null);
      setSiblingStats({});
      setSiblingSummaries([]);
    }
  }, [runId]);

  useEffect(() => {
    if (!runId || summary?.summary.status !== "running") {
      return;
    }
    const intervalId = window.setInterval(() => {
      void loadOverview(runId, true);
    }, 30000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [runId, summary?.summary.status]);


  const occupancyOption = useMemo(
    () => ({
      animation: false,
      tooltip: { trigger: "axis" },
      grid: { left: 52, right: 18, top: 40, bottom: 42 },
      xAxis: {
        type: "value",
        splitNumber: 6,
        min: "dataMin",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      series: [
        {
          type: "line",
          smooth: true,
          symbolSize: 4,
          lineStyle: { color: "#2563eb", width: 3 },
          data: thinStats((timeseries?.stats ?? []).filter((entry) => Number(entry.step ?? -1) >= 0)).map((entry) => [
            Number(entry.step ?? 0),
            occupancyMode === "percent"
              ? occupancyPercent(Number(entry.archive_occupancy ?? 0), Number(summary?.config?.num_centroids ?? 0))
              : Number(entry.archive_occupancy ?? 0),
          ]),
        },
      ],
    }),
    [occupancyMode, summary?.config?.num_centroids, timeseries],
  );

  const occupancyMultiOption = useMemo(() => {
    const allIslandIds = [islandInfo?.island_id, ...((islandInfo?.sibling_run_ids ?? []))].filter((id): id is string => Boolean(id)).sort();
    return {
      animation: false,
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#f4f4ef" } },
      grid: { left: 52, right: 18, top: 48, bottom: 42 },
      xAxis: {
        type: "value",
        splitNumber: 6,
        min: "dataMin",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      series: allIslandIds.map((islandId, idx) => {
        const stats = siblingStats[islandId] ?? [];
        return {
          name: islandShortLabel(islandId, idx),
          type: "line",
          smooth: true,
          symbolSize: 4,
          lineStyle: { color: islandColor(islandId), width: 2 },
          itemStyle: { color: islandColor(islandId) },
          data: thinStats(stats.filter((entry) => Number(entry.step ?? -1) >= 0)).map((entry) => [
            Number(entry.step ?? 0),
            occupancyMode === "percent"
              ? occupancyPercent(Number((entry as Record<string, unknown>).archive_occupancy ?? 0), Number(summary?.config?.num_centroids ?? 0))
              : Number((entry as Record<string, unknown>).archive_occupancy ?? 0),
          ]),
        };
      }),
    };
  }, [islandInfo, siblingStats, occupancyMode, summary?.config?.num_centroids]);

  const stageMarkers = useMemo<StageMarker[]>(() => {
    const events = timeseries?.events ?? timeseries?.timeline_events ?? [];
    return events
      .filter((event) => event.type === "stage_end" && typeof event.step === "number" && typeof (event as Record<string, unknown>).stage === "number")
      .map((event) => ({ step: event.step as number, stage: (event as Record<string, unknown>).stage as number }));
  }, [timeseries]);

  const stageEndEvents = useMemo(() => {
    const events = timeseries?.events ?? timeseries?.timeline_events ?? [];
    return events.filter((event) => event.type === "stage_end");
  }, [timeseries]);

  const totalStages = summary?.summary.total_stages ?? 1;

  const islandStats = useMemo(() => {
    if (!islandInfo?.island_id) return {};
    const result: Record<string, Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>> = {};
    if (islandInfo.island_id && timeseries) {
      result[islandInfo.island_id] = thinStats(timeseries.stats as Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>);
    }
    Object.entries(siblingStats).forEach(([id, s]) => {
      if (id !== islandInfo.island_id) {
        result[id] = thinStats(s);
      }
    });
    return result;
  }, [islandInfo, timeseries, siblingStats]);

  const islandIds = Object.keys(islandStats).sort();

  // Aggregated header values across all islands (used in all-islands mode)
  const allIslandSummaries = useMemo(() => {
    if (!summary) return [];
    return [summary, ...siblingSummaries];
  }, [summary, siblingSummaries]);

  const aggregatedSummary = useMemo(() => {
    if (!islandMode || islandMode !== "all" || islandIds.length < 2 || allIslandSummaries.length === 0) return null;
    const sums = allIslandSummaries.map((s) => s.summary);
    const maxStep = Math.max(...sums.map((s) => s.current_step ?? 0));
    const totalCandidates = sums.reduce((acc, s) => acc + (s.candidate_count ?? 0), 0);
    const totalCells = sums.reduce((acc, s) => acc + (s.archive_occupancy ?? 0), 0);
    const bestPrimary = sums.reduce((best, s) => {
      const v = s.best_primary_fitness;
      return v != null && (best == null || v > best) ? v : best;
    }, null as number | null);
    const bestSecondary = sums.reduce((best, s) => {
      const v = s.best_secondary_fitness;
      return v != null && (best == null || v > best) ? v : best;
    }, null as number | null);
    return { maxStep, totalCandidates, totalCells, bestPrimary, bestSecondary };
  }, [islandMode, islandIds.length, allIslandSummaries]);

  const primaryMetricName = metricLabel(summary?.primary_metric_label, "Primary");
  const primaryValidationMetricName = metricLabel(summary?.primary_validation_metric_label, "Primary Validation");
  const secondaryMetricName = metricLabel(summary?.secondary_metric_label, "Secondary");
  const secondaryValidationMetricName = metricLabel(summary?.secondary_validation_metric_label, "Secondary Validation");
  const showPrimaryValidation =
    timeseries?.stats.some((entry) => entry.best_primary_validation_fitness != null) === true ||
    summary?.summary.best_primary_validation_fitness != null;
  const showSecondary =
    hasSecondaryMetric(summary?.secondary_metric_label) &&
    ((compare?.secondary_scatter?.length ?? 0) > 0 || timeseries?.stats.some((entry) => entry.best_secondary_fitness != null) === true);
  const showSecondaryValidation =
    hasSecondaryMetric(summary?.secondary_metric_label) &&
    (timeseries?.stats.some((entry) => entry.best_secondary_validation_fitness != null) === true ||
      summary?.summary.best_secondary_validation_fitness != null);

  const scatterCorrelation = useMemo(() => {
    const pts = (compare?.secondary_scatter ?? []).filter(
      (e) => typeof e.primary_fitness === "number" && typeof e.secondary_fitness === "number",
    );
    if (pts.length < 3) return null;
    const xs = pts.map((e) => e.primary_fitness as number);
    const ys = pts.map((e) => e.secondary_fitness as number);
    const rank = (arr: number[]) => {
      const sorted = [...arr].map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const ranks = new Array(arr.length);
      sorted.forEach(({ i }, r) => { ranks[i] = r + 1; });
      return ranks;
    };
    const rx = rank(xs), ry = rank(ys);
    const n = xs.length;
    const mx = rx.reduce((s, v) => s + v, 0) / n;
    const my = ry.reduce((s, v) => s + v, 0) / n;
    const num = rx.reduce((s, v, i) => s + (v - mx) * (ry[i] - my), 0);
    const dx = Math.sqrt(rx.reduce((s, v) => s + (v - mx) ** 2, 0));
    const dy = Math.sqrt(ry.reduce((s, v) => s + (v - my) ** 2, 0));
    return dx * dy === 0 ? null : num / (dx * dy);
  }, [compare]);

  const scatterOption = useMemo(
    () => ({
      animation: false,
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const data = (params as { data?: { candidateId?: string; step?: number; value?: [number, number] } }).data;
          if (!data) {
            return "";
          }
          return [
            data.candidateId ?? "candidate",
            `Step ${data.step ?? "-"}`,
            `${primaryMetricName}: ${format(typeof data.value?.[0] === "number" ? data.value[0] : null)}`,
            `${secondaryMetricName}: ${format(typeof data.value?.[1] === "number" ? data.value[1] : null)}`,
          ].join("<br/>");
        },
      },
      graphic: scatterCorrelation !== null ? [
        {
          type: "text",
          right: 24,
          top: 10,
          style: {
            text: `ρ = ${scatterCorrelation.toFixed(3)}`,
            fill: Math.abs(scatterCorrelation) > 0.6 ? "#4ade80" : Math.abs(scatterCorrelation) > 0.3 ? "#f59e0b" : "#94a3b8",
            fontSize: 13,
            fontWeight: "bold",
          },
        },
      ] : [],
      grid: { left: 48, right: 20, top: 36, bottom: 42 },
      xAxis: {
        type: "value",
        splitNumber: 6,
        min: "dataMin",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
      },
      yAxis: {
        type: "value",
        splitNumber: 6,
        min: "dataMin",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      visualMap: {
        type: "continuous",
        min: Math.min(...(compare?.secondary_scatter ?? []).map((entry) => Number(entry.created_at_step ?? 0)), 0),
        max: Math.max(...(compare?.secondary_scatter ?? []).map((entry) => Number(entry.created_at_step ?? 0)), 1),
        dimension: 2,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        text: ["Later", "Earlier"],
        textStyle: { color: "#d9d5c6" },
        inRange: {
          color: ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"],
        },
      },
      series: [
        {
          type: "scatter",
          symbolSize: 6,
          data: (compare?.secondary_scatter ?? []).map((entry) => ({
            value: [entry.primary_fitness, entry.secondary_fitness, entry.created_at_step ?? 0],
            candidateId: entry.candidate_id,
            step: entry.created_at_step ?? 0,
          })),
        },
      ],
    }),
    [compare, primaryMetricName, secondaryMetricName, scatterCorrelation],
  );

  const recordTimelineOption = useMemo(
    () => ({
      animation: false,
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const data = (params as { data?: { candidateId?: string; step?: number; value?: [number, number] } }).data;
          if (!data) {
            return "";
          }
          return [
            data.candidateId ?? "candidate",
            `Step ${data.step ?? "-"}`,
            `Fitness: ${format(typeof data.value?.[1] === "number" ? data.value[1] : null)}`,
          ].join("<br/>");
        },
      },
      legend: { textStyle: { color: "#f4f4ef" } },
      grid: { left: 52, right: 20, top: 48, bottom: 40 },
      xAxis: {
        type: "value",
        splitNumber: 6,
        min: "dataMin",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#d9d5c6" },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      series: [
        {
          name: bestMetricLabel(summary?.primary_metric_label, "Primary"),
          type: "line",
          showSymbol: true,
          symbolSize: 8,
          data: (compare?.record_timeline?.primary ?? []).filter((entry) => Number(entry.created_at_step ?? -1) >= 0).map((entry) => ({
            value: [entry.created_at_step, entry.fitness],
            candidateId: entry.candidate_id,
            step: entry.created_at_step,
          })),
          lineStyle: { color: "#d97706", width: 3 },
          itemStyle: { color: "#d97706" },
        },
        ...(showPrimaryValidation
          ? [
              {
                name: bestMetricLabel(summary?.primary_validation_metric_label, "Primary Validation"),
                type: "line",
                showSymbol: true,
                symbolSize: 8,
                data: (compare?.record_timeline?.primary_validation ?? []).filter((entry) => Number(entry.created_at_step ?? -1) >= 0).map((entry) => ({
                  value: [entry.created_at_step, entry.fitness],
                  candidateId: entry.candidate_id,
                  step: entry.created_at_step,
                })),
                lineStyle: { color: "#dc2626", width: 2, type: "dashed" },
                itemStyle: { color: "#dc2626" },
              },
            ]
          : []),
        ...(showSecondary
          ? [
              {
                name: bestMetricLabel(summary?.secondary_metric_label, "Secondary"),
                type: "line",
                showSymbol: true,
                symbolSize: 8,
                data: (compare?.record_timeline?.secondary ?? []).filter((entry) => Number(entry.created_at_step ?? -1) >= 0).map((entry) => ({
                  value: [entry.created_at_step, entry.fitness],
                  candidateId: entry.candidate_id,
                  step: entry.created_at_step,
                })),
                lineStyle: { color: "#16a34a", width: 3 },
                itemStyle: { color: "#16a34a" },
              },
            ]
          : []),
        ...(showSecondaryValidation
          ? [
              {
                name: bestMetricLabel(summary?.secondary_validation_metric_label, "Secondary Validation"),
                type: "line",
                showSymbol: true,
                symbolSize: 8,
                data: (compare?.record_timeline?.secondary_validation ?? []).map((entry) => ({
                  value: [entry.created_at_step, entry.fitness],
                  candidateId: entry.candidate_id,
                  step: entry.created_at_step,
                })),
                lineStyle: { color: "#0f766e", width: 2, type: "dashed" },
                itemStyle: { color: "#0f766e" },
              },
            ]
          : []),
      ],
    }),
    [
      compare,
      showPrimaryValidation,
      showSecondary,
      showSecondaryValidation,
      summary?.primary_metric_label,
      summary?.primary_validation_metric_label,
      summary?.secondary_metric_label,
      summary?.secondary_validation_metric_label,
    ],
  );

  const recordTimelineMultiOption = useMemo(() => {
    const allIslandIds = [islandInfo?.island_id, ...((islandInfo?.sibling_run_ids ?? []))].filter((id): id is string => Boolean(id)).sort();
    return {
      animation: false,
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const p = params as { seriesName?: string; data?: [number, number] };
          if (!p?.data) return "";
          return [`Island: ${p.seriesName ?? "-"}`, `Step: ${p.data[0]}`, `Fitness: ${format(p.data[1])}`].join("<br/>");
        },
      },
      legend: { textStyle: { color: "#f4f4ef" } },
      grid: { left: 52, right: 20, top: 48, bottom: 40 },
      xAxis: {
        type: "value",
        splitNumber: 6,
        min: "dataMin",
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#d9d5c6" },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      series: allIslandIds.map((islandId, idx) => {
        const stats = islandStats[islandId] ?? [];
        let runningBest: number | null = null;
        const points: [number, number][] = [];
        for (const entry of stats) {
          const val = entry.best_primary_fitness;
          if (val != null && (runningBest === null || val > runningBest)) {
            runningBest = val;
            points.push([Number(entry.step ?? 0), val]);
          }
        }
        return {
          name: islandShortLabel(islandId, idx),
          type: "line",
          showSymbol: true,
          symbolSize: 6,
          lineStyle: { color: islandColor(islandId), width: 2 },
          itemStyle: { color: islandColor(islandId) },
          data: points,
        };
      }),
    };
  }, [islandInfo, islandStats]);

  const timingOption = useMemo(() => {
    const records = timing ?? [];
    if (records.length === 0) return null;

    // Rolling average over last 5 candidates
    const WINDOW = 5;
    const smooth = (values: (number | null)[]) => {
      return values.map((_, i) => {
        const slice = values.slice(Math.max(0, i - WINDOW + 1), i + 1).filter((v): v is number => v !== null);
        return slice.length > 0 ? slice.reduce((a, b) => a + b, 0) / slice.length : null;
      });
    };

    const xs = records.map((r) => r.n_candidates);
    const batchSize = Math.max(1, Number((summary?.config as Record<string, unknown>)?.fitness_batch_size ?? 1));
    const wallSmooth     = smooth(records.map((r) => r.wall_seconds));
    const genSmooth      = smooth(records.map((r) => r.generation_seconds));
    const evalTotalSmooth = smooth(records.map((r) => r.eval_seconds ?? null));
    const evalPerCandSmooth = batchSize > 1
      ? smooth(records.map((r) => r.eval_seconds != null ? r.eval_seconds / batchSize : null))
      : null;

    const makeSeries = (name: string, data: (number | null)[], color: string, dashed = false) => ({
      name,
      type: "line",
      smooth: true,
      showSymbol: false,
      lineStyle: { color, width: 2, ...(dashed ? { type: "dashed" } : {}) },
      itemStyle: { color },
      data: xs.map((x, i) => data[i] !== null ? [x, data[i]] : null).filter(Boolean),
    });

    return {
      animation: false,
      tooltip: { trigger: "axis", formatter: (params: unknown) => {
        const items = params as Array<{ seriesName: string; value: [number, number] }>;
        if (!items?.length) return "";
        return [`Candidates: ${items[0].value[0]}`, ...items.map((p) => `${p.seriesName}: ${p.value[1].toFixed(1)}s`)].join("<br/>");
      }},
      legend: { textStyle: { color: "#f4f4ef" } },
      grid: { left: 52, right: 20, top: 48, bottom: 40 },
      xAxis: { type: "value",
        min: "dataMin", axisLabel: { color: "#d9d5c6" }, axisLine: { lineStyle: { color: "#5f6b68" } } },
      yAxis: { type: "value",
        axisLabel: { color: "#d9d5c6" }, splitLine: { lineStyle: { color: "#2b3836" } } },
      series: [
        makeSeries("Wall time", wallSmooth, "#d97706"),
        makeSeries("Generation (LLM)", genSmooth, "#2563eb"),
        makeSeries("Evaluation (total)", evalTotalSmooth, "#16a34a"),
        ...(evalPerCandSmooth ? [makeSeries(`Evaluation (÷${batchSize} per cand)`, evalPerCandSmooth, "#16a34a", true)] : []),
      ],
    };
  }, [timing]);

  if (!runId) {
    return <NoRunSelected message="Choose a run on Home, then use the left navigation to inspect its overview." />;
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }
  if (!summary || !timeseries || !compare) {
    return (
      <>
        <LoadingBar active={loading} label="Loading run overview" />
        <div className="loading-panel">Loading run overview...</div>
      </>
    );
  }

  async function openCandidateDetail(candidateId: string | null | undefined) {
    if (!runId || !candidateId) {
      return;
    }
    try {
      const response = await api.getCandidate(runId, candidateId);
      setSelectedCandidateDetail(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleScatterClick(params: unknown) {
    if (!params || typeof params !== "object") {
      return;
    }
    const data = (params as { data?: { candidateId?: string } }).data;
    if (data?.candidateId) {
      void openCandidateDetail(data.candidateId);
    }
  }

  function handleRefreshRun() {
    if (!runId) {
      return;
    }
    void loadOverview(runId, true);
  }

  return (
    <div className="page-grid overview-page-grid">
      <LoadingBar active={loading} label="Loading run overview" />
      <section className="panel compact-overview-panel panel-full-width">
        <div className="panel-header">
          <div className="overview-header-main">
            <div className="compact-overview-header">
              <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
                <h1 style={{ margin: 0 }}>{summary.summary.project_id}</h1>
                <span style={{ color: "var(--color-text-muted, #888)", fontSize: "0.85rem", fontWeight: 400 }}>{formatRunStamp(summary.summary.created_at)}</span>
                {(summary.summary.total_stages ?? 1) > 1 ? (
                  <span className="stage-badge">Stage {summary.summary.current_stage ?? "?"} / {summary.summary.total_stages}</span>
                ) : null}
              </div>
            </div>
            <div className="overview-header-actions">
              <button type="button" className="link-button ghost panel-toggle-button" onClick={handleRefreshRun}>
                Refresh Run
              </button>
            </div>
          </div>
        </div>
        <div className="overview-header-sections">
          <OverviewKeyValueSection
            title=""
            items={[
              ["Step", String(aggregatedSummary ? aggregatedSummary.maxStep : summary.summary.current_step)],
              ["Candidates", String(aggregatedSummary ? aggregatedSummary.totalCandidates : summary.summary.candidate_count)],
              ["Cells", String(aggregatedSummary ? aggregatedSummary.totalCells : summary.summary.archive_occupancy)],
              [bestMetricLabel(summary.primary_metric_label, "Primary"), format(aggregatedSummary ? aggregatedSummary.bestPrimary : summary.summary.best_primary_fitness)],
              ...(showPrimaryValidation
                ? [[bestMetricLabel(summary.primary_validation_metric_label, "Primary Validation"), format(summary.summary.best_primary_validation_fitness)] as [string, ReactNode]]
                : []),
              [
                `Top ${metricLabel(summary.primary_metric_label, "Primary")}`,
                bestPrimaryCandidate ? (
                  <Link className="metric-link" to={`/candidates/${bestPrimaryCandidate.id}${(bestPrimaryCandidate as Record<string, unknown>).island_run_id ? `?run=${(bestPrimaryCandidate as Record<string, unknown>).island_run_id}` : ""}`}>
                    {bestPrimaryCandidate.id}
                  </Link>
                ) : (
                  "unavailable"
                ),
              ],
              ...formatSamplingConfig(summary.config),
              ["Prompting", formatPromptMethod(summary.config)],
              ["Workers", formatConfigValue(summary.config.parallel_workers)],
              ...((summary.config as Record<string, unknown>).fitness_batch_size != null
                ? [["Eval Batch", formatConfigValue((summary.config as Record<string, unknown>).fitness_batch_size)] as [string, ReactNode]]
                : []),
              ["Max Steps", formatConfigValue(summary.config.max_steps)],
              ["Centroids", formatConfigValue(summary.config.num_centroids)],
              ["Init Random", formatConfigValue(summary.config.initial_random_steps)],
              ["Descriptors", summary.descriptor_labels.join(", ")],
            ]}
          />
        </div>
      </section>
      {islandMode === "all" && islandIds.length >= 2 ? (
        <>
          <section className="panel panel-full-width">
            <div className="panel-header">
              <div className="title-with-help">
                <h3>Fitness Over Time</h3>
                <HelpBadge text="Best is best-so-far and can remain flat while search continues. Mean is the mean fitness of current archive elites, not all generated candidates. Check Archive Occupancy to see whether cells are changing." />
              </div>
            </div>
            <IslandFitnessChart islandStats={islandStats} islandIds={islandIds} />
          </section>
          <section className="panel panel-half">
            <div className="panel-header">
              <div className="title-with-help">
                <h3>Archive Occupancy</h3>
                <HelpBadge text="Shows how many archive cells are occupied over time per island." />
              </div>
              <button type="button" className="link-button ghost panel-toggle-button" onClick={() => setOccupancyMode((current) => (current === "count" ? "percent" : "count"))}>
                {occupancyMode === "count" ? "Show %" : "Show Count"}
              </button>
            </div>
            <EChart option={occupancyMultiOption} />
          </section>
        </>
      ) : (
        <>
          <div className="panel panel-full-width">
            <div className="panel-header">
              <div className="title-with-help">
                <h3>Fitness Over Time</h3>
                <HelpBadge text="Tracks best and mean fitness across steps." />
              </div>
            </div>
            <FitnessChart
              stats={thinStats(timeseries.stats)}
              secondaryMarkers={timeseries.secondary_eval_markers}
              primaryMetricLabel={summary.primary_metric_label}
              primaryValidationMetricLabel={summary.primary_validation_metric_label}
              secondaryMetricLabel={summary.secondary_metric_label}
              secondaryValidationMetricLabel={summary.secondary_validation_metric_label}
              stageMarkers={stageMarkers.length > 0 ? stageMarkers : undefined}
            />
          </div>
          <div className="panel panel-half">
            <div className="panel-header">
              <div className="title-with-help">
                <h3>Archive Occupancy</h3>
                <HelpBadge text="Shows how many archive cells are occupied over time. Rising occupancy means the search is covering more of feature space." />
              </div>
              <button type="button" className="link-button ghost panel-toggle-button" onClick={() => setOccupancyMode((current) => (current === "count" ? "percent" : "count"))}>
                {occupancyMode === "count" ? "Show %" : "Show Count"}
              </button>
            </div>
            <EChart option={occupancyOption} />
          </div>
        </>
      )}

      {timingOption ? (
        <div className="panel panel-half">
          <div className="panel-header">
            <div className="title-with-help">
              <h3>Candidate Timing</h3>
              <HelpBadge text="Rolling average over last 5 candidates, broken down into LLM generation time and evaluation time. X-axis is cumulative candidates evaluated." />
            </div>
          </div>
          <EChart option={timingOption} />
        </div>
      ) : null}
      {totalStages > 1 && stageEndEvents.length > 0 ? (
        <section className="panel">
          <div className="panel-header">
            <h3>Stages</h3>
          </div>
          <div className="stage-cards-row">
            {stageEndEvents.map((event) => {
              const ev = event as Record<string, unknown>;
              const stageNum = typeof ev.stage === "number" ? ev.stage : "?";
              const bestFitness = typeof ev.best_primary_fitness === "number" ? ev.best_primary_fitness.toFixed(4) : "-";
              const archiveOcc = typeof ev.archive_occupancy === "number" ? ev.archive_occupancy : "-";
              const seedsSelected = typeof ev.seeds_selected === "number" ? ev.seeds_selected : "-";
              return (
                <div key={String(stageNum)} className="stage-card">
                  <span className="stage-card-label">Stage {stageNum}</span>
                  <div className="stage-card-row"><span>Best Primary</span><strong>{bestFitness}</strong></div>
                  <div className="stage-card-row"><span>Archive Cells</span><strong>{String(archiveOcc)}</strong></div>
                  <div className="stage-card-row"><span>Seeds</span><strong>{String(seedsSelected)}</strong></div>
                </div>
              );
            })}
          </div>
        </section>
      ) : null}
      <section className="panel panel-half">
        <div className="panel-header">
          <div className="title-with-help">
            <h3>Top Candidates Over Time</h3>
            <HelpBadge text="Shows only the moments when a new best-so-far candidate appears for primary fitness and, when present, secondary fitness. Click a marker to inspect that candidate." />
          </div>
        </div>
        {islandMode === "all" && islandIds.length >= 2 ? (
          <EChart option={recordTimelineMultiOption} />
        ) : (
          <EChart option={recordTimelineOption} onClick={handleScatterClick} />
        )}
      </section>
      <section className="overview-timeline-panel panel-half">
        <EventTimeline
          className="timeline-panel-compact"
          events={timeseries.timeline_events}
        />
      </section>
      {selectedCandidateDetail ? (
        <CandidateDetailModal detail={selectedCandidateDetail} onClose={() => setSelectedCandidateDetail(null)} />
      ) : null}
    </div>
  );
}

function format(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return value.toFixed(4);
}

function occupancyPercent(occupied: number, total: number) {
  if (total <= 0) {
    return 0;
  }
  return (occupied / total) * 100;
}

function formatRunStamp(createdAt?: string) {
  if (!createdAt) {
    return "Unknown time";
  }
  const date = new Date(createdAt);
  return Number.isNaN(date.getTime()) ? createdAt : date.toLocaleString();
}

function NoRunSelected({ message }: { message: string }) {
  return (
    <section className="panel empty-state">
      <h2>No Run Selected</h2>
      <p>{message}</p>
    </section>
  );
}

function buildTopCandidateParams(sortField: string, secondaryOnly = false) {
  const params = new URLSearchParams();
  params.set("limit", "1");
  params.set("sort", `${sortField}:desc`);
  params.set("include_code", "false");
  if (secondaryOnly) {
    params.set("secondary_evaluated", "true");
  }
  return params;
}

function OverviewKeyValueSection({
  title,
  items,
}: {
  title: string;
  items: Array<[string, ReactNode]>;
}) {
  return (
    <section className="overview-kv-section">
      {title ? <h3>{title}</h3> : null}
      <div className="overview-kv-grid">
        {items.map(([label, value]) => (
          <div key={label} className="overview-kv-item">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatConfigValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function formatSamplingMethod(config: Record<string, unknown>) {
  const generationMethod = readNonEmptyString(config.generation_method) ?? "standard";
  const gridStrategy = formatStrategyWithWeights(
    readNonEmptyString(config.grid_selection_strategy) ?? "uniform",
    config.grid_sampling_weights,
    config.grid_sampling_schedule,
  );
  const eliteStrategy = formatStrategyWithWeights(
    readNonEmptyString(config.elite_selection_strategy) ?? "fitness_weighted",
    config.elite_sampling_weights,
    config.elite_sampling_schedule,
  );
  const parts = [`grid ${gridStrategy}`, `elite ${eliteStrategy}`];
  if (generationMethod === "llm_emitters") {
    const emitterNames = Array.isArray(config.emitters)
      ? config.emitters
          .map((entry) =>
            entry && typeof entry === "object" && "name" in entry && typeof entry.name === "string"
              ? entry.name
              : null,
          )
          .filter((entry): entry is string => Boolean(entry))
      : [];
    parts.push(`generation emitters${emitterNames.length ? ` (${emitterNames.join(", ")})` : ""}`);
  } else {
    parts.push(`generation ${generationMethod}`);
  }
  return parts.join(" · ");
}

function formatPromptMethod(config: Record<string, unknown>) {
  const ancestorCount = formatConfigValue(config.ancestor_count);
  const inspirationCount = formatConfigValue(config.inspiration_elite_count);
  return `single elite diff · ${ancestorCount} ancestors · ${inspirationCount} inspirations`;
}

function formatStrategyWithWeights(strategy: string, weights: unknown, schedule: unknown) {
  if (strategy !== "weighted") {
    return strategy;
  }
  if (Array.isArray(schedule) && schedule.length > 0) {
    const phases = schedule
      .map((phase) => {
        if (!phase || typeof phase !== "object") {
          return null;
        }
        const untilStep = "until_step" in phase ? phase.until_step : null;
        const label = formatWeightList("weights" in phase ? phase.weights : null);
        if (!label) {
          return null;
        }
        return untilStep == null ? `${label} thereafter` : `${label} to ${untilStep}`;
      })
      .filter((entry): entry is string => Boolean(entry));
    if (phases.length > 0) {
      return `weighted: ${phases.join("; ")}`;
    }
  }
  const weightLabel = formatWeightList(weights);
  return weightLabel ? `weighted: ${weightLabel}` : "weighted";
}

function formatWeightList(weights: unknown) {
  if (!Array.isArray(weights) || weights.length === 0) {
    return "";
  }
  const entries = weights
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }
      const strategy = "strategy" in entry && typeof entry.strategy === "string" ? entry.strategy : null;
      const weight = "weight" in entry ? entry.weight : null;
      if (!strategy) {
        return null;
      }
      if (typeof weight === "number") {
        return `${strategy} ${trimNumber(weight)}`;
      }
      return strategy;
    })
    .filter((entry): entry is string => Boolean(entry));
  return entries.join(" + ");
}

function trimNumber(value: number) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function readNonEmptyString(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}

function formatSecondaryEvalConfig(
  config: Record<string, unknown>,
  secondaryMetricLabel?: string | null,
): Array<[string, ReactNode]> {
  const sec = (config.secondary_eval && typeof config.secondary_eval === "object"
    ? config.secondary_eval
    : config) as Record<string, unknown>;

  const stages = sec.stages ?? config.stages;
  const p1 = sec.p1 ?? config.p1;
  const p2 = sec.p2 ?? config.p2;

  const hasAny = stages != null || p1 != null || p2 != null || secondaryMetricLabel;
  if (!hasAny) return [];

  const rows: Array<[string, ReactNode]> = [];

  if (secondaryMetricLabel) {
    rows.push(["Secondary Metric", secondaryMetricLabel]);
  }
  if (stages != null) {
    rows.push(["Secondary Stages", formatConfigValue(stages)]);
  }
  if (p1 != null) {
    rows.push(["Secondary P1 (promote)", formatConfigValue(p1)]);
  }
  if (p2 != null) {
    rows.push(["Secondary P2 (keep)", formatConfigValue(p2)]);
  }

  return rows;
}


function formatSamplingConfig(config: Record<string, unknown>): Array<[string, ReactNode]> {
  const sampling = config.sampling as Record<string, unknown> | undefined;
  const rows: Array<[string, ReactNode]> = [];
  if (sampling) {
    const cellMethod = String(sampling.cell_method ?? "ucb");
    const emitterMethod = String(sampling.emitter_method ?? "single_curiosity");
    const ucbC = sampling.cell_ucb_c;
    const meanDiffT = sampling.cell_mean_diff_temperature;
    const warmup = sampling.emitter_cell_warmup_trials;
    const pExploit = sampling.emitter_p_exploit;
    let cellLabel = cellMethod;
    if (cellMethod === "mean_diff" && meanDiffT != null) cellLabel = `mean_diff (T=${meanDiffT})`;
    else if (cellMethod === "ucb" && ucbC != null) cellLabel = `ucb (c=${ucbC})`;
    const momentumBeta = sampling.momentum_ema_beta;
    const momentumLo = sampling.momentum_exploit_lo;
    const momentumHi = sampling.momentum_exploit_hi;
    let emitterLabel = emitterMethod;
    if (emitterMethod === "fixed" && pExploit != null) emitterLabel = `fixed (exploit=${pExploit}, explore=${+(1 - +pExploit).toFixed(2)})`;
    else if (emitterMethod === "single_curiosity" && warmup != null) emitterLabel = `single_curiosity (warmup=${warmup})`;
    else if (emitterMethod === "momentum") emitterLabel = `momentum (lo=${momentumLo ?? 0.2}, hi=${momentumHi ?? 0.8}, β=${momentumBeta ?? 0.05})`;
    rows.push(["Cell Sampling", cellLabel]);
    rows.push(["Emitter Sampling", emitterLabel]);
  } else {
    const genMethod = readNonEmptyString(config.generation_method) ?? "standard";
    rows.push(["Cell Sampling", "uniform (legacy)"]);
    rows.push(["Emitter Sampling", genMethod.replace("llm_emitters_emitter_curiosity", "curiosity").replace("llm_emitters", "emitters")]);
  }
  return rows;
}
