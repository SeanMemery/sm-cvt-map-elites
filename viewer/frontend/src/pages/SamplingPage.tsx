import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Delaunay } from "d3-delaunay";
import { useNavigate } from "react-router-dom";

import { api } from "../api";
import { EChart } from "../components/EChart";
import { HelpBadge } from "../components/HelpBadge";
import { LoadingBar } from "../components/LoadingBar";
import { MetricSelector } from "../components/MetricSelector";
import type { EventRecord, RunSummaryResponse, SnapshotResponse } from "../types";


type SamplingPageProps = {
  runId: string | null;
  islandMode: string;
};

export function SamplingPage({ runId, islandMode }: SamplingPageProps) {
  const navigate = useNavigate();
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [snapshotSteps, setSnapshotSteps] = useState<number[]>([]);
  const [step, setStep] = useState<number | null>(null);
  const [stepToStage, setStepToStage] = useState<Map<number, number>>(new Map());
  const [snapshot, setSnapshot] = useState<SnapshotResponse | null>(null);
  const [xFeature, setXFeature] = useState("0");
  const [yFeature, setYFeature] = useState("1");
  const [samplingView, setSamplingView] = useState("cell_reward");
  const [loading, setLoading] = useState(false);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const snapshotCacheRef = useRef<Map<number, SnapshotResponse>>(new Map());

  useEffect(() => {
    snapshotCacheRef.current = new Map();
  }, [runId]);

  useEffect(() => {
    if (!runId) {
      setSummary(null);
      setSnapshotSteps([]);
      setStep(null);
      setSnapshot(null);
      setStepToStage(new Map());
      setError(null);
      return;
    }
    const activeRunId = runId;
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [summaryResponse, stepsResponse, timeseriesResponse] = await Promise.all([
          api.getRunSummary(activeRunId),
          api.getSnapshotSteps(activeRunId),
          api.getTimeseries(activeRunId),
        ]);
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
        if (cancelled) {
          return;
        }
        setSummary(summaryResponse);
        setSnapshotSteps(stepsResponse.steps);
        setStepToStage(stageMap);
        setXFeature("0");
        setYFeature(summaryResponse.descriptor_labels.length > 1 ? "1" : "0");
        setSamplingView("cell_reward");
        const latestStep = stepsResponse.steps.length ? stepsResponse.steps[stepsResponse.steps.length - 1] : null;
        setStep(latestStep);
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
  }, [runId]);

  useEffect(() => {
    if (!runId || step == null) {
      setSnapshot(null);
      return;
    }
    const cached = snapshotCacheRef.current.get(step);
    if (cached) {
      setSnapshot(cached);
      return;
    }
    const activeRunId = runId;
    const selectedStep = step;
    let cancelled = false;
    async function loadSnapshot() {
      setSnapshotLoading(true);
      try {
        const response = await api.getSnapshot(activeRunId, selectedStep);
        snapshotCacheRef.current.set(selectedStep, response);
        if (!cancelled) {
          setSnapshot(response);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setSnapshotLoading(false);
        }
      }
    }
    void loadSnapshot();
    return () => {
      cancelled = true;
    };
  }, [runId, step]);

  const descriptorLabels = summary?.descriptor_labels ?? [];
  const featureOptions = descriptorLabels.map((label, index) => ({
    value: String(index),
    label,
  }));
  // Detect active sampling config
  const samplingConfig = (summary?.config as Record<string, unknown>)?.sampling as Record<string, unknown> | undefined;
  const cellMethod = String(samplingConfig?.cell_method ?? "legacy");
  const emitterMethod = String(samplingConfig?.emitter_method ?? "legacy");

  // Emitter names from config (e.g. ["exploit", "explore"])
  const emitterNames = useMemo(() => {
    const emitters = Array.isArray(summary?.config?.emitters) ? summary?.config?.emitters : [];
    return emitters
      .map((entry) => (entry && typeof entry === "object" ? String((entry as { name?: unknown }).name ?? "") : ""))
      .filter((name) => name.length > 0);
  }, [summary]);

  // Compute relevant view options based on active sampling methods
  const samplingViewOptions = useMemo(() => {
    const opts: { value: string; label: string }[] = [];
    const isNewSampling = samplingConfig != null;

    if (isNewSampling) {
      // New sampling system: show only metrics relevant to the chosen methods
      if (cellMethod === "ucb") {
        opts.push({ value: "cell_reward", label: "Cell Reward EMA" });
        opts.push({ value: "cell_n_trials", label: "Cell Trial Count" });
      }
      if (cellMethod === "mean_diff") {
        opts.push({ value: "mean_diff_prob", label: "Sampling Probability" });
        opts.push({ value: "mean_diff_raw", label: "Mean Diff (fitness − mean)" });
        opts.push({ value: "primary_fitness", label: "Primary Fitness" });
      }
      if (emitterMethod === "single_curiosity") {
        opts.push({ value: "cell_logit_exploit", label: "Exploit Probability" });
      }
      if (emitterMethod === "momentum") {
        opts.push({ value: "cell_reward", label: "Cell Reward EMA" });
        opts.push({ value: "cell_n_trials", label: "Cell Trial Count" });
      }
      if (emitterMethod === "softmax") {
        // Per-emitter softmax logits — show probability for each emitter
        for (const name of emitterNames) {
          opts.push({ value: `emitter_logit:${name}`, label: `P(${name})` });
        }
      }
      if (emitterMethod === "ucb" || emitterMethod === "thompson") {
        opts.push({ value: "overall", label: "Curiosity (overall)" });
        for (const name of emitterNames) {
          opts.push({ value: name, label: `Curiosity (${name})` });
        }
      }
    } else {
      // Legacy run: per-emitter curiosity scores from old system
      if (summary?.summary.curiosity_enabled) {
        opts.push({ value: "overall", label: "Curiosity (overall)" });
        for (const name of emitterNames) {
          opts.push({ value: name, label: `Curiosity (${name})` });
        }
      }
    }
    return opts;
  }, [cellMethod, emitterMethod, emitterNames, samplingConfig, summary?.summary.curiosity_enabled]);

  // Keep selected view valid when options change; for mean_diff default to its primary view
  useEffect(() => {
    if (samplingViewOptions.length && !samplingViewOptions.find((o) => o.value === samplingView)) {
      const defaultView = cellMethod === "mean_diff" ? "mean_diff_prob" : samplingViewOptions[0].value;
      setSamplingView(samplingViewOptions.find((o) => o.value === defaultView) ? defaultView : samplingViewOptions[0].value);
    }
  }, [samplingViewOptions, samplingView, cellMethod]);

  const emitterOptions = samplingViewOptions; // alias for legacy refs below
  const emitterCuriosityEnabled = emitterOptions.length > 1;
  const xIndex = Number.parseInt(xFeature, 10) || 0;
  const yIndex = Number.parseInt(yFeature, 10) || 0;
  const currentStepIndex = step == null ? -1 : snapshotSteps.indexOf(step);

  // Pre-compute mean-diff derived values: mean_diff_raw (fitness - mean) and mean_diff_prob (sampling probability)
  // Stored as two maps so getCuriosityValue can look them up by cell_id
  const meanDiffWeightByIdForStats = useMemo(() => {
    if (cellMethod !== "mean_diff") return { prob: new Map<number, number>(), raw: new Map<number, number>() };
    const fits = (snapshot?.occupied_cells ?? [])
      .map((c) => typeof (c as Record<string, unknown>).primary_fitness === "number" ? (c as Record<string, unknown>).primary_fitness as number : null)
      .filter((f): f is number => f != null);
    if (!fits.length) return { prob: new Map<number, number>(), raw: new Map<number, number>() };
    const mean = fits.reduce((a, b) => a + b, 0) / fits.length;
    const diffs = (snapshot?.occupied_cells ?? []).map((c) => {
      const f = typeof (c as Record<string, unknown>).primary_fitness === "number" ? (c as Record<string, unknown>).primary_fitness as number : mean;
      return { id: c.cell_id, diff: f - mean };
    });
    const minDiff = Math.min(...diffs.map((d) => d.diff));
    const shifted = diffs.map((d) => ({ id: d.id, val: d.diff + Math.abs(minDiff) }));
    const T = typeof samplingConfig?.cell_mean_diff_temperature === "number" ? samplingConfig.cell_mean_diff_temperature : 0.4;
    const weighted = shifted.map((d) => ({ id: d.id, val: d.val > 0 ? Math.pow(d.val, 1 / T) : 0 }));
    const total = weighted.reduce((a, d) => a + d.val, 0);
    const prob = new Map<number, number>();
    const raw = new Map<number, number>();
    for (const d of diffs) raw.set(d.id, d.diff);
    for (const d of weighted) prob.set(d.id, total > 0 ? d.val / total : 0);
    return { prob, raw };
  }, [cellMethod, samplingConfig, snapshot]);

  const chartOption = useMemo(() => {
    const centroids = snapshot?.centroids ?? [];
    const occupiedByCell = new Map((snapshot?.occupied_cells ?? []).map((cell) => [cell.cell_id, cell]));

    // normalizer_bounds: maps [0,1] centroid coords → raw feature units for display
    const normBounds = (snapshot as Record<string, unknown>)?.normalizer_bounds as
      { lower: number[]; upper: number[] } | undefined;
    const rawLower = normBounds?.lower ?? [];
    const rawUpper = normBounds?.upper ?? [];

    // Convert a normalised coordinate to raw using the observed bounds
    const toRaw = (norm: number, dim: number): number => {
      const lo = rawLower[dim];
      const hi = rawUpper[dim];
      if (lo == null || hi == null || hi === lo) return norm;
      return lo + norm * (hi - lo);
    };

    // Use normalised [0,1] coordinates for Voronoi layout so the chart always has a
    // good aspect ratio regardless of raw feature scale differences.
    // For occupied cells, show the candidate's actual descriptor_raw values in the
    // tooltip (not the centroid position) so they match the candidate detail page.
    const centroidPoints = centroids.map((coords, cellId) => {
      const occupied = occupiedByCell.get(cellId) ?? null;
      const raw = occupied?.descriptor_raw;
      return {
        cellId,
        x: Number(coords?.[xIndex] ?? 0),
        y: Number(coords?.[yIndex] ?? 0),
        xRaw: raw?.[xIndex] != null ? raw[xIndex] : toRaw(Number(coords?.[xIndex] ?? 0), xIndex),
        yRaw: raw?.[yIndex] != null ? raw[yIndex] : toRaw(Number(coords?.[yIndex] ?? 0), yIndex),
        occupied,
      };
    });

    const occupiedValues = (snapshot?.occupied_cells ?? [])
      .map((cell) => getCuriosityValue(cell, samplingView, meanDiffWeightByIdForStats))
      .filter((value): value is number => value != null);
    const minCuriosity = occupiedValues.length ? Math.min(...occupiedValues) : 0;
    const maxCuriosity = occupiedValues.length ? Math.max(...occupiedValues) : 1;

    // Chart always spans [0,1] for good aspect ratio
    const [xMin, xMax] = expandRange(0, 1);
    const [yMin, yMax] = expandRange(0, 1);
    const delaunay = centroidPoints.length
      ? Delaunay.from(
          centroidPoints.map((point): [number, number] => [point.x, point.y]),
          (point: [number, number]) => point[0],
          (point: [number, number]) => point[1],
        )
      : null;
    const voronoi = delaunay ? delaunay.voronoi([xMin, yMin, xMax, yMax]) : null;
    const cellPolygons = centroidPoints.map((point, index) => {
      const polygon = voronoi?.cellPolygon(index);
      const cleanedPolygon = polygon
        ? polygon.slice(0, -1).map((point: [number, number]) => [Number(point[0]), Number(point[1])])
        : [];
      const curiosityValue = getCuriosityValue(point.occupied, samplingView, meanDiffWeightByIdForStats);
      return {
        value: [point.x, point.y, Number(curiosityValue ?? minCuriosity)],
        cellId: point.cellId,
        xRaw: point.xRaw,
        yRaw: point.yRaw,
        candidateId: point.occupied?.candidate_id,
        curiosity: curiosityValue,
        occupied: point.occupied != null,
        polygon: cleanedPolygon,
      };
    });
    const emptyCells = cellPolygons.filter((cell) => !cell.occupied);
    const occupiedCells = cellPolygons.filter((cell) => cell.occupied);

    return {
      animation: false,
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const data = (params as { data?: { cellId?: number; candidateId?: string; curiosity?: number } }).data;
          if (!data) {
            return "";
          }
          const xLabel = descriptorLabels[xIndex] ?? `dim ${xIndex}`;
          const yLabel = descriptorLabels[yIndex] ?? `dim ${yIndex}`;
          const xVal = typeof (data as Record<string, unknown>).xRaw === "number" ? ((data as Record<string, unknown>).xRaw as number).toFixed(2) : "-";
          const yVal = typeof (data as Record<string, unknown>).yRaw === "number" ? ((data as Record<string, unknown>).yRaw as number).toFixed(2) : "-";
          return [
            `Cell ${data.cellId ?? "-"}`,
            data.candidateId ? `Candidate ${data.candidateId}` : "Unoccupied",
            `${xLabel}: ${xVal}`,
            `${yLabel}: ${yVal}`,
            data.candidateId ? `${samplingView === "overall" ? "Curiosity" : `${samplingView} Curiosity`} ${formatCuriosity(data.curiosity)}` : "",
          ].filter(Boolean).join("<br/>");
        },
      },
      grid: { left: 52, right: 20, top: 28, bottom: 78 },
      xAxis: {
        type: "value",
        name: descriptorLabels[xIndex] ?? `dim ${xIndex}`,
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: { color: "#7a8f8d", fontSize: 11 },
        min: xMin,
        max: xMax,
        splitNumber: 5,
        axisLabel: {
          color: "#d9d5c6",
          formatter: (v: number) => {
            const raw = toRaw(v, xIndex);
            return raw % 1 === 0 ? String(Math.round(raw)) : raw.toFixed(1);
          },
        },
        axisLine: { lineStyle: { color: "#5f6b68" } },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      yAxis: {
        type: "value",
        name: descriptorLabels[yIndex] ?? `dim ${yIndex}`,
        nameLocation: "middle",
        nameGap: 44,
        nameTextStyle: { color: "#7a8f8d", fontSize: 11 },
        min: yMin,
        max: yMax,
        splitNumber: 5,
        axisLabel: {
          color: "#d9d5c6",
          formatter: (v: number) => {
            const raw = toRaw(v, yIndex);
            return raw % 1 === 0 ? String(Math.round(raw)) : raw.toFixed(1);
          },
        },
        axisLine: { lineStyle: { color: "#5f6b68" } },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      visualMap: {
        type: "continuous",
        min: minCuriosity,
        max: maxCuriosity <= minCuriosity ? minCuriosity + 1e-6 : maxCuriosity,
        dimension: 2,
        orient: "horizontal",
        left: "center",
        bottom: 8,
        text: ["High", "Low"],
        textStyle: { color: "#d9d5c6" },
        inRange: {
          color: ["#2563eb", "#14b8a6", "#f59e0b", "#dc2626"],
        },
      },
      series: [
        {
          name: "Empty Voronoi Cells",
          type: "custom",
          coordinateSystem: "cartesian2d",
          data: emptyCells,
          renderItem: (params: unknown, api: {
            coord: (point: [number, number]) => [number, number];
          }) => renderVoronoiCell(emptyCells, params, api, {
            fill: "rgba(185, 199, 196, 0.12)",
            stroke: "rgba(185, 199, 196, 0.28)",
          }),
          silent: false,
        },
        {
          name: "Occupied Voronoi Cells",
          type: "custom",
          coordinateSystem: "cartesian2d",
          data: occupiedCells,
          cursor: "pointer",
          renderItem: (params: unknown, api: {
            coord: (point: [number, number]) => [number, number];
            visual: (key: string) => unknown;
          }) => renderVoronoiCell(occupiedCells, params, api, {
            fill: String(api.visual("color") ?? "rgba(20, 184, 166, 0.72)"),
            stroke: "rgba(12, 18, 16, 0.52)",
          }),
        },
      ],
    };
  }, [samplingView, snapshot, step, xIndex, yIndex, meanDiffWeightByIdForStats, descriptorLabels]);

  const stats = useMemo(() => {
    const values = (snapshot?.occupied_cells ?? [])
      .map((cell) => getCuriosityValue(cell, samplingView, meanDiffWeightByIdForStats))
      .filter((value): value is number => value != null);
    const occupiedCount = snapshot?.occupied_cells.length ?? 0;
    const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    const min = values.length ? Math.min(...values) : null;
    const max = values.length ? Math.max(...values) : null;
    return { occupiedCount, mean, min, max };
  }, [samplingView, snapshot, meanDiffWeightByIdForStats]);

  const handleCellClick = useCallback((params: unknown) => {
    const data = (params as { data?: { candidateId?: string } }).data;
    if (!data?.candidateId) return;
    // islandMode holds the specific island run_id when one is selected — use it
    // so the candidate always opens in the island whose cells we're viewing.
    const targetRunId = (islandMode && islandMode !== "all" && islandMode !== "single")
      ? islandMode
      : runId;
    if (!targetRunId) return;
    navigate(`/candidates/${data.candidateId}?run=${targetRunId}`);
  }, [navigate, runId, islandMode]);

  if (!runId) {
    return <NoRunSelected message="Choose a run on Home, then use the left navigation to inspect curiosity." />;
  }

  if (islandMode === "all") {
    return (
      <section className="panel empty-state">
        <h2>Select an Island</h2>
        <p>Sampling statistics are per-island. Choose a specific island from the sidebar to view its sampling map.</p>
      </section>
    );
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }

  if (!summary) {
    return (
      <>
        <LoadingBar active={loading} label="Loading sampling view" />
        <div className="loading-panel">Loading sampling view...</div>
      </>
    );
  }

  const hasSamplingViews = samplingViewOptions.length > 0;
  if (!hasSamplingViews) {
    return (
      <section className="panel empty-state">
        <h2>Sampling</h2>
        <p>This run uses uniform/fixed sampling with no bandit or curiosity tracking data.</p>
      </section>
    );
  }

  if (!snapshotSteps.length) {
    return (
      <section className="panel empty-state">
        <h2>No Snapshots</h2>
        <p>This run has no archive snapshots, so sampling cannot be scrubbed over time.</p>
      </section>
    );
  }

  const methodHelpContent = buildMethodHelp(cellMethod, emitterMethod, samplingConfig);
  const viewHelpContent = buildViewHelp(samplingView, cellMethod, emitterMethod);

  return (
    <div className="page-grid">
      <LoadingBar active={loading || snapshotLoading} label="Loading sampling map" />
      <section className="panel">
        <div className="panel-header">
          <div className="title-with-help">
            <h1>Sampling</h1>
            <HelpBadge>{methodHelpContent}</HelpBadge>
          </div>
        </div>
        <div className="controls-grid">
          <MetricSelector label="X Feature" value={xFeature} onChange={setXFeature} options={featureOptions} />
          <MetricSelector label="Y Feature" value={yFeature} onChange={setYFeature} options={featureOptions} />
          <MetricSelector label="View" value={samplingView} onChange={setSamplingView} options={samplingViewOptions} />
        </div>
        {samplingConfig ? (
          <div className="muted compact-feature-list" style={{ marginTop: "0.5rem" }}>
            <strong>Cell:</strong> {String(cellMethod)}
            {" · "}
            <strong>Emitter:</strong> {String(emitterMethod)}
            {cellMethod === "ucb" && samplingConfig.cell_ucb_c != null ? ` · UCB c=${samplingConfig.cell_ucb_c}` : ""}
            {cellMethod === "mean_diff" && samplingConfig.cell_mean_diff_temperature != null ? ` · T=${samplingConfig.cell_mean_diff_temperature}` : ""}
            {emitterMethod === "single_curiosity" && samplingConfig.emitter_cell_warmup_trials != null ? ` · warmup=${samplingConfig.emitter_cell_warmup_trials}` : ""}
            {emitterMethod === "fixed" && samplingConfig.emitter_p_exploit != null ? ` · exploit=${samplingConfig.emitter_p_exploit}` : ""}
          </div>
        ) : null}
        <div className="curiosity-step-row">
          <button
            type="button"
            className="link-button ghost panel-toggle-button"
            onClick={() => setStep((current) => snapshotSteps[Math.max(0, currentStepIndex - 1)] ?? current)}
            disabled={currentStepIndex <= 0}
          >
            Prev
          </button>
          <label className="curiosity-slider-field">
            <span>Snapshot Step {step ?? "-"}{step != null && stepToStage.size > 0 && stepToStage.has(step) ? ` · S${stepToStage.get(step)}` : ""}</span>
            <input
              type="range"
              min="0"
              max={Math.max(snapshotSteps.length - 1, 0)}
              step="1"
              value={Math.max(currentStepIndex, 0)}
              onChange={(event) => {
                const nextIndex = Number.parseInt(event.target.value, 10);
                setStep(snapshotSteps[nextIndex] ?? (snapshotSteps.length ? snapshotSteps[snapshotSteps.length - 1] : null));
              }}
            />
          </label>
          <button
            type="button"
            className="link-button ghost panel-toggle-button"
            onClick={() => setStep((current) => snapshotSteps[Math.min(snapshotSteps.length - 1, currentStepIndex + 1)] ?? current)}
            disabled={currentStepIndex < 0 || currentStepIndex >= snapshotSteps.length - 1}
          >
            Next
          </button>
        </div>
      </section>
      <section className="panel">
        <div className="panel-header">
          <div className="title-with-help">
            <h3>Sampling Map</h3>
            <HelpBadge>{viewHelpContent}</HelpBadge>
          </div>
        </div>
        <EChart option={chartOption} onClick={handleCellClick} />
        <div className="archive-map-stats curiosity-stats">
          <span><strong>Snapshot</strong> {step ?? "-"}</span>
          <span><strong>Occupied</strong> {stats.occupiedCount}</span>
          <span><strong>View</strong> {samplingViewOptions.find(o => o.value === samplingView)?.label ?? samplingView}</span>
          <span><strong>Min</strong> {formatCuriosity(stats.min)}</span>
          <span><strong>Mean</strong> {formatCuriosity(stats.mean)}</span>
          <span><strong>Max</strong> {formatCuriosity(stats.max)}</span>
          {emitterMethod === "momentum" && (() => {
            const rate = typeof (snapshot as Record<string, unknown>)?.insertion_rate_ema === "number"
              ? (snapshot as Record<string, unknown>).insertion_rate_ema as number
              : null;
            const lo = typeof samplingConfig?.momentum_exploit_lo === "number" ? samplingConfig.momentum_exploit_lo : 0.2;
            const hi = typeof samplingConfig?.momentum_exploit_hi === "number" ? samplingConfig.momentum_exploit_hi : 0.8;
            const pExploit = rate != null ? lo + (hi - lo) * rate : null;
            return (
              <>
                <span><strong>Momentum</strong> {rate != null ? rate.toFixed(3) : "-"}</span>
                <span><strong>P(exploit)</strong> {pExploit != null ? pExploit.toFixed(3) : "-"}</span>
              </>
            );
          })()}
        </div>
      </section>
    </div>
  );
}

function getCuriosityValue(
  cell: SnapshotResponse["occupied_cells"][number] | null | undefined,
  samplingView: string,
  meanDiffMaps?: { prob: Map<number, number>; raw: Map<number, number> },
) {
  if (!cell) return null;
  const c = cell as Record<string, unknown>;

  if (samplingView === "mean_diff_prob") {
    return meanDiffMaps?.prob.get(cell.cell_id) ?? null;
  }
  if (samplingView === "mean_diff_raw") {
    return meanDiffMaps?.raw.get(cell.cell_id) ?? null;
  }
  if (samplingView === "primary_fitness") {
    const v = c.primary_fitness;
    return typeof v === "number" ? v : null;
  }
  if (samplingView === "cell_reward") {
    const v = c.cell_reward_ema;
    return typeof v === "number" ? v : null;
  }
  if (samplingView === "cell_n_trials") {
    const v = c.cell_n_trials;
    return typeof v === "number" ? v : null;
  }
  if (samplingView === "cell_logit_exploit") {
    const v = c.cell_logit_exploit;
    if (typeof v !== "number") return null;
    return 1.0 / (1.0 + Math.exp(-v));  // sigmoid → probability of exploit
  }
  // softmax per-emitter probability: emitter_logit:<name>
  if (samplingView.startsWith("emitter_logit:")) {
    const emitterName = samplingView.slice("emitter_logit:".length);
    const logits = c.cell_emitter_logits as Record<string, number> | null | undefined;
    if (!logits || typeof logits !== "object") return null;
    // Compute softmax probability for this emitter
    const allLogits = Object.values(logits).filter((v): v is number => typeof v === "number");
    if (allLogits.length === 0) return null;
    const thisLogit = typeof logits[emitterName] === "number" ? logits[emitterName] : 0;
    const maxL = Math.max(...allLogits);
    const expThis = Math.exp(thisLogit - maxL);
    const expSum = allLogits.reduce((s, l) => s + Math.exp(l - maxL), 0);
    return expSum > 0 ? expThis / expSum : null;
  }
  if (samplingView === "overall") {
    return typeof cell.curiosity_score === "number" ? cell.curiosity_score : null;
  }
  // emitter-specific curiosity (exploit / explore)
  const emitterScores = cell.emitter_curiosity_scores;
  if (!emitterScores || typeof emitterScores !== "object") return null;
  const value = (emitterScores as Record<string, unknown>)[samplingView];
  return typeof value === "number" ? value : null;
}

function formatCuriosity(value: number | null | undefined) {
  if (value == null) {
    return "-";
  }
  return value.toFixed(3);
}

function expandRange(minValue: number, maxValue: number) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [0, 1];
  }
  if (minValue === maxValue) {
    const padding = Math.max(Math.abs(minValue) * 0.1, 1);
    return [minValue - padding, maxValue + padding];
  }
  const padding = (maxValue - minValue) * 0.08;
  return [minValue - padding, maxValue + padding];
}

function renderVoronoiCell(
  cells: Array<{ polygon: number[][] }>,
  params: unknown,
  api: {
    coord: (point: [number, number]) => [number, number];
  },
  style: {
    fill: string;
    stroke: string;
  },
) {
  const dataIndex = (params as { dataIndex?: number }).dataIndex ?? -1;
  const polygon = dataIndex >= 0 ? cells[dataIndex]?.polygon ?? [] : [];
  if (!polygon.length) {
    return null;
  }
  const points = polygon.map(([x, y]) => api.coord([x, y]));
  return {
    type: "polygon",
    shape: { points },
    style: {
      fill: style.fill,
      stroke: style.stroke,
      lineWidth: 1,
    },
  };
}

function buildMethodHelp(cellMethod: string, emitterMethod: string, samplingConfig: Record<string, unknown> | undefined) {
  const cellDesc: Record<string, string> = {
    ucb: `UCB (Upper Confidence Bound) selects cells by balancing exploitation of cells with high past reward against exploration of under-tried cells. The c parameter (currently ${samplingConfig?.cell_ucb_c ?? "default"}) controls this tradeoff — higher c = more exploration.`,
    uniform: "Uniform sampling selects cells at random with equal probability, regardless of past performance.",
    mean_diff: `Mean-Diff sampling selects cells proportional to how much their fitness exceeds the archive mean. Differences are shifted to remove negatives then temperature-scaled (T=${samplingConfig?.cell_mean_diff_temperature ?? 0.4}) — lower T concentrates probability on the best cells; T=1 is purely linear in fitness difference.`,
    emitter_curiosity_weighted: "Curiosity-weighted sampling selects cells proportional to their curiosity score — favouring cells that are underexplored or whose best elite has high improvement potential.",
    legacy: "Legacy uniform sampling — cell is selected at random.",
  };
  const emitterDesc: Record<string, string> = {
    single_curiosity: `Single Curiosity uses a single per-cell bandit logit to decide whether to exploit (refine the best known elite) or explore (generate from an explore emitter). The logit shifts toward exploit when candidates are inserted and toward explore when they are rejected. Warmup trials (${samplingConfig?.emitter_cell_warmup_trials ?? "default"}) gate how quickly this adapts.`,
    fixed: `Fixed emitter selection uses a constant probability — ${samplingConfig?.emitter_p_exploit != null ? `exploit ${samplingConfig.emitter_p_exploit} / explore ${+(1 - +samplingConfig.emitter_p_exploit).toFixed(2)}` : "no adaptive bandit logic"}.`,
    momentum: `Momentum-based selection interpolates P(exploit) from ${samplingConfig?.momentum_exploit_lo ?? 0.2} (stagnating) to ${samplingConfig?.momentum_exploit_hi ?? 0.8} (hot archive) using a per-island EMA of the archive insertion rate (β=${samplingConfig?.momentum_ema_beta ?? 0.05}). High insertion rate = archive still improving = exploit more. Low insertion rate = stagnating = explore more.`,
    ucb: "UCB emitter selection uses Upper Confidence Bound across emitters, favouring those with the best reward-to-uncertainty ratio.",
    thompson: "Thompson sampling picks the emitter by sampling from a Beta distribution fitted to each emitter's success/failure history.",
    softmax: "Softmax sampling maintains a per-emitter logit for each cell. Logits are converted to probabilities via softmax, so all emitters (including non-LLM local emitters) compete on equal footing. Successful emitters have their logit reinforced via policy gradient.",
    legacy: "Legacy generation method — emitter is selected by the run's generation_method setting.",
  };
  return (
    <div className="help-content">
      <p><strong>Cell sampling:</strong> {cellDesc[cellMethod] ?? cellMethod}</p>
      <p><strong>Emitter sampling:</strong> {emitterDesc[emitterMethod] ?? emitterMethod}</p>
      <p className="help-note">Use the View selector to inspect different per-cell metrics. Scrub the step slider to watch how sampling adapts over the run.</p>
    </div>
  );
}

function buildViewHelp(samplingView: string, cellMethod: string, emitterMethod: string) {
  if (samplingView === "mean_diff_prob") {
    return (
      <div className="help-content">
        <p><strong>Sampling Probability</strong> — the probability that this cell is selected for the next mutation under mean-diff sampling.</p>
        <p>Computed by taking each cell's fitness difference from the archive mean, shifting so the minimum is 0, applying temperature scaling (T={String(0.4)}), then normalising to sum to 1. Cells well above the mean dominate at low temperature.</p>
        <p className="help-note">Higher = more likely to be sampled. The distribution sharpens as temperature decreases toward 0.</p>
      </div>
    );
  }
  if (samplingView === "mean_diff_raw") {
    return (
      <div className="help-content">
        <p><strong>Mean Diff (fitness − mean)</strong> — each cell's raw fitness minus the current archive mean fitness.</p>
        <p>Positive values (above mean) are shown warm; negative values (below mean) are shown cool. This is the input to the temperature scaling step before normalisation into sampling probabilities.</p>
        <p className="help-note">Cells at 0 are exactly at the archive mean. After shifting and scaling, even negative-diff cells have a non-zero (small) probability of being sampled.</p>
      </div>
    );
  }
  if (samplingView === "primary_fitness") {
    return (
      <div className="help-content">
        <p><strong>Primary Fitness</strong> — the raw primary fitness score of the current elite in each cell.</p>
        <p>This is the value mean-diff sampling uses to compute relative weights. Cells with fitness well above the archive mean are sampled more often.</p>
      </div>
    );
  }
  if (samplingView === "cell_reward") {
    return (
      <div className="help-content">
        <p><strong>Cell Reward EMA</strong> — exponential moving average of the reward received each time this cell was sampled.</p>
        <p>Reward is +1 when a candidate is inserted into the archive and 0 otherwise, with a bonus proportional to the fitness improvement. Higher values indicate the cell has been productive recently.</p>
        <p className="help-note">Used by <em>{cellMethod}</em> cell sampling to compute the exploitation term of the UCB score.</p>
      </div>
    );
  }
  if (samplingView === "cell_n_trials") {
    return (
      <div className="help-content">
        <p><strong>Cell Trial Count</strong> — total number of times this cell has been sampled for mutation.</p>
        <p>Unvisited cells have a low count and receive a large UCB exploration bonus, making them attractive even before any reward is observed. As trials accumulate the bonus shrinks and reward dominates.</p>
        <p className="help-note">Used by <em>{cellMethod}</em> cell sampling to compute the exploration term of the UCB score.</p>
      </div>
    );
  }
  if (samplingView === "cell_logit_exploit") {
    return (
      <div className="help-content">
        <p><strong>Exploit Probability</strong> — sigmoid of the per-cell bandit logit, showing the probability that the algorithm will choose to exploit this cell (vs. explore).</p>
        <p>Values near 1.0 mean the run strongly prefers to refine the cell's current elite. Values near 0.5 indicate uncertainty. Values near 0 mean the run prefers exploration for this cell.</p>
        <p className="help-note">Driven by <em>{emitterMethod}</em>: the logit rises when candidates from this cell are accepted and falls when they are rejected.</p>
      </div>
    );
  }
  if (samplingView === "overall") {
    return (
      <div className="help-content">
        <p><strong>Overall Curiosity</strong> — combined curiosity score across all emitters, reflecting how interesting this cell is to the algorithm.</p>
        <p>Higher scores indicate cells the algorithm considers most worth sampling, balancing exploit and explore signals from all emitters.</p>
      </div>
    );
  }
  // Softmax per-emitter probability
  if (samplingView.startsWith("emitter_logit:")) {
    const emitterName = samplingView.slice("emitter_logit:".length);
    return (
      <div className="help-content">
        <p><strong>P({emitterName})</strong> — softmax probability that this cell's next mutation uses the <em>{emitterName}</em> emitter.</p>
        <p>Computed from all emitters' per-cell logits via softmax. The logit for <em>{emitterName}</em> rises when it produces candidates that are inserted, and stays flat otherwise.</p>
        <p className="help-note">Higher values mean this cell has found <em>{emitterName}</em> most productive so far.</p>
      </div>
    );
  }
  // Emitter-specific curiosity
  return (
    <div className="help-content">
      <p><strong>{samplingView} Curiosity</strong> — per-emitter curiosity score for the <em>{samplingView}</em> emitter.</p>
      <p>Shows how strongly the <em>{samplingView}</em> emitter wants to sample each cell. High values indicate cells this emitter has found rewarding or still finds uncertain.</p>
      <p className="help-note">Driven by <em>{emitterMethod}</em> emitter sampling.</p>
    </div>
  );
}

function NoRunSelected({ message }: { message: string }) {
  return (
    <section className="panel empty-state">
      <h2>No Run Selected</h2>
      <p>{message}</p>
    </section>
  );
}
