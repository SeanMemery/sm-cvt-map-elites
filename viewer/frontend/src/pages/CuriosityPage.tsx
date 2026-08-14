import { useEffect, useMemo, useRef, useState } from "react";
import { Delaunay } from "d3-delaunay";

import { api } from "../api";
import { EChart } from "../components/EChart";
import { HelpBadge } from "../components/HelpBadge";
import { LoadingBar } from "../components/LoadingBar";
import { MetricSelector } from "../components/MetricSelector";
import type { EventRecord, RunSummaryResponse, SnapshotResponse } from "../types";

type CuriosityPageProps = {
  runId: string | null;
};

export function CuriosityPage({ runId }: CuriosityPageProps) {
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [snapshotSteps, setSnapshotSteps] = useState<number[]>([]);
  const [step, setStep] = useState<number | null>(null);
  const [stepToStage, setStepToStage] = useState<Map<number, number>>(new Map());
  const [snapshot, setSnapshot] = useState<SnapshotResponse | null>(null);
  const [xFeature, setXFeature] = useState("0");
  const [yFeature, setYFeature] = useState("1");
  const [curiosityMode, setCuriosityMode] = useState("overall");
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
        setCuriosityMode("overall");
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
  const emitterOptions = useMemo(() => {
    const emitters = Array.isArray(summary?.config?.emitters)
      ? summary?.config?.emitters
      : [];
    const names = emitters
      .map((entry) => (entry && typeof entry === "object" ? String((entry as { name?: unknown }).name ?? "") : ""))
      .filter((name) => name.length > 0);
    return [
      { value: "overall", label: "Overall" },
      ...names.map((name) => ({ value: name, label: name })),
    ];
  }, [summary]);
  const emitterCuriosityEnabled = emitterOptions.length > 1;
  const xIndex = Number.parseInt(xFeature, 10) || 0;
  const yIndex = Number.parseInt(yFeature, 10) || 0;
  const currentStepIndex = step == null ? -1 : snapshotSteps.indexOf(step);
  const chartOption = useMemo(() => {
    const centroids = snapshot?.centroids ?? [];
    const occupiedByCell = new Map((snapshot?.occupied_cells ?? []).map((cell) => [cell.cell_id, cell]));
    const occupiedValues = (snapshot?.occupied_cells ?? [])
      .map((cell) => getCuriosityValue(cell, curiosityMode))
      .filter((value): value is number => value != null);
    const minCuriosity = occupiedValues.length ? Math.min(...occupiedValues) : 0;
    const maxCuriosity = occupiedValues.length ? Math.max(...occupiedValues) : 1;
    const centroidPoints = centroids.map((coords, cellId) => ({
      cellId,
      x: Number(coords?.[xIndex] ?? 0),
      y: Number(coords?.[yIndex] ?? 0),
      occupied: occupiedByCell.get(cellId) ?? null,
    }));
    const xValues = centroidPoints.map((point) => point.x);
    const yValues = centroidPoints.map((point) => point.y);
    const [xMin, xMax] = expandRange(Math.min(...xValues, 0), Math.max(...xValues, 1));
    const [yMin, yMax] = expandRange(Math.min(...yValues, 0), Math.max(...yValues, 1));
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
      const curiosityValue = getCuriosityValue(point.occupied, curiosityMode);
      return {
        value: [point.x, point.y, Number(curiosityValue ?? minCuriosity)],
        cellId: point.cellId,
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
          return [
            `Cell ${data.cellId ?? "-"}`,
            data.candidateId ? `Candidate ${data.candidateId}` : "Unoccupied",
            data.candidateId ? `${curiosityMode === "overall" ? "Curiosity" : `${curiosityMode} Curiosity`} ${formatCuriosity(data.curiosity)}` : "Curiosity -",
            `Snapshot Step ${step ?? "-"}`,
          ].join("<br/>");
        },
      },
      grid: { left: 52, right: 20, top: 28, bottom: 78 },
      xAxis: {
        type: "value",
        min: xMin,
        max: xMax,
        splitNumber: 6,
        axisLabel: { color: "#d9d5c6" },
        axisLine: { lineStyle: { color: "#5f6b68" } },
        splitLine: { lineStyle: { color: "#2b3836" } },
      },
      yAxis: {
        type: "value",
        min: yMin,
        max: yMax,
        splitNumber: 6,
        axisLabel: { color: "#d9d5c6" },
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
  }, [curiosityMode, snapshot, step, xIndex, yIndex]);

  const stats = useMemo(() => {
    const values = (snapshot?.occupied_cells ?? [])
      .map((cell) => getCuriosityValue(cell, curiosityMode))
      .filter((value): value is number => value != null);
    const occupiedCount = snapshot?.occupied_cells.length ?? 0;
    const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    const max = values.length ? Math.max(...values) : null;
    return { occupiedCount, mean, max };
  }, [curiosityMode, snapshot]);

  if (!runId) {
    return <NoRunSelected message="Choose a run on Home, then use the left navigation to inspect curiosity." />;
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }

  if (!summary) {
    return (
      <>
        <LoadingBar active={loading} label="Loading curiosity view" />
        <div className="loading-panel">Loading curiosity view...</div>
      </>
    );
  }

  if (!summary.summary.curiosity_enabled) {
    return (
      <section className="panel empty-state">
        <h2>Curiosity Not Enabled</h2>
        <p>This run does not use curiosity-based sampling, so there is no curiosity map to inspect.</p>
      </section>
    );
  }

  if (!snapshotSteps.length) {
    return (
      <section className="panel empty-state">
        <h2>No Snapshots</h2>
        <p>This run has no archive snapshots, so curiosity cannot be scrubbed over time.</p>
      </section>
    );
  }

  return (
    <div className="page-grid">
      <LoadingBar active={loading || snapshotLoading} label="Loading curiosity map" />
      <section className="panel">
        <div className="panel-header">
          <div className="title-with-help">
            <h1>Curiosity</h1>
            <HelpBadge text="Shows archive cell curiosity over the Voronoi centroid grid, projected onto two selected feature dimensions. Use the step slider to scrub through archive snapshots over time." />
          </div>
        </div>
        <div className="controls-grid">
          <MetricSelector label="X Feature" value={xFeature} onChange={setXFeature} options={featureOptions} />
          <MetricSelector label="Y Feature" value={yFeature} onChange={setYFeature} options={featureOptions} />
          {emitterCuriosityEnabled ? (
            <MetricSelector label="Curiosity View" value={curiosityMode} onChange={setCuriosityMode} options={emitterOptions} />
          ) : null}
        </div>
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
            <h3>Curiosity Map</h3>
            <HelpBadge text="Shows the actual Voronoi cells induced by the archive centroids in the selected feature projection. Gray cells are unoccupied in this snapshot; colored cells are occupied and use either overall curiosity or the selected emitter-specific curiosity when that data exists for the run." />
          </div>
        </div>
        <EChart option={chartOption} />
        <div className="archive-map-stats curiosity-stats">
          <span><strong>Snapshot</strong> {step ?? "-"}</span>
          <span><strong>Occupied</strong> {stats.occupiedCount}</span>
          <span><strong>Mode</strong> {curiosityMode === "overall" ? "Overall" : curiosityMode}</span>
          <span><strong>Mean Curiosity</strong> {formatCuriosity(stats.mean)}</span>
          <span><strong>Max Curiosity</strong> {formatCuriosity(stats.max)}</span>
        </div>
      </section>
    </div>
  );
}

function getCuriosityValue(
  cell: SnapshotResponse["occupied_cells"][number] | null | undefined,
  curiosityMode: string,
) {
  if (!cell) {
    return null;
  }
  if (curiosityMode === "overall") {
    return typeof cell.curiosity_score === "number" ? cell.curiosity_score : null;
  }
  const emitterScores = cell.emitter_curiosity_scores;
  if (!emitterScores || typeof emitterScores !== "object") {
    return null;
  }
  const value = emitterScores[curiosityMode];
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

function NoRunSelected({ message }: { message: string }) {
  return (
    <section className="panel empty-state">
      <h2>No Run Selected</h2>
      <p>{message}</p>
    </section>
  );
}
