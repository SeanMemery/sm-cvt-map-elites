import { useMemo } from "react";

import type { CandidateRecord, SnapshotResponse } from "../types";

type PointRecord = {
  id: string;
  x: number;
  y: number;
  primary?: number | null;
  primaryValidation?: number | null;
  secondary?: number | null;
  secondaryValidation?: number | null;
  size: number;
  raw?: number[] | null;
  norm?: number[] | null;
  candidate?: CandidateRecord;
};

type ArchiveMapProps = {
  snapshot: SnapshotResponse;
  candidates: CandidateRecord[];
  xMetric: string;
  yMetric: string;
  colorMetric: string;
  sizeMetric: string;
  fixedSize: number;
  primaryMetricLabel: string;
  primaryValidationMetricLabel: string;
  secondaryMetricLabel: string;
  secondaryValidationMetricLabel: string;
  onCandidateClick: (candidateId: string) => void;
};

const WIDTH = 920;
const HEIGHT = 540;
const PADDING = 36;

export function ArchiveMap({
  snapshot,
  candidates,
  xMetric,
  yMetric,
  colorMetric,
  sizeMetric,
  fixedSize,
  primaryMetricLabel,
  primaryValidationMetricLabel,
  secondaryMetricLabel,
  secondaryValidationMetricLabel,
  onCandidateClick,
}: ArchiveMapProps) {
  const points = useMemo(() => {
    const activeById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
    const snapshotPoints: PointRecord[] = snapshot.occupied_cells.map((cell) => ({
      id: cell.candidate_id,
      x: readMetricValue(cell, xMetric),
      y: readMetricValue(cell, yMetric),
      primary: cell.primary_fitness,
      primaryValidation: cell.primary_validation_fitness,
      secondary: cell.secondary_fitness,
      secondaryValidation: cell.secondary_validation_fitness,
      size: 8,
      raw: cell.descriptor_raw,
      norm: cell.descriptor_norm,
      candidate: activeById.get(cell.candidate_id),
    }));
    return snapshotPoints;
  }, [candidates, snapshot, xMetric, yMetric]);

  const projected = useMemo(() => projectPoints(points), [points]);
  const axes = useMemo(() => buildAxes(points), [points]);
  const colorScale = useMemo(
    () =>
      buildColorScale(points, colorMetric, {
        primaryMetricLabel,
        primaryValidationMetricLabel,
        secondaryMetricLabel,
        secondaryValidationMetricLabel,
      }),
    [colorMetric, points, primaryMetricLabel, primaryValidationMetricLabel, secondaryMetricLabel, secondaryValidationMetricLabel],
  );
  const sizeScale = useMemo(() => buildSizeScale(points, sizeMetric), [points, sizeMetric]);
  const plotStats = useMemo(() => buildPlotStats(points), [points]);

  return (
    <div className="archive-map-shell">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="archive-map">
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} rx={24} fill="#0e1716" />
        {axes.xTicks.map((tick) => (
          <line
            key={`x-grid-${tick.value}`}
            x1={tick.position}
            x2={tick.position}
            y1={PADDING}
            y2={HEIGHT - PADDING}
            stroke="#233433"
            strokeWidth={1}
            strokeDasharray="4 6"
          />
        ))}
        {axes.yTicks.map((tick) => (
          <line
            key={`y-grid-${tick.value}`}
            x1={PADDING}
            x2={WIDTH - PADDING}
            y1={tick.position}
            y2={tick.position}
            stroke="#233433"
            strokeWidth={1}
            strokeDasharray="4 6"
          />
        ))}
        <line x1={PADDING} x2={WIDTH - PADDING} y1={HEIGHT - PADDING} y2={HEIGHT - PADDING} stroke="#5f6b68" strokeWidth={1.5} />
        <line x1={PADDING} x2={PADDING} y1={PADDING} y2={HEIGHT - PADDING} stroke="#5f6b68" strokeWidth={1.5} />
        {projected.map((point) => (
          <g key={point.id} onClick={() => onCandidateClick(point.id)} className="archive-point">
            <circle
              cx={point.sx}
              cy={point.sy}
              r={resolveSize(point, sizeMetric, sizeScale, fixedSize)}
              fill={resolveColor(point, colorMetric, colorScale)}
              stroke="#f4f4ef"
              strokeOpacity={0.6}
            />
          </g>
        ))}
        {axes.xTicks.map((tick) => (
          <g key={`x-label-${tick.value}`}>
            <line
              x1={tick.position}
              x2={tick.position}
              y1={HEIGHT - PADDING}
              y2={HEIGHT - PADDING + 8}
              stroke="#7f8b88"
              strokeWidth={1}
            />
            <text x={tick.position} y={HEIGHT - 10} textAnchor="middle" className="archive-axis-tick">
              {formatTick(tick.value)}
            </text>
          </g>
        ))}
        {axes.yTicks.map((tick) => (
          <g key={`y-label-${tick.value}`}>
            <line
              x1={PADDING - 8}
              x2={PADDING}
              y1={tick.position}
              y2={tick.position}
              stroke="#7f8b88"
              strokeWidth={1}
            />
            <text x={PADDING - 12} y={tick.position + 4} textAnchor="end" className="archive-axis-tick">
              {formatTick(tick.value)}
            </text>
          </g>
        ))}
      </svg>
      <div className="archive-map-footer">
        <div className="archive-color-legend">
          <div className="archive-color-legend-header">
            <span>Color</span>
            <strong>{colorScale.label}</strong>
          </div>
          <div className="archive-color-legend-bar" style={{ background: colorScale.legendGradient }} />
          <div className="archive-color-legend-scale">
            <span>{colorScale.startLabel}</span>
            <span>{colorScale.endLabel}</span>
          </div>
        </div>
        <div className="archive-plot-stats">
          <div className="archive-plot-stat">
            <span>Points</span>
            <strong>{plotStats.pointCount}</strong>
          </div>
          <div className="archive-plot-stat">
            <span>Correlation</span>
            <strong>{formatLegendValue(plotStats.correlation)}</strong>
          </div>
          <div className="archive-plot-stat">
            <span>Corr. p-value</span>
            <strong>{formatPValue(plotStats.correlationPValue)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

type Tick = {
  value: number;
  position: number;
};

function readMetricValue(
  cell: {
    primary_fitness?: number | null;
    primary_validation_fitness?: number | null;
    secondary_fitness?: number | null;
    secondary_validation_fitness?: number | null;
    descriptor_raw?: number[] | null;
    descriptor_norm?: number[] | null;
  },
  metric: string,
) {
  if (metric === "primary_fitness") {
    return cell.primary_fitness ?? 0;
  }
  if (metric === "primary_validation_fitness") {
    return cell.primary_validation_fitness ?? 0;
  }
  if (metric === "secondary_fitness") {
    return cell.secondary_fitness ?? 0;
  }
  if (metric === "secondary_validation_fitness") {
    return cell.secondary_validation_fitness ?? 0;
  }
  return readMetric(cell.descriptor_raw, cell.descriptor_norm, metric);
}

function readMetric(raw: number[] | null | undefined, norm: number[] | null | undefined, metric: string) {
  if (metric.startsWith("raw:")) {
    return raw?.[extractDimension(metric)] ?? 0;
  }
  if (metric.startsWith("norm:")) {
    return norm?.[extractDimension(metric)] ?? 0;
  }
  return 0;
}

function extractDimension(metric: string) {
  return Number(metric.split(":")[1] ?? "0");
}

function projectPoints(points: PointRecord[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1);
  return points.map((point) => ({
    ...point,
    sx: scale(point.x, minX, maxX, PADDING, WIDTH - PADDING),
    sy: scale(point.y, minY, maxY, HEIGHT - PADDING, PADDING),
  }));
}

function buildAxes(points: PointRecord[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1);
  return {
    xTicks: buildTicks(minX, maxX, (value) => scale(value, minX, maxX, PADDING, WIDTH - PADDING)),
    yTicks: buildTicks(minY, maxY, (value) => scale(value, minY, maxY, HEIGHT - PADDING, PADDING)),
  };
}

function buildTicks(min: number, max: number, project: (value: number) => number): Tick[] {
  const tickCount = 5;
  if (min === max) {
    return [{ value: min, position: project(min) }];
  }
  const span = niceNumber(max - min, false);
  const step = niceNumber(span / (tickCount - 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks: Tick[] = [];
  for (let value = niceMin; value <= niceMax + step * 0.5; value += step) {
    ticks.push({
      value,
      position: project(value),
    });
  }
  return ticks;
}

function scale(value: number, min: number, max: number, outMin: number, outMax: number) {
  if (max === min) {
    return (outMin + outMax) / 2;
  }
  const ratio = (value - min) / (max - min);
  return outMin + ratio * (outMax - outMin);
}

type ColorScale = {
  label: string;
  min: number;
  max: number;
  startLabel: string;
  endLabel: string;
  legendGradient: string;
  colors: string[];
};

type SizeScale = {
  min: number;
  max: number;
};

type PlotStats = {
  pointCount: number;
  correlation: number;
  correlationPValue: number | null;
};

function resolveColor(point: PointRecord, metric: string, colorScale: ColorScale) {
  const value = readColorValue(point, metric);
  return interpolatePalette(colorScale.colors, normalizeValue(value, colorScale.min, colorScale.max));
}

function readColorValue(point: PointRecord, metric: string) {
  if (metric === "solid") {
    return 1;
  }
  if (metric === "primary_validation_fitness") {
    return point.primaryValidation ?? 0;
  }
  if (metric === "secondary_fitness") {
    return point.secondary ?? 0;
  }
  if (metric === "secondary_validation_fitness") {
    return point.secondaryValidation ?? 0;
  }
  if (metric === "age") {
    return point.candidate?.created_at_step ?? 0;
  }
  return point.primary ?? 0;
}

function buildColorScale(
  points: PointRecord[],
  metric: string,
  labels: {
    primaryMetricLabel: string;
    primaryValidationMetricLabel: string;
    secondaryMetricLabel: string;
    secondaryValidationMetricLabel: string;
  },
): ColorScale {
  const values = points
    .map((point) => readColorValue(point, metric))
    .filter((value) => Number.isFinite(value));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const colors =
    metric === "solid"
      ? ["#6ba8ff", "#6ba8ff"]
      : metric === "age"
      ? ["#1d4ed8", "#06b6d4", "#f59e0b", "#dc2626"]
      : metric === "primary_validation_fitness"
        ? ["#7f1d1d", "#dc2626", "#fb7185", "#fecaca"]
      : metric === "secondary_fitness"
        ? ["#14532d", "#22c55e", "#a3e635", "#fef08a"]
        : metric === "secondary_validation_fitness"
          ? ["#134e4a", "#0f766e", "#14b8a6", "#99f6e4"]
        : ["#1e3a8a", "#2563eb", "#14b8a6", "#f59e0b", "#fef08a"];
  const metricLabel =
    metric === "solid"
      ? "Solid"
      : metric === "age"
        ? "Candidate Age"
        : metric === "primary_validation_fitness"
          ? labels.primaryValidationMetricLabel
        : metric === "secondary_fitness"
          ? labels.secondaryMetricLabel
          : metric === "secondary_validation_fitness"
            ? labels.secondaryValidationMetricLabel
            : labels.primaryMetricLabel;
  return {
    label: metricLabel,
    min,
    max,
    startLabel:
      metric === "solid" ? "Single color" : metric === "age" ? `Older (${formatLegendValue(min)})` : formatLegendValue(min),
    endLabel:
      metric === "solid" ? "Single color" : metric === "age" ? `Newer (${formatLegendValue(max)})` : formatLegendValue(max),
    legendGradient: `linear-gradient(90deg, ${colors.join(", ")})`,
    colors,
  };
}

function readSizeValue(point: PointRecord, metric: string) {
  if (metric === "primary_fitness") {
    return point.primary ?? 0;
  }
  if (metric === "primary_validation_fitness") {
    return point.primaryValidation ?? 0;
  }
  if (metric === "secondary_fitness") {
    return point.secondary ?? 0;
  }
  if (metric === "secondary_validation_fitness") {
    return point.secondaryValidation ?? 0;
  }
  if (metric === "age") {
    return point.candidate?.created_at_step ?? 0;
  }
  if (metric.startsWith("raw:")) {
    return point.raw?.[extractDimension(metric)] ?? 0;
  }
  return 0;
}

function buildSizeScale(points: PointRecord[], metric: string): SizeScale {
  if (metric === "none") {
    return { min: 0, max: 1 };
  }
  const values = points
    .map((point) => readSizeValue(point, metric))
    .filter((value) => Number.isFinite(value));
  return {
    min: values.length ? Math.min(...values) : 0,
    max: values.length ? Math.max(...values) : 1,
  };
}

function resolveSize(point: PointRecord, metric: string, sizeScale: SizeScale, fixedSize: number) {
  if (metric === "none") {
    return fixedSize;
  }
  const value = readSizeValue(point, metric);
  const normalized = normalizeValue(value, sizeScale.min, sizeScale.max);
  return 5 + normalized * 11;
}

function buildPlotStats(points: PointRecord[]): PlotStats {
  const xs = points.map((point) => point.x).filter((value) => Number.isFinite(value));
  const ys = points.map((point) => point.y).filter((value) => Number.isFinite(value));
  return {
    pointCount: points.length,
    correlation: pearsonCorrelation(xs, ys),
    correlationPValue: pearsonCorrelationPValue(xs, ys),
  };
}

function formatTick(value: number) {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

function formatPValue(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  if (value < 0.001) {
    return "< 0.001";
  }
  if (value < 0.01) {
    return value.toFixed(3);
  }
  return value.toFixed(2);
}

function formatLegendValue(value: number) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}

function normalizeValue(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (max === min) {
    return 0.5;
  }
  return Math.min(1, Math.max(0, (value - min) / (max - min)));
}

function interpolatePalette(colors: string[], t: number) {
  if (!colors.length) {
    return "#6ba8ff";
  }
  if (colors.length === 1) {
    return colors[0];
  }
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (colors.length - 1);
  const lowerIndex = Math.floor(scaled);
  const upperIndex = Math.min(colors.length - 1, lowerIndex + 1);
  const localT = scaled - lowerIndex;
  return mixHex(colors[lowerIndex], colors[upperIndex], localT);
}

function mixHex(left: string, right: string, t: number) {
  const a = hexToRgb(left);
  const b = hexToRgb(right);
  const mixed = {
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t),
  };
  return `rgb(${mixed.r}, ${mixed.g}, ${mixed.b})`;
}

function hexToRgb(hex: string) {
  const normalized = hex.replace("#", "");
  const value = normalized.length === 3
    ? normalized.split("").map((part) => `${part}${part}`).join("")
    : normalized;
  return {
    r: Number.parseInt(value.slice(0, 2), 16),
    g: Number.parseInt(value.slice(2, 4), 16),
    b: Number.parseInt(value.slice(4, 6), 16),
  };
}

function pearsonCorrelation(xs: number[], ys: number[]) {
  const count = Math.min(xs.length, ys.length);
  if (count < 2) {
    return 0;
  }
  const left = xs.slice(0, count);
  const right = ys.slice(0, count);
  const meanX = left.reduce((sum, value) => sum + value, 0) / count;
  const meanY = right.reduce((sum, value) => sum + value, 0) / count;
  let numerator = 0;
  let denomX = 0;
  let denomY = 0;
  for (let index = 0; index < count; index += 1) {
    const dx = left[index] - meanX;
    const dy = right[index] - meanY;
    numerator += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }
  if (denomX <= 0 || denomY <= 0) {
    return 0;
  }
  return numerator / Math.sqrt(denomX * denomY);
}

function pearsonCorrelationPValue(xs: number[], ys: number[]) {
  const count = Math.min(xs.length, ys.length);
  if (count < 4) {
    return null;
  }
  const correlation = pearsonCorrelation(xs, ys);
  const clamped = Math.max(-0.999999, Math.min(0.999999, correlation));
  const fisherZ = 0.5 * Math.log((1 + clamped) / (1 - clamped));
  const zScore = Math.abs(fisherZ) * Math.sqrt(count - 3);
  return Math.max(0, Math.min(1, erfc(zScore / Math.SQRT2)));
}

function erfc(value: number) {
  return 1 - erf(value);
}

function erf(value: number) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function niceNumber(value: number, round: boolean) {
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const fraction = Math.abs(value) / 10 ** exponent;
  let niceFraction = 1;
  if (round) {
    if (fraction < 1.5) {
      niceFraction = 1;
    } else if (fraction < 3) {
      niceFraction = 2;
    } else if (fraction < 7) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
  } else if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }
  return Math.sign(value) * niceFraction * 10 ** exponent;
}
