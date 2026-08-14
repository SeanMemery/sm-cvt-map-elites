import type { EventRecord, StageMarker } from "../types";
import { bestMetricLabel, hasSecondaryMetric, meanMetricLabel, metricLabel } from "../metricLabels";
import { EChart } from "./EChart";

type FitnessChartProps = {
  stats: Record<string, unknown>[];
  secondaryMarkers: EventRecord[];
  primaryMetricLabel: string;
  primaryValidationMetricLabel: string;
  secondaryMetricLabel: string;
  secondaryValidationMetricLabel: string;
  stageMarkers?: StageMarker[];
};

export function FitnessChart({
  stats,
  secondaryMarkers: _secondaryMarkers,
  primaryMetricLabel,
  primaryValidationMetricLabel,
  secondaryMetricLabel,
  secondaryValidationMetricLabel,
  stageMarkers,
}: FitnessChartProps) {
  const bestPrimaryColor = "#d97706";
  const meanPrimaryColor = "#2563eb";
  const bestPrimaryValidationColor = "#dc2626";
  const bestSecondaryColor = "#16a34a";
  const bestSecondaryValidationColor = "#0f766e";
  const chartStats = stats.filter((entry) => Number(entry.step ?? -1) >= 0);
  const showSecondary = hasSecondaryMetric(secondaryMetricLabel) && chartStats.some((entry) => entry.best_secondary_fitness != null);
  const showPrimaryValidation = chartStats.some((entry) => entry.best_primary_validation_fitness != null);
  const showSecondaryValidation =
    hasSecondaryMetric(secondaryMetricLabel) && chartStats.some((entry) => entry.best_secondary_validation_fitness != null);

  return (
    <EChart
      option={{
        animation: false,
        tooltip: { trigger: "axis" },
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
            name: bestMetricLabel(primaryMetricLabel, "Primary"),
            type: "line",
            data: [...chartStats.map((entry) => [Number(entry.step ?? 0), entry.best_primary_fitness])],
            ...(stageMarkers && stageMarkers.length > 0
              ? {
                  markLine: {
                    silent: true,
                    symbol: "none",
                    lineStyle: { color: "#6366f1", type: "dashed", width: 1, opacity: 0.55 },
                    label: { show: true, formatter: (p: { value?: number }) => `S${stageMarkers.find((m) => m.step === p.value)?.stage ?? ""}`, color: "#a5b4fc", fontSize: 10, position: "insideEndTop" },
                    data: stageMarkers.map((m) => ({ xAxis: m.step })),
                  },
                }
              : {}),
            smooth: true,
            symbolSize: 4,
            lineStyle: { color: bestPrimaryColor, width: 3 },
            itemStyle: { color: bestPrimaryColor },
          },
          {
            name: meanMetricLabel(primaryMetricLabel, "Primary"),
            type: "line",
            data: [...chartStats.map((entry) => [Number(entry.step ?? 0), entry.mean_primary_fitness])],
            smooth: true,
            symbolSize: 4,
            lineStyle: { color: meanPrimaryColor, width: 2 },
            itemStyle: { color: meanPrimaryColor },
          },
          ...(showPrimaryValidation
            ? [
                {
                  name: bestMetricLabel(primaryValidationMetricLabel, "Primary Validation"),
                  type: "line",
                  data: [...chartStats.map((entry) => [Number(entry.step ?? 0), entry.best_primary_validation_fitness])],
                  smooth: true,
                  symbolSize: 4,
                  lineStyle: { color: bestPrimaryValidationColor, width: 2, type: "dashed" },
                  itemStyle: { color: bestPrimaryValidationColor },
                },
              ]
            : []),
          ...(showSecondary
            ? [
                {
                  name: bestMetricLabel(secondaryMetricLabel, "Secondary"),
                  type: "line",
                  data: [...chartStats.map((entry) => [Number(entry.step ?? 0), entry.best_secondary_fitness])],
                  smooth: true,
                  symbolSize: 4,
                  lineStyle: { color: bestSecondaryColor, width: 2 },
                  itemStyle: { color: bestSecondaryColor },
                },
              ]
            : []),
          ...(showSecondaryValidation
            ? [
                {
                  name: bestMetricLabel(secondaryValidationMetricLabel, "Secondary Validation"),
                  type: "line",
                  data: [...chartStats.map((entry) => [Number(entry.step ?? 0), entry.best_secondary_validation_fitness])],
                  smooth: true,
                  symbolSize: 4,
                  lineStyle: { color: bestSecondaryValidationColor, width: 2, type: "dashed" },
                  itemStyle: { color: bestSecondaryValidationColor },
                },
              ]
            : []),
        ],
      }}
    />
  );
}
