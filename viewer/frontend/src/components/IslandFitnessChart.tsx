import { EChart } from "../components/EChart";

type IslandFitnessChartProps = {
  islandStats: Record<string, Array<{ step: number; mean_primary_fitness?: number | null; best_primary_fitness?: number | null }>>;
  islandIds: string[];
};

const ISLAND_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

function islandLabel(id: string, idx: number): string {
  const match = id.match(/island[_\s]?(\d+)/i);
  return match ? `Island ${match[1]}` : `Island ${idx + 1}`;
}

function islandColor(id: string): string {
  const match = id.match(/island[_\s]?(\d+)/i);
  const idx = match ? parseInt(match[1], 10) : 0;
  return ISLAND_COLORS[idx % ISLAND_COLORS.length];
}

export function IslandFitnessChart({ islandStats, islandIds }: IslandFitnessChartProps) {
  if (islandIds.length < 2) return null;

  // Collect all steps across all islands
  const allSteps = new Set<number>();
  for (const id of islandIds) {
    const entries = islandStats[id] ?? [];
    for (const e of entries) {
      if (e.step != null) allSteps.add(e.step);
    }
  }
  if (allSteps.size === 0) return null;

  const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

  // Build per-island mean fitness series
  const islandSeries = islandIds.map((id, idx) => {
    const color = islandColor(id);
    const entries = (islandStats[id] ?? []).filter((e) => e.step != null && e.mean_primary_fitness != null);
    entries.sort((a, b) => a.step - b.step);
    const data = entries.map((e) => [e.step, e.mean_primary_fitness] as [number, number]);
    return {
      name: `${islandLabel(id, idx)} (mean)`,
      type: "line" as const,
      data,
      smooth: true,
      symbolSize: 4,
      lineStyle: { color, width: 2 },
      itemStyle: { color },
    };
  });

  // Build "best across all islands" series — forward-fill each island's last known best
  // so a stuck/slow island doesn't cause the line to drop at later steps.
  const bestByStep = new Map<number, number>();
  for (const id of islandIds) {
    const entries = (islandStats[id] ?? []).filter((e) => e.step != null && e.best_primary_fitness != null);
    entries.sort((a, b) => a.step - b.step);
    let lastBest: number | null = null;
    let entryIdx = 0;
    for (const step of sortedSteps) {
      // Advance to entries up to this step
      while (entryIdx < entries.length && entries[entryIdx].step <= step) {
        const v = entries[entryIdx].best_primary_fitness;
        if (v != null) lastBest = lastBest == null ? v : Math.max(lastBest, v);
        entryIdx++;
      }
      if (lastBest != null) {
        const current = bestByStep.get(step);
        if (current == null || lastBest > current) {
          bestByStep.set(step, lastBest);
        }
      }
    }
  }
  const bestData = sortedSteps
    .filter((step) => bestByStep.has(step))
    .map((step) => [step, bestByStep.get(step)!] as [number, number]);

  const bestSeries = {
    name: "Best (all islands)",
    type: "line" as const,
    data: bestData,
    smooth: true,
    symbolSize: 5,
    lineStyle: { color: "#f8fafc", width: 3 },
    itemStyle: { color: "#f8fafc" },
  };

  const option = {
    animation: false,
    tooltip: { trigger: "axis" },
    legend: { top: 0, textStyle: { color: "#f4f4ef" } },
    grid: { left: 52, right: 20, top: 72, bottom: 40 },
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
    series: [...islandSeries, bestSeries],
  };

  return <EChart option={option} />;
}
