import { useState } from "react";
import { Link } from "react-router-dom";

import { hasSecondaryMetric, metricLabel } from "../metricLabels";
import type { CandidateRecord } from "../types";

type SortKey = string; // "id" | "created_at_step" | "cell_id" | "primary_fitness" | "secondary_fitness" | "parent" | `desc_${number}`
type SortDir = "asc" | "desc";

type CandidateTableProps = {
  candidates: CandidateRecord[];
  primaryMetricLabel: string;
  secondaryMetricLabel: string;
  descriptorLabels?: string[];
  /** Map from island run_id to that island's descriptor_labels. When present,
   *  each candidate uses its own island's labels via island_run_id. */
  descriptorMap?: Record<string, string[]>;
};

function SortTh({ col, label, sortKey, sortDir, onSort }: {
  col: SortKey; label: string; sortKey: SortKey; sortDir: SortDir; onSort: (k: SortKey) => void;
}) {
  const active = sortKey === col;
  const arrow = active ? (sortDir === "asc" ? " ↑" : " ↓") : "";
  return (
    <th
      className={`sortable-th${active ? " sort-active" : ""}`}
      onClick={() => onSort(col)}
    >
      {label}{arrow}
    </th>
  );
}

export function CandidateTable({ candidates, primaryMetricLabel, secondaryMetricLabel, descriptorLabels, descriptorMap }: CandidateTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("primary_fitness");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const showSecondary = hasSecondaryMetric(secondaryMetricLabel)
    && candidates.some(c => c.secondary_fitness != null);

  // Collect the union of all descriptor keys across candidates (from descriptor_values dict).
  // Fall back to descriptorLabels if no candidate has descriptor_values yet.
  const allDescLabels: string[] = (() => {
    const keys = new Set<string>();
    for (const c of candidates) {
      for (const k of Object.keys(c.descriptor_values ?? {})) keys.add(k);
    }
    if (keys.size > 0) return [...keys].sort();
    return descriptorLabels ?? [];
  })();

  function descValueFor(candidate: CandidateRecord, label: string): string {
    const dv = candidate.descriptor_values;
    if (dv && label in dv) return formatDesc(dv[label]);
    return "—";
  }

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function sortValue(c: CandidateRecord): string | number {
    if (sortKey.startsWith("desc_")) {
      const label = allDescLabels[parseInt(sortKey.slice(5), 10)];
      return c.descriptor_values?.[label] ?? -Infinity;
    }
    switch (sortKey) {
      case "id": return c.id;
      case "created_at_step": return c.created_at_step ?? -Infinity;
      case "cell_id": return c.cell_id ?? -1;
      case "primary_fitness": return c.primary_fitness ?? -Infinity;
      case "secondary_fitness": return c.secondary_fitness ?? -Infinity;
      case "parent": return (c.parent_ids ?? [])[0] ?? "";
      default: return 0;
    }
  }

  const sorted = [...candidates].sort((a, b) => {
    const av = sortValue(a);
    const bv = sortValue(b);
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortDir === "asc" ? cmp : -cmp;
  });

  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            <SortTh col="id" label="ID" sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
            <SortTh col="created_at_step" label="Step" sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
            <SortTh col="cell_id" label="Cell" sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
            <SortTh col="primary_fitness" label={metricLabel(primaryMetricLabel, "Primary")} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
            {showSecondary ? <SortTh col="secondary_fitness" label={metricLabel(secondaryMetricLabel, "Secondary")} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} /> : null}
            {allDescLabels.map((label, i) => (
              <SortTh key={`desc_${i}`} col={`desc_${i}`} label={label} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
            ))}
            <SortTh col="parent" label="Parent" sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
          </tr>
        </thead>
        <tbody>
          {sorted.map((candidate) => {
            const islandRunId = (candidate as Record<string, unknown>).island_run_id as string | undefined;
            const detailPath = islandRunId
              ? `/candidates/${candidate.id}?run=${islandRunId}`
              : `/candidates/${candidate.id}`;
            return (
            <tr key={`${islandRunId ?? ""}:${candidate.id}`}>
              <td><Link to={detailPath}>{candidate.id}</Link></td>
              <td>{candidate.created_at_step}</td>
              <td>{candidate.cell_id ?? "-"}</td>
              <td>{format(candidate.primary_fitness)}</td>
              {showSecondary ? <td>{format(candidate.secondary_fitness)}</td> : null}
              {allDescLabels.map((label) => (
                <td key={label}>{descValueFor(candidate, label)}</td>
              ))}
              <td>{(candidate.parent_ids ?? [])[0] ?? "-"}</td>
            </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function format(value: number | null | undefined) {
  if (value === null || value === undefined) return "-";
  return value.toFixed(4);
}

function formatDesc(value: number | null | undefined) {
  if (value === null || value === undefined) return "-";
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}
