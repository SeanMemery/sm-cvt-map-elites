import type { CandidateRecord } from "../types";
import { hasSecondaryMetric, metricLabel } from "../metricLabels";

type CandidatePanelProps = {
  candidate?: CandidateRecord | null;
  descriptorLabels: string[];
  primaryMetricLabel: string;
  primaryValidationMetricLabel: string;
  secondaryMetricLabel: string;
  secondaryValidationMetricLabel: string;
  onOpenFull?: () => void;
};

export function CandidatePanel({
  candidate,
  descriptorLabels,
  primaryMetricLabel,
  primaryValidationMetricLabel,
  secondaryMetricLabel,
  secondaryValidationMetricLabel,
  onOpenFull,
}: CandidatePanelProps) {
  const showSecondary = hasSecondaryMetric(secondaryMetricLabel);
  const showPrimaryValidation = candidate?.primary_validation_fitness != null;
  const showSecondaryValidation = candidate?.secondary_validation_fitness != null;
  if (!candidate) {
    return (
      <div className="panel sticky-panel">
        <div className="panel-header">
          <h3>Candidate</h3>
          <p>Select a cell or candidate to inspect it.</p>
        </div>
      </div>
    );
  }

  const rows: Array<[string, string]> = [
    ["Cell", formatValue(candidate.cell_id)] as [string, string],
    ["Generation", String(candidate.generation)] as [string, string],
    ["Step", String(candidate.created_at_step)] as [string, string],
    ...(candidate.stage != null && candidate.stage > 0 ? [["Stage", String(candidate.stage)] as [string, string]] : []),
    [metricLabel(primaryMetricLabel, "Primary"), formatNumber(candidate.primary_fitness)] as [string, string],
    ...(showPrimaryValidation
      ? [[metricLabel(primaryValidationMetricLabel, "Primary Validation"), formatNumber(candidate.primary_validation_fitness)] as [string, string]]
      : []),
    ...(showSecondary ? [[metricLabel(secondaryMetricLabel, "Secondary"), formatNumber(candidate.secondary_fitness)] as [string, string]] : []),
    ...(showSecondaryValidation
      ? [[metricLabel(secondaryValidationMetricLabel, "Secondary Validation"), formatNumber(candidate.secondary_validation_fitness)] as [string, string]]
      : []),
    ...descriptorLabels.map((label, index) => [label, formatNumber(candidate.descriptor_raw?.[index])] as [string, string]),
  ];

  const isInit = candidate.parent_ids.length === 0;

  return (
    <div className="panel sticky-panel candidate-side-panel">
      <div className="panel-header">
        <div>
          <h3>{candidate.id}</h3>
          <p>{isInit ? <span className="stage-badge" style={{ fontSize: "0.7rem", padding: "2px 8px" }}>Random Init</span> : "Selected candidate"}</p>
        </div>
        {onOpenFull ? (
          <button type="button" className="ghost candidate-panel-open-button" onClick={onOpenFull}>
            View Full Candidate
          </button>
        ) : null}
      </div>
      <div className="candidate-panel-table">
        {rows.map(([label, value]) => (
          <div key={label} className="candidate-panel-row">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <details className="candidate-panel-code">
        <summary>Code</summary>
        <pre className="code-preview">{candidate.code}</pre>
      </details>
    </div>
  );
}

function formatNumber(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return value.toFixed(4);
}

function formatValue(value: number | string | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return String(value);
}
