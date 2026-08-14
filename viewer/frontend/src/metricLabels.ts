export function metricLabel(label: string | null | undefined, fallback: string): string {
  const value = label?.trim();
  return value ? value : fallback;
}

export function hasSecondaryMetric(
  label: string | null | undefined,
  hasData?: boolean,
): boolean {
  const value = label?.trim().toLowerCase();
  if (!value || value === "disabled") return false;
  // If the caller knows whether data exists, require it
  if (hasData !== undefined) return hasData;
  return true;
}

export function bestMetricLabel(label: string | null | undefined, fallback: string): string {
  return `Best ${metricLabel(label, fallback)}`;
}

export function meanMetricLabel(label: string | null | undefined, fallback: string): string {
  return `Mean ${metricLabel(label, fallback)}`;
}
