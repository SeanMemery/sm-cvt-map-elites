type SnapshotSelectorProps = {
  steps: number[];
  value: number;
  onChange: (step: number) => void;
};

export function SnapshotSelector({ steps, value, onChange }: SnapshotSelectorProps) {
  return (
    <label className="field">
      <span>Snapshot</span>
      <select value={value} onChange={(event) => onChange(Number(event.target.value))}>
        {steps.map((step) => (
          <option key={step} value={step}>
            step_{String(step).padStart(6, "0")}
          </option>
        ))}
      </select>
    </label>
  );
}
