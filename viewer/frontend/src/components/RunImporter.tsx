import { useState } from "react";

type RunImporterProps = {
  onImport: (path: string) => Promise<void>;
  onRefreshAll: () => Promise<void>;
};

export function RunImporter({ onImport, onRefreshAll }: RunImporterProps) {
  const [path, setPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleImport(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await onImport(path);
      setPath("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleRefreshAll() {
    setRefreshing(true);
    setError(null);
    try {
      await onRefreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <form className="panel importer" onSubmit={handleImport}>
      <div className="panel-header">
        <h2>Import Runs</h2>
        <p>Import one run folder or a parent directory containing many runs.</p>
      </div>
      <div className="import-row">
        <input
          value={path}
          onChange={(event) => setPath(event.target.value)}
          placeholder="/absolute/path/to/runs or /absolute/path/to/run"
        />
        <button type="submit" disabled={loading || !path.trim()}>
          {loading ? "Importing..." : "Import"}
        </button>
        <button type="button" className="ghost" onClick={handleRefreshAll} disabled={refreshing}>
          {refreshing ? "Refreshing..." : "Refresh Imported"}
        </button>
      </div>
      {error ? <div className="error-banner">{error}</div> : null}
    </form>
  );
}
