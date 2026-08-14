import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, NavLink, Route, Routes, useLocation, useParams } from "react-router-dom";
import { Activity, Boxes, CandlestickChart, FolderOpen, Sparkles } from "lucide-react";

import { api } from "./api";
import { ArchivePage } from "./pages/ArchivePage";
import { CandidateBrowserPage } from "./pages/CandidateBrowserPage";
import { CandidatePage } from "./pages/CandidatePage";
import { HomePage } from "./pages/HomePage";
import { SamplingPage } from "./pages/SamplingPage";
import { RunOverviewPage } from "./pages/RunOverviewPage";
import type { RunSummary } from "./types";
const SELECTED_RUN_STORAGE_KEY = "viewer:selectedRunId";

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [allRuns, setAllRuns] = useState<RunSummary[]>([]); // full list including island runs
  const [error, setError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunIdState] = useState<string | null>(() => window.localStorage.getItem(SELECTED_RUN_STORAGE_KEY));
  const [islandMode, setIslandMode] = useState<string>("all");
  const newestRun = useMemo(() => {
    const pool = allRuns.length ? allRuns : runs;
    if (!pool.length) return null;
    return [...pool].sort(compareRunsDescending)[0] ?? null;
  }, [runs, allRuns]);

  const refreshRuns = useCallback(async () => {
    try {
      const [jobs, all] = await Promise.all([api.listJobs(), api.listRuns()]);
      setRuns(jobs);
      setAllRuns(all);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  async function handleImport(path: string) {
    await api.importRuns(path);
    await refreshRuns();
  }

  async function handleRefresh(runId: string) {
    await api.refreshRun(runId);
    await refreshRuns();
  }

  async function handleRefreshAll() {
    await api.refreshAllRuns();
    await refreshRuns();
  }

  async function handleRemove(runId: string) {
    await api.removeRun(runId);
    await refreshRuns();
  }

  async function handleCancel(runId: string) {
    await api.cancelRun(runId);
    await handleRefresh(runId);
  }

  async function handlePause(runId: string) {
    await api.pauseRun(runId);
    await handleRefresh(runId);
  }

  async function handleResume(runId: string) {
    await api.resumeRun(runId);
    await handleRefresh(runId);
  }

  const location = useLocation();
  const routeRunId = location.pathname.match(/^\/runs\/([^/]+)/)?.[1];
  const activeRun = useMemo(
    () => (allRuns.length ? allRuns : runs).find((run) => run.run_id === selectedRunId) ?? null,
    [runs, allRuns, selectedRunId],
  );
  const setSelectedRunId = useCallback((runId: string | null) => {
    setSelectedRunIdState(runId);
    if (runId) {
      window.localStorage.setItem(SELECTED_RUN_STORAGE_KEY, runId);
      return;
    }
    window.localStorage.removeItem(SELECTED_RUN_STORAGE_KEY);
  }, []);

  const prevJobIdRef = useRef<string | null>(null);
  useEffect(() => {
    const allKnown = allRuns.length ? allRuns : runs;
    const currentRun = allKnown.find((r) => r.run_id === selectedRunId);
    const currentJobId = currentRun?.job_id ?? selectedRunId;
    if (currentJobId !== prevJobIdRef.current) {
      prevJobIdRef.current = currentJobId ?? null;
      // If this job has exactly one island, auto-select it
      const islandRuns = allKnown.filter((r) => r.job_id === currentJobId && !r.is_job);
      if (islandRuns.length === 1) {
        setIslandMode("single");
        setSelectedRunId(islandRuns[0].run_id);
      } else {
        setIslandMode("all");
      }
    }
  }, [selectedRunId, allRuns, runs]);

  useEffect(() => {
    if (routeRunId && routeRunId !== selectedRunId) {
      setSelectedRunId(routeRunId);
    }
  }, [routeRunId, selectedRunId, setSelectedRunId]);

  useEffect(() => {
    const allKnown = allRuns.length ? allRuns : runs;
    if (selectedRunId && !allKnown.some((run) => run.run_id === selectedRunId)) {
      setSelectedRunId(null);
    }
  }, [runs, allRuns, selectedRunId, setSelectedRunId]);

  useEffect(() => {
    if (!selectedRunId && newestRun) {
      const runId = newestRun.island_run_ids?.[0] ?? newestRun.run_id;
      setSelectedRunId(runId);
    }
  }, [newestRun, selectedRunId, setSelectedRunId]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link to="/" className="brand">
          <div className="brand-mark">CVT</div>
          <div>
            <strong>Run Explorer</strong>
            <span>Adaptive MAP-Elites Viewer</span>
          </div>
        </Link>
        <nav className="nav-list">
          <NavLink to="/" end><FolderOpen size={18} /> Home</NavLink>
          <NavLink to={selectedRunId ? "/overview" : "/"} end><Activity size={18} /> Overview</NavLink>
          <NavLink to={selectedRunId ? "/archive" : "/"} end><Boxes size={18} /> Archive</NavLink>
          {activeRun?.curiosity_enabled ? (
            <NavLink to={selectedRunId ? "/sampling" : "/"}><Sparkles size={18} /> Sampling</NavLink>
          ) : null}
          <NavLink to={selectedRunId ? "/candidates" : "/"}><CandlestickChart size={18} /> Candidates</NavLink>
        </nav>
        <IslandSelector runs={allRuns.length ? allRuns : runs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} islandMode={islandMode} setIslandMode={setIslandMode} />
        <div className="sidebar-footer">
          <span>{runs.length} imported run{runs.length === 1 ? "" : "s"}</span>
          <span>{activeRun ? `${activeRun.project_id} · ${formatRunStamp(activeRun.created_at)}` : "No run selected"}</span>
        </div>
      </aside>
      <main className="main-shell">
        <header className="topbar" />
        {error ? <div className="error-banner">{error}</div> : null}
        <Routes>
          <Route
            path="/"
            element={
              <HomePage
                runs={runs}
                onImport={handleImport}
                onRefreshAll={handleRefreshAll}
                onRefresh={handleRefresh}
                onRemove={handleRemove}
                onCancel={handleCancel}
                onPause={handlePause}
                onResume={handleResume}
                selectedRunId={selectedRunId}
                onSelectRun={setSelectedRunId}
              />
            }
          />
          <Route
            path="/overview"
            element={<RunOverviewPage key={`${selectedRunId}:${islandMode}`} runId={selectedRunId} islandMode={islandMode} />}
          />
          <Route
            path="/archive"
            element={<ArchivePage key={`${selectedRunId}:${islandMode}`} runId={selectedRunId} islandMode={islandMode} />}
          />
          <Route
            path="/sampling"
            element={<SamplingPage key={`${selectedRunId}:${islandMode}`} runId={selectedRunId} islandMode={islandMode} />}
          />
          <Route path="/candidates" element={<CandidateBrowserPage key={`${selectedRunId}:${islandMode}`} runId={selectedRunId} islandMode={islandMode} />} />
          <Route path="/candidates/:candidateId" element={<CandidatePage runId={selectedRunId} />} />
          <Route path="/runs/:runId" element={<LegacyRunRedirect onSelectRun={setSelectedRunId} target="/overview" />} />
          <Route path="/runs/:runId/archive" element={<LegacyRunRedirect onSelectRun={setSelectedRunId} target="/archive" />} />
          <Route path="/runs/:runId/sampling" element={<LegacyRunRedirect onSelectRun={setSelectedRunId} target="/sampling" />} />
          <Route path="/runs/:runId/candidates" element={<LegacyRunRedirect onSelectRun={setSelectedRunId} target="/candidates" />} />
          <Route path="/runs/:runId/candidates/:candidateId" element={<LegacyCandidateRedirect onSelectRun={setSelectedRunId} />} />
        </Routes>
      </main>
    </div>
  );
}

function LegacyRunRedirect({ onSelectRun, target }: { onSelectRun: (runId: string | null) => void; target: string }) {
  const { runId } = useParams();

  useEffect(() => {
    if (runId) {
      onSelectRun(runId);
    }
  }, [onSelectRun, runId]);

  return <Navigate to={target} replace />;
}

function LegacyCandidateRedirect({ onSelectRun }: { onSelectRun: (runId: string | null) => void }) {
  const { runId, candidateId } = useParams();

  useEffect(() => {
    if (runId) {
      onSelectRun(runId);
    }
  }, [onSelectRun, runId]);

  return <Navigate to={candidateId ? `/candidates/${candidateId}` : "/candidates"} replace />;
}

function formatRunStamp(value?: string) {
  if (!value) {
    return "Unknown time";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function IslandSelector({ runs, selectedRunId, onSelect, islandMode, setIslandMode }: { runs: RunSummary[]; selectedRunId: string | null; onSelect: (id: string) => void; islandMode: string; setIslandMode: (mode: string) => void }) {
  const selected = runs.find((r) => r.run_id === selectedRunId);
  if (!selected) return null;

  // Find all runs sharing the same job_id
  const jobId = selected.job_id;
  if (!jobId) return null;
  const siblings = runs.filter((r) => r.job_id === jobId).sort((a, b) => a.run_id.localeCompare(b.run_id));
  if (siblings.length <= 1) return null;

  const selectValue = islandMode === "all" ? "all" : (selectedRunId ?? "");

  return (
    <div className="island-selector">
      <label className="island-selector-label">Island</label>
      <select
        value={selectValue}
        onChange={(e) => {
          const val = e.target.value;
          if (val === "all") {
            setIslandMode("all");
            // Reset to first island so all pages have a valid runId
            const firstIsland = siblings[0]?.run_id;
            if (firstIsland) onSelect(firstIsland);
          } else {
            setIslandMode(val);
            onSelect(val);
          }
        }}
        className="island-selector-select"
      >
        <option value="all">All islands</option>
        {siblings.map((s) => (
          <option key={s.run_id} value={s.run_id}>
            {s.run_id.match(/island_\d+/)?.[0] ?? s.run_id.slice(-8)}
          </option>
        ))}
      </select>
    </div>
  );
}

function compareRunsDescending(left: RunSummary, right: RunSummary) {
  const leftTime = left.created_at ? Date.parse(left.created_at) : Number.NEGATIVE_INFINITY;
  const rightTime = right.created_at ? Date.parse(right.created_at) : Number.NEGATIVE_INFINITY;
  if (leftTime !== rightTime) {
    return rightTime - leftTime;
  }
  return right.run_id.localeCompare(left.run_id);
}
