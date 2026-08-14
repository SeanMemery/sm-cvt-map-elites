import { useEffect, useState } from "react";

import { api } from "../api";
import { CandidateTable } from "../components/CandidateTable";
import { LoadingBar } from "../components/LoadingBar";
import type { CandidateRecord, IslandInfoResponse, RunSummaryResponse } from "../types";

/** Build a map from run_id → descriptor_labels by loading all sibling summaries. */
async function buildDescriptorMap(
  primaryRunId: string,
  primaryLabels: string[],
  siblingIds: string[],
): Promise<Record<string, string[]>> {
  const map: Record<string, string[]> = { [primaryRunId]: primaryLabels };
  await Promise.all(
    siblingIds.map(async (sid) => {
      try {
        const s = await api.getRunSummary(sid);
        map[sid] = s.descriptor_labels ?? primaryLabels;
      } catch { /* use primary labels as fallback */ }
    })
  );
  return map;
}

type CandidateBrowserPageProps = {
  runId: string | null;
  islandMode: string;
};

export function CandidateBrowserPage({ runId, islandMode }: CandidateBrowserPageProps) {
  const [summary, setSummary] = useState<RunSummaryResponse | null>(null);
  const [candidates, setCandidates] = useState<CandidateRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [islandInfo, setIslandInfo] = useState<IslandInfoResponse | null>(null);
  const [descriptorMap, setDescriptorMap] = useState<Record<string, string[]>>({});

  useEffect(() => {
    if (!runId) {
      setSummary(null);
      setCandidates([]);
      setIslandInfo(null);
      setError(null);
      return;
    }
    const activeRunId = runId;
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        params.set("limit", "750");
        params.set("sort", "created_at_step:desc");
        params.set("include_code", "false");
        const [summaryResponse, islandInfoResponse] = await Promise.all([
          api.getRunSummary(activeRunId),
          api.getIslandInfo(activeRunId).catch(() => null),
        ]);
        const hasSiblings = (islandInfoResponse?.sibling_run_ids.length ?? 0) > 0;
        const candidateResponse = (islandMode === "all" && hasSiblings)
          ? await api.getCombinedCandidates(activeRunId)
          : await api.getCandidates(activeRunId, params);
        if (!cancelled) {
          setSummary(summaryResponse);
          setIslandInfo(islandInfoResponse);
          setCandidates(candidateResponse.items);
          setTotal(candidateResponse.total);
          // Build descriptor label map for all islands
          const siblingIds = islandInfoResponse?.sibling_run_ids ?? [];
          if (siblingIds.length > 0) {
            buildDescriptorMap(activeRunId, summaryResponse.descriptor_labels, siblingIds)
              .then((map) => { if (!cancelled) setDescriptorMap(map); });
          } else {
            setDescriptorMap({ [activeRunId]: summaryResponse.descriptor_labels });
          }
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => { cancelled = true; };
  }, [runId, islandMode]);

  if (!runId) {
    return (
      <section className="panel empty-state">
        <h2>No Run Selected</h2>
        <p>Choose a run on Home, then use the left navigation to inspect its candidate list.</p>
      </section>
    );
  }
  if (error) return <div className="error-banner">{error}</div>;
  if (!summary) {
    return (
      <>
        <LoadingBar active={loading} label="Loading candidates" />
        <div className="loading-panel">Loading candidates...</div>
      </>
    );
  }

  return (
    <div className="page-grid">
      <LoadingBar active={loading} label="Loading candidates" />
      <section className="panel">
        <div className="panel-header">
          <div>
            <h1>Candidate Browser</h1>
            <p>{formatRunStamp(summary.summary.project_id, summary.summary.created_at)}</p>
          </div>
          <div className="muted">
            {islandMode === "all" && (islandInfo?.sibling_run_ids.length ?? 0) > 0
              ? `All islands — ${total} candidates`
              : `${total} candidates`}
          </div>
        </div>
        <CandidateTable
          candidates={candidates}
          primaryMetricLabel={summary.primary_metric_label}
          secondaryMetricLabel={summary.secondary_metric_label}
          descriptorLabels={summary.descriptor_labels}
          descriptorMap={descriptorMap}
        />
      </section>
    </div>
  );
}

function formatRunStamp(projectId: string, createdAt?: string) {
  if (!createdAt) return projectId;
  const date = new Date(createdAt);
  return `${projectId} · ${Number.isNaN(date.getTime()) ? createdAt : date.toLocaleString()}`;
}
