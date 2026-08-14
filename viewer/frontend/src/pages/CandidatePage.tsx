import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { CandidateDetailContent } from "../components/CandidateDetailContent";
import { LoadingBar } from "../components/LoadingBar";
import type { CandidateDetailResponse } from "../types";

type CandidatePageProps = {
  runId: string | null;
};

export function CandidatePage({ runId }: CandidatePageProps) {
  const { candidateId = "" } = useParams();
  const [searchParams] = useSearchParams();
  // ?run= overrides the global runId for cross-island candidate lookups
  const effectiveRunId = searchParams.get("run") ?? runId;
  const [detail, setDetail] = useState<CandidateDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!effectiveRunId || !candidateId) {
      setDetail(null);
      return;
    }
    const activeRunId = effectiveRunId;
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const response = await api.getCandidate(activeRunId, candidateId);
        if (!cancelled) {
          setDetail(response);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [candidateId, effectiveRunId]);

  if (!runId) {
    return <NoRunSelected message="Choose a run on Home before opening candidate detail pages." />;
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }
  if (!detail) {
    return (
      <>
        <LoadingBar active={loading} label="Loading candidate" />
        <div className="loading-panel">Loading candidate...</div>
      </>
    );
  }

  const candidate = detail.candidate;
  return (
    <>
      <LoadingBar active={loading} label="Loading candidate" />
      <section className="panel hero-panel candidate-page-panel">
        <div className="panel-header">
          <div>
            <h1>{candidate.id}</h1>
            <p>
              Cell {formatValue(candidate.cell_id)} · Generation {candidate.generation} · Step {candidate.created_at_step}
            </p>
          </div>
        </div>
        <CandidateDetailContent detail={detail} />
      </section>
    </>
  );
}

function formatValue(value: number | string | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return String(value);
}

function NoRunSelected({ message }: { message: string }) {
  return (
    <section className="panel empty-state">
      <h2>No Run Selected</h2>
      <p>{message}</p>
    </section>
  );
}
