import type {
  CandidateDetailResponse,
  CandidateListResponse,
  IslandInfoResponse,
  RunCompareResponse,
  RunSummary,
  RunSummaryResponse,
  SnapshotResponse,
  TimingRecord,
  TimeseriesResponse,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(String(payload.detail ?? response.statusText));
  }
  return (await response.json()) as T;
}

export const api = {
  importRuns(path: string) {
    return request<{ status: string; runs: RunSummary[] }>("/api/runs/import", {
      method: "POST",
      body: JSON.stringify({ path }),
    });
  },
  listRuns() {
    return request<RunSummary[]>("/api/runs");
  },
  listJobs() {
    return request<RunSummary[]>("/api/jobs");
  },
  getMergedSnapshot(runId: string, step?: number): Promise<SnapshotResponse> {
    const q = step !== undefined ? `?step=${step}` : "";
    return request<SnapshotResponse>(`/api/runs/${runId}/snapshot/merged${q}`);
  },
  refreshAllRuns() {
    return request<{ status: string; runs: RunSummary[] }>("/api/runs/refresh-all", {
      method: "POST",
    });
  },
  refreshRun(runId: string) {
    return request<RunSummary>(`/api/runs/${runId}/refresh`, {
      method: "POST",
    });
  },
  refreshProject(projectId: string) {
    return request<{ status: string; count: number }>(`/api/runs/refresh-project/${encodeURIComponent(projectId)}`, {
      method: "POST",
    });
  },
  removeRun(runId: string) {
    return request<{ status: string }>(`/api/runs/${runId}`, {
      method: "DELETE",
    });
  },
  cancelRun(runId: string) {
    return request<{ status: string; pid: string }>(`/api/runs/${runId}/cancel`, {
      method: "POST",
    });
  },
  pauseRun(runId: string) {
    return request<{ status: string; pid: string }>(`/api/runs/${runId}/pause`, {
      method: "POST",
    });
  },
  resumeRun(runId: string) {
    return request<{ status: string; pid: string }>(`/api/runs/${runId}/resume`, {
      method: "POST",
    });
  },
  getRunSummary(runId: string) {
    return request<RunSummaryResponse>(`/api/runs/${runId}/summary`);
  },
  getTimeseries(runId: string) {
    return request<TimeseriesResponse>(`/api/runs/${runId}/timeseries`);
  },
  getTiming(runId: string) {
    return request<{ timing: TimingRecord[] }>(`/api/runs/${runId}/timing`);
  },
  getLog(runId: string, lines = 50) {
    return request<{ lines: string[] }>(`/api/runs/${runId}/log?lines=${lines}`);
  },
  getEvents(runId: string, query = "") {
    return request<Record<string, unknown>[]>(`/api/runs/${runId}/events${query ? `?${query}` : ""}`);
  },
  getSnapshotSteps(runId: string) {
    return request<{ steps: number[] }>(`/api/runs/${runId}/archive/snapshots`);
  },
  getSnapshot(runId: string, step: number) {
    return request<SnapshotResponse>(`/api/runs/${runId}/archive/snapshots/${step}`);
  },
  getCandidates(runId: string, params: URLSearchParams) {
    return request<CandidateListResponse>(`/api/runs/${runId}/candidates?${params.toString()}`);
  },
  getCandidate(runId: string, candidateId: string) {
    return request<CandidateDetailResponse>(`/api/runs/${runId}/candidates/${candidateId}`);
  },
  getRunCompare(runId: string) {
    return request<RunCompareResponse>(`/api/runs/${runId}/compare`);
  },
  async getIslandInfo(runId: string): Promise<IslandInfoResponse> {
    const resp = await fetch(`/api/runs/${runId}/island_info`);
    if (!resp.ok) throw new Error(`Failed to fetch island info: ${resp.status}`);
    return resp.json();
  },
  getCombinedCandidates(runId: string, limit = 5000) {
    return request<CandidateListResponse>(`/api/runs/${runId}/candidates/combined?limit=${limit}`);
  },
};
