export type RunSummary = {
  run_id: string;
  run_name: string;
  project_id: string;
  job_id?: string | null;
  path: string;
  created_at?: string;
  status?: string;
  raw_status?: string;
  last_updated_at?: string;
  current_step: number;
  candidate_count: number;
  archive_occupancy: number;
  active_candidate_count: number;
  best_primary_fitness?: number | null;
  best_primary_validation_fitness?: number | null;
  best_secondary_fitness?: number | null;
  best_secondary_validation_fitness?: number | null;
  latest_stats?: Record<string, unknown>;
  descriptor_labels: string[];
  primary_metric_label: string;
  primary_validation_metric_label: string;
  secondary_metric_label: string;
  secondary_validation_metric_label: string;
  curiosity_enabled?: boolean;
  current_stage?: number;
  total_stages?: number;
  is_job?: boolean;
  island_count?: number;
  island_run_ids?: string[];
};

export type RunSummaryResponse = {
  summary: RunSummary;
  metadata: Record<string, unknown>;
  config: Record<string, unknown>;
  descriptor_labels: string[];
  primary_metric_label: string;
  primary_validation_metric_label: string;
  secondary_metric_label: string;
  secondary_validation_metric_label: string;
  snapshot_steps: number[];
  numeric_metadata_fields: string[];
  island_id: string | null;
  job_id: string | null;
};

export type IslandInfoResponse = {
  island_id: string | null;
  job_id: string | null;
  sibling_run_ids: string[];
};

export type EventRecord = Record<string, unknown> & {
  type: string;
  step?: number;
  candidate_id?: string;
};

export type TimelineBadge = {
  label: string;
  value: unknown;
};

export type TimelineField = {
  label: string;
  value: unknown;
  kind?: string;
};

export type TimelineEventRecord = EventRecord & {
  title?: string;
  summary?: string;
  tone?: string;
  badges?: TimelineBadge[];
  fields?: TimelineField[];
};

export type CandidateRecord = {
  id: string;
  code: string;
  metadata: Record<string, unknown>;
  llm_call_id?: string | null;
  descriptor_raw?: number[] | null;
  descriptor_norm?: number[] | null;
  descriptor_values?: Record<string, number | null>;
  primary_fitness?: number | null;
  primary_validation_fitness?: number | null;
  secondary_fitness?: number | null;
  secondary_validation_fitness?: number | null;
  parent_ids: string[];
  generation: number;
  created_at_step: number;
  cell_id?: number | null;
  is_active: boolean;
  record_version: number;
  stage?: number;
  stats?: Record<string, unknown> | null;
};

export type CandidateDetailResponse = {
  candidate: CandidateRecord;
  history: CandidateRecord[];
  events: EventRecord[];
  parents: CandidateRecord[];
  target_candidate?: CandidateRecord | null;
  ancestor_candidates?: CandidateRecord[];
  inspiration_candidates?: CandidateRecord[];
  children: CandidateRecord[];
  llm_call?: { phase?: string | null; [key: string]: unknown } | null;
  descriptor_labels: string[];
  primary_metric_label: string;
  secondary_metric_label: string;
};

export type CandidateListResponse = {
  total: number;
  items: CandidateRecord[];
  descriptor_labels: string[];
  primary_metric_label: string;
  secondary_metric_label: string;
};

export type SnapshotCell = {
  cell_id: number;
  candidate_id: string;
  primary_fitness?: number | null;
  primary_validation_fitness?: number | null;
  secondary_fitness?: number | null;
  secondary_validation_fitness?: number | null;
  curiosity_score?: number | null;
  emitter_curiosity_scores?: Record<string, number> | null;
  descriptor_raw?: number[] | null;
  descriptor_norm?: number[] | null;
  descriptor_values?: Record<string, number | null>;
};

export type SnapshotResponse = {
  step: number;
  normalizer_bounds: {
    lower: number[];
    upper: number[];
  };
  descriptor_labels?: string[];
  centroids?: number[][];
  occupied_cells: SnapshotCell[];
};

export type StageMarker = { step: number; stage: number };

export type TimeseriesResponse = {
  stats: Record<string, unknown>[];
  remaps: EventRecord[];
  secondary_eval_markers: EventRecord[];
  checkpoints: EventRecord[];
  timeline_events: TimelineEventRecord[];
  events?: EventRecord[];
  stage_markers?: StageMarker[];
};

export type TimingRecord = {
  step: number;
  n_candidates: number;
  generation_seconds: number | null;
  wall_seconds: number | null;
  eval_seconds: number | null;
};

export type RunCompareResponse = {
  secondary_scatter: Array<{
    candidate_id: string;
    primary_fitness?: number | null;
    secondary_fitness?: number | null;
    created_at_step?: number | null;
    record_version?: number | null;
    is_active?: boolean;
  }>;
  descriptor_histograms: Array<{
    label: string;
    values: number[];
  }>;
  feature_extremes: Array<{
    label: string;
    min: {
      candidate_id?: string | null;
      feature_value?: number | null;
      primary_fitness?: number | null;
      secondary_fitness?: number | null;
      cell_id?: number | null;
    };
    max: {
      candidate_id?: string | null;
      feature_value?: number | null;
      primary_fitness?: number | null;
      secondary_fitness?: number | null;
      cell_id?: number | null;
    };
  }>;
  record_timeline: {
    primary: Array<{
      candidate_id: string;
      created_at_step: number;
      fitness: number;
    }>;
    primary_validation: Array<{
      candidate_id: string;
      created_at_step: number;
      fitness: number;
    }>;
    secondary: Array<{
      candidate_id: string;
      created_at_step: number;
      fitness: number;
    }>;
    secondary_validation: Array<{
      candidate_id: string;
      created_at_step: number;
      fitness: number;
    }>;
  };
  remaps: EventRecord[];
  primary_metric_label: string;
  primary_validation_metric_label: string;
  secondary_metric_label: string;
  secondary_validation_metric_label: string;
};
