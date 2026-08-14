import type { CandidateDetailResponse } from "../types";
import { CandidateDetailContent } from "./CandidateDetailContent";
import { DetailModal } from "./DetailModal";

type CandidateDetailModalProps = {
  detail: CandidateDetailResponse;
  onClose: () => void;
};

export function CandidateDetailModal({ detail, onClose }: CandidateDetailModalProps) {
  const candidate = detail.candidate;
  return (
    <DetailModal
      title={candidate.id}
      subtitle={`Cell ${formatValue(candidate.cell_id)} · Generation ${candidate.generation} · Step ${candidate.created_at_step}`}
      onClose={onClose}
    >
      <CandidateDetailContent detail={detail} />
    </DetailModal>
  );
}

function formatValue(value: number | string | null | undefined) {
  if (value === null || value === undefined) {
    return "-";
  }
  return String(value);
}
