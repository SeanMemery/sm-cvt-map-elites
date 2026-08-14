from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from viewer.backend.run_loader import ImportedRun, RunLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "viewer" / "frontend" / "dist"
VIEWER_DATA_DIR = REPO_ROOT / "viewer_data"
IMPORT_REGISTRY = VIEWER_DATA_DIR / "imported_runs.json"


def create_app() -> FastAPI:
    app = FastAPI(title="Adaptive CVT-MAP-Elites Run Explorer")
    registry_path = Path(os.environ.get("VIEWER_IMPORT_REGISTRY", str(IMPORT_REGISTRY))).expanduser().resolve()
    discovery_roots_env = os.environ.get("VIEWER_DISCOVERY_ROOTS", "").strip()
    discovery_roots = [
        Path(item).expanduser().resolve()
        for item in discovery_roots_env.split(":")
        if item.strip()
    ]
    run_name_prefixes_env = os.environ.get("VIEWER_RUN_NAME_PREFIXES", "").strip()
    run_name_prefixes = tuple(item.strip() for item in run_name_prefixes_env.split(",") if item.strip())
    loader = RunLoader(
        registry_path,
        discovery_roots=discovery_roots,
        run_name_prefixes=run_name_prefixes,
    )

    def _get_run_or_404(run_id: str) -> ImportedRun:
        try:
            return loader.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc

    @app.post("/api/runs/import")
    def import_run(body: dict[str, str]) -> dict[str, Any]:
        path = body.get("path", "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="Path is required")
        try:
            runs = loader.import_path(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok", "runs": runs}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return loader.list_runs()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        """List runs grouped by job. Multi-island runs appear as one job entry."""
        all_runs = loader.list_runs()
        jobs: dict[str, dict[str, Any]] = {}
        standalone: list[dict[str, Any]] = []
        for run in all_runs:
            job_id = run.get("job_id")
            if job_id:
                if job_id not in jobs:
                    jobs[job_id] = {
                        **run,  # inherit all fields from first island (project_id, labels, etc.)
                        "job_id": job_id,
                        "run_id": job_id,
                        "run_name": run.get("run_name", "").rsplit("_island_", 1)[0],
                        "is_job": True,
                        "island_count": 0,
                        "island_run_ids": [],
                        "best_primary_fitness": None,
                        "current_step": run.get("current_step", 0),
                    }
                entry = jobs[job_id]
                entry["island_count"] += 1
                entry["island_run_ids"].append(run["run_id"])
                # Aggregate best fitness across islands
                bf = run.get("best_primary_fitness")
                if bf is not None:
                    entry["best_primary_fitness"] = max(entry["best_primary_fitness"] or 0.0, bf)
                # Latest step across islands
                entry["current_step"] = max(entry["current_step"], run.get("current_step", 0))
                # Keep running if any island is running
                if run.get("status") == "running":
                    entry["status"] = "running"
            else:
                standalone.append({**run, "is_job": False, "island_count": 1})
        return list(jobs.values()) + standalone

    @app.post("/api/runs/refresh-all")
    def refresh_all_runs() -> dict[str, Any]:
        return {"status": "ok", "runs": loader.refresh_all_runs()}

    @app.post("/api/runs/refresh-project/{project_id}")
    def refresh_project_runs(project_id: str) -> dict[str, Any]:
        refreshed = loader.refresh_project_runs(project_id)
        return {"status": "ok", "count": len(refreshed), "project_id": project_id, "runs": refreshed}

    @app.post("/api/runs/{run_id}/refresh")
    def refresh_run(run_id: str) -> dict[str, Any]:
        try:
            return loader.refresh_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc

    @app.delete("/api/runs/{run_id}")
    def remove_run(run_id: str) -> dict[str, str]:
        try:
            loader.remove_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc
        return {"status": "ok"}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, str]:
        run = _get_run_or_404(run_id)
        pid_file = Path(run.path) / "pid"
        if not pid_file.exists():
            raise HTTPException(status_code=404, detail="No PID file found — run may not be active")
        try:
            import signal
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            return {"status": "ok", "pid": str(pid)}
        except ProcessLookupError:
            raise HTTPException(status_code=404, detail="Process not found — already stopped?")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/runs/{run_id}/pause")
    def pause_run(run_id: str) -> dict[str, str]:
        run = _get_run_or_404(run_id)
        pid_file = Path(run.path) / "pid"
        if not pid_file.exists():
            raise HTTPException(status_code=404, detail="No PID file — run may not be active")
        try:
            import signal
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGSTOP)
            (Path(run.path) / "paused").write_text(str(pid))
            return {"status": "paused", "pid": str(pid)}
        except ProcessLookupError:
            raise HTTPException(status_code=404, detail="Process not found")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str) -> dict[str, str]:
        run = _get_run_or_404(run_id)
        pid_file = Path(run.path) / "pid"
        paused_file = Path(run.path) / "paused"
        if not pid_file.exists():
            raise HTTPException(status_code=404, detail="No PID file — run may not be active")
        try:
            import signal
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGCONT)
            paused_file.unlink(missing_ok=True)
            return {"status": "resumed", "pid": str(pid)}
        except ProcessLookupError:
            paused_file.unlink(missing_ok=True)
            raise HTTPException(status_code=404, detail="Process not found")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        return {
            "summary": run.summary(),
            "metadata": run.metadata,
            "config": run.config,
            "descriptor_labels": run.descriptor_labels,
            "primary_metric_label": run.primary_metric_label,
            "primary_validation_metric_label": run.primary_validation_metric_label,
            "secondary_metric_label": run.secondary_metric_label,
            "secondary_validation_metric_label": run.secondary_validation_metric_label,
            "snapshot_steps": run.snapshot_steps,
            "numeric_metadata_fields": run.numeric_metadata_fields,
            "island_id": run.island_id,
            "job_id": run.job_id,
        }

    @app.get("/api/runs/{run_id}/island_info")
    def run_island_info(run_id: str) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        sibling_run_ids: list[str] = []
        if run.job_id is not None:
            sibling_run_ids = [
                other_id
                for other_id, other_run in loader.runs.items()
                if other_id != run_id and other_run.job_id == run.job_id
            ]
        return {
            "island_id": run.island_id,
            "job_id": run.job_id,
            "sibling_run_ids": sibling_run_ids,
        }

    @app.get("/api/runs/{run_id}/snapshot/merged")
    def run_snapshot_merged(run_id: str, step: int = Query(default=-1)) -> dict[str, Any]:
        """Merge archive snapshots across all islands in the same job.

        Returns a snapshot where each cell holds the best elite across all islands.
        If step=-1, uses the latest snapshot per island.
        """
        run = _get_run_or_404(run_id)
        if run.job_id is None:
            # Not part of a multi-island job — fall back to the run's own snapshot
            snap = run.latest_snapshot() if step == -1 else run.load_snapshot(step)
            return snap or {}

        sibling_ids = [
            rid for rid, r in loader.runs.items()
            if r.job_id == run.job_id
        ]

        # Collect snapshots from all siblings
        merged_cells: dict[int, dict[str, Any]] = {}
        centroids: list | None = None
        normalizer_bounds: dict | None = None
        descriptor_labels: list | None = None

        for sid in sibling_ids:
            sibling = loader.runs.get(sid)
            if sibling is None:
                continue
            if step == -1:
                snap = sibling.latest_snapshot()
            else:
                # find the closest step <= requested step
                available = [s for s in sibling.snapshot_steps if s <= step]
                snap = sibling.load_snapshot(max(available)) if available else None
            if not snap:
                continue
            if centroids is None:
                centroids = snap.get("centroids")
            if normalizer_bounds is None:
                normalizer_bounds = snap.get("normalizer_bounds")
            if descriptor_labels is None:
                descriptor_labels = snap.get("descriptor_labels")
            for cell in snap.get("occupied_cells", []):
                cell_id = cell.get("cell_id")
                if cell_id is None:
                    continue
                existing = merged_cells.get(cell_id)
                if existing is None or (cell.get("primary_fitness") or 0) > (existing.get("primary_fitness") or 0):
                    merged_cells[cell_id] = {**cell, "island_id": sibling.island_id}

        return {
            "centroids": centroids or [],
            "normalizer_bounds": normalizer_bounds or {},
            "descriptor_labels": descriptor_labels or run.descriptor_labels,
            "occupied_cells": list(merged_cells.values()),
            "island_count": len(sibling_ids),
        }

    @app.get("/api/runs/{run_id}/timeseries")
    def run_timeseries(run_id: str) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        remaps = [event for event in run.events if event.get("type") == "remap"]
        secondary_events = _enrich_secondary_events(run)
        checkpoints = [event for event in run.events if event.get("type") == "checkpoint_saved"]
        return {
            "stats": run.stats,
            "remaps": remaps,
            "secondary_eval_markers": secondary_events,
            "checkpoints": checkpoints,
            "timeline_events": _build_timeline_events(run),
        }

    @app.get("/api/runs/{run_id}/timing")
    def run_timing(run_id: str) -> dict[str, Any]:
        """Per-step timing data derived from llm_calls.jsonl.

        Returns a list of records, one per LLM call, with:
          step, n_candidates (cumulative), generation_seconds, wall_seconds, eval_seconds
        """
        import math
        from datetime import datetime, timezone

        run = _get_run_or_404(run_id)
        calls = sorted(run.llm_calls, key=lambda c: c.get("step", 0))

        def _parse_dt(s: str | None):
            if not s:
                return None
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

        # With parallel workers, calls overlap. For each call:
        # - generation_seconds = LLM API latency (directly tracked)
        # - eval_seconds = gap from this call's finished_at to the next call that
        #   started AFTER it finished (best proxy for evaluation time in the same worker)
        # - wall_seconds = generation + eval

        # Sort by started_at for gap estimation
        dt_calls = []
        for call in calls:
            dt_calls.append({
                **call,
                "_started": _parse_dt(call.get("started_at")),
                "_finished": _parse_dt(call.get("finished_at")),
            })

        records = []
        cumulative = 0
        for i, call in enumerate(dt_calls):
            step = call.get("step", i)
            n_children = len(call.get("child_ids") or [])
            cumulative += max(n_children, 1)

            generation_s = call.get("latency_seconds")
            finished = call["_finished"]

            # Find the next call that started after this one finished (same-worker proxy)
            eval_s = None
            if finished is not None:
                later_starts = [
                    c["_started"] for c in dt_calls
                    if c["_started"] is not None and c["_started"] > finished
                ]
                if later_starts:
                    next_start = min(later_starts)
                    gap = (next_start - finished).total_seconds()
                    eval_s = max(0.0, gap)

            wall_s = ((generation_s or 0) + eval_s) if eval_s is not None else generation_s

            records.append({
                "step": step,
                "n_candidates": cumulative,
                "generation_seconds": generation_s,
                "wall_seconds": wall_s,
                "eval_seconds": eval_s,
            })

        return {"timing": records}

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str, lines: int = 50) -> dict[str, Any]:
        """Return recent run activity as formatted log lines for the terminal view."""
        run = _get_run_or_404(run_id)

        log_lines: list[str] = []

        # Recent events → human-readable lines
        recent_events = run.events[-200:] if run.events else []
        for event in recent_events:
            t = event.get("type", "")
            step = event.get("step", "?")
            if t == "candidate_inserted":
                cid = event.get("candidate_id", "?")
                cell = event.get("cell_id", "?")
                replaced = event.get("replaced_candidate_id")
                msg = f"[step {step}] ✓ {cid} → cell {cell}"
                if replaced:
                    msg += f" (replaced {replaced})"
                log_lines.append(msg)
            elif t == "candidate_not_inserted":
                cid = event.get("candidate_id", "?")
                log_lines.append(f"[step {step}] ✗ {cid} not inserted")
            elif t == "candidate_rejected":
                cid = event.get("candidate_id", "?")
                reason = event.get("reason", "unknown")
                log_lines.append(f"[step {step}] ✗ {cid} rejected ({reason})")
            elif t == "remap":
                occupied = event.get("after_occupied_cells", "?")
                log_lines.append(f"[step {step}] ~ archive remapped → {occupied} cells occupied")
            elif t == "checkpoint_saved":
                log_lines.append(f"[step {step}] ✦ checkpoint saved")
            elif t == "stage_end":
                stage = event.get("stage", "?")
                log_lines.append(f"[step {step}] ► stage {stage} complete")

        # Recent LLM calls → status lines
        recent_calls = run.llm_calls[-20:] if run.llm_calls else []
        for call in recent_calls:
            step = call.get("step", "?")
            status = call.get("status", "?")
            latency = call.get("latency_seconds")
            emitter = call.get("emitter_name", "")
            child_ids = call.get("child_ids") or []
            lat_str = f" ({latency:.1f}s)" if latency else ""
            children_str = f" → {', '.join(child_ids)}" if child_ids else ""
            log_lines.append(f"[step {step}] LLM {emitter} {status}{lat_str}{children_str}")

        # Add current status header
        summary_data = run.summary()
        current_step = summary_data.get("current_step", 0)
        run_status = summary_data.get("status", "unknown")
        best_fitness = summary_data.get("best_primary_fitness")
        header = f"── {run_id} | step {current_step} | {run_status}"
        if best_fitness is not None:
            header += f" | best {best_fitness:.4f}"
        log_lines = [header, ""] + log_lines

        return {"lines": log_lines[-lines:]}

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str,
        type: str | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        run = _get_run_or_404(run_id)
        events = run.events
        if type:
            events = [event for event in events if event.get("type") == type]
        if start_step is not None:
            events = [event for event in events if int(event.get("step", -1)) >= start_step]
        if end_step is not None:
            events = [event for event in events if int(event.get("step", -1)) <= end_step]
        if candidate_id:
            events = [
                event
                for event in events
                if event.get("candidate_id") == candidate_id
                or candidate_id in event.get("parent_ids", [])
                or candidate_id in event.get("lost_candidate_ids", [])
            ]
        return events

    @app.get("/api/runs/{run_id}/archive/snapshots")
    def archive_snapshot_steps(run_id: str) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        return {"steps": run.snapshot_steps}

    @app.get("/api/runs/{run_id}/archive/snapshots/{step}")
    def archive_snapshot(run_id: str, step: int) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        try:
            return run.load_snapshot(step)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown snapshot step: {step}") from exc

    @app.get("/api/runs/{run_id}/candidates")
    def list_candidates(
        run_id: str,
        active: bool | None = None,
        secondary_evaluated: bool | None = None,
        min_primary_fitness: float | None = None,
        min_secondary_fitness: float | None = None,
        candidate_ids: str | None = None,
        include_code: bool = True,
        cell_id: int | None = None,
        parent_candidate_id: str | None = None,
        limit: int = Query(200, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        sort: str = "created_at_step:desc",
        search: str | None = None,
    ) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        candidates = list(run.candidates.latest_by_id.values())
        if active is not None:
            candidates = [candidate for candidate in candidates if bool(candidate.get("is_active", True)) is active]
        if secondary_evaluated is not None:
            candidates = [
                candidate
                for candidate in candidates
                if (candidate.get("secondary_fitness") is not None) is secondary_evaluated
            ]
        if min_primary_fitness is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("primary_fitness") is not None and float(candidate["primary_fitness"]) >= min_primary_fitness
            ]
        if min_secondary_fitness is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("secondary_fitness") is not None and float(candidate["secondary_fitness"]) >= min_secondary_fitness
            ]
        if candidate_ids:
            requested_ids = {item.strip() for item in candidate_ids.split(",") if item.strip()}
            candidates = [candidate for candidate in candidates if candidate.get("id") in requested_ids]
        if cell_id is not None:
            candidates = [candidate for candidate in candidates if candidate.get("cell_id") == cell_id]
        if parent_candidate_id:
            candidates = [candidate for candidate in candidates if parent_candidate_id in candidate.get("parent_ids", [])]
        if search:
            lowered = search.lower()
            candidates = [
                candidate
                for candidate in candidates
                if lowered in str(candidate.get("id", "")).lower() or lowered in candidate.get("code", "").lower()
            ]
        sort_field, _, direction = sort.partition(":")
        reverse = direction.lower() != "asc"
        candidates = sorted(candidates, key=lambda item: _sort_value(item.get(sort_field)), reverse=reverse)
        total = len(candidates)
        items = [_add_descriptor_values(c) for c in candidates[offset : offset + limit]]
        if not include_code:
            items = [_strip_candidate_code(c) for c in items]
        return {
            "total": total,
            "items": items,
            "descriptor_labels": run.descriptor_labels,
            "primary_metric_label": run.primary_metric_label,
            "primary_validation_metric_label": run.primary_validation_metric_label,
            "secondary_metric_label": run.secondary_metric_label,
            "secondary_validation_metric_label": run.secondary_validation_metric_label,
        }

    @app.get("/api/runs/{run_id}/candidates/combined")
    def list_candidates_combined(
        run_id: str,
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> dict[str, Any]:
        """Return candidates from all islands in the same job, combined and deduplicated."""
        run = _get_run_or_404(run_id)
        if run.job_id is None:
            candidates = [_add_descriptor_values(c) for c in run.candidates.latest_by_id.values()]
            candidates = sorted(candidates, key=lambda item: _sort_value(item.get("primary_fitness")), reverse=True)
            total = len(candidates)
            return {
                "total": total,
                "items": [_strip_candidate_code(c) for c in candidates[:limit]],
                "descriptor_labels": run.descriptor_labels,
                "primary_metric_label": run.primary_metric_label,
                "primary_validation_metric_label": run.primary_validation_metric_label,
                "secondary_metric_label": run.secondary_metric_label,
                "secondary_validation_metric_label": run.secondary_validation_metric_label,
            }
        sibling_ids = [rid for rid, r in loader.runs.items() if r.job_id == run.job_id]

        all_candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for sid in sibling_ids:
            sibling = loader.runs.get(sid)
            if sibling is None:
                continue
            for cand_id, record in sibling.candidates.latest_by_id.items():
                key = (sid, cand_id)
                if key in seen:
                    continue
                seen.add(key)
                enriched = _add_descriptor_values(
                    {**record, "island_run_id": sid, "_island_labels": sibling.descriptor_labels}
                )
                all_candidates.append(enriched)

        # Collect union of all descriptor label keys across candidates
        all_labels_set: set[str] = set()
        for c in all_candidates:
            all_labels_set.update(c.get("descriptor_values", {}).keys())
        all_labels = sorted(all_labels_set)
        all_candidates = sorted(all_candidates, key=lambda item: _sort_value(item.get("primary_fitness")), reverse=True)
        total = len(all_candidates)
        items = [_strip_candidate_code(c) for c in all_candidates[:limit]]
        return {
            "total": total,
            "items": items,
            "descriptor_labels": all_labels,
            "primary_metric_label": run.primary_metric_label,
            "primary_validation_metric_label": run.primary_validation_metric_label,
            "secondary_metric_label": run.secondary_metric_label,
            "secondary_validation_metric_label": run.secondary_validation_metric_label,
        }

    @app.get("/api/runs/{run_id}/candidates/{candidate_id}")
    def candidate_detail(run_id: str, candidate_id: str) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        latest = run.candidates.latest_by_id.get(candidate_id)
        if latest is None:
            raise HTTPException(status_code=404, detail=f"Unknown candidate: {candidate_id}")
        history = run.candidates.history_by_id.get(candidate_id, [])
        events = [
            event
            for event in run.events
            if event.get("candidate_id") == candidate_id
            or candidate_id in event.get("lost_candidate_ids", [])
        ]
        parent_summaries = [
            run.candidates.latest_by_id[parent_id]
            for parent_id in latest.get("parent_ids", [])
            if parent_id in run.candidates.latest_by_id
        ]
        child_summaries = [
            run.candidates.latest_by_id[child_id]
            for child_id in run.candidates.children_by_parent_id.get(candidate_id, [])
            if child_id in run.candidates.latest_by_id
        ]
        metadata = latest.get("metadata") or {}
        target_candidate_id = metadata.get("target_candidate_id")
        ancestor_ids = metadata.get("ancestor_ids") if isinstance(metadata.get("ancestor_ids"), list) else []
        inspiration_ids = metadata.get("inspiration_ids") if isinstance(metadata.get("inspiration_ids"), list) else []
        llm_call = run.llm_call_by_id.get(latest.get("llm_call_id", ""))
        return {
            "candidate": latest,
            "history": history,
            "events": events,
            "parents": parent_summaries,
            "target_candidate": run.candidates.latest_by_id.get(str(target_candidate_id)) if target_candidate_id else None,
            "ancestor_candidates": [
                run.candidates.latest_by_id[candidate_ref]
                for candidate_ref in ancestor_ids
                if isinstance(candidate_ref, str) and candidate_ref in run.candidates.latest_by_id
            ],
            "inspiration_candidates": [
                run.candidates.latest_by_id[candidate_ref]
                for candidate_ref in inspiration_ids
                if isinstance(candidate_ref, str) and candidate_ref in run.candidates.latest_by_id
            ],
            "children": child_summaries,
            "llm_call": llm_call,
            "descriptor_labels": run.descriptor_labels,
            "primary_metric_label": run.primary_metric_label,
            "primary_validation_metric_label": run.primary_validation_metric_label,
            "secondary_metric_label": run.secondary_metric_label,
            "secondary_validation_metric_label": run.secondary_validation_metric_label,
        }

    @app.get("/api/runs/{run_id}/compare")
    def compare_run(run_id: str) -> dict[str, Any]:
        run = _get_run_or_404(run_id)
        latest_snapshot = run.latest_snapshot() or {"occupied_cells": []}
        secondary_pairs = [
            {
                "candidate_id": candidate["id"],
                "primary_fitness": candidate.get("primary_fitness"),
                "secondary_fitness": candidate.get("secondary_fitness"),
                "created_at_step": candidate.get("created_at_step"),
                "record_version": candidate.get("record_version"),
                "is_active": candidate.get("is_active", True),
            }
            for candidate_history in run.candidates.history_by_id.values()
            for candidate in candidate_history
            if candidate.get("primary_fitness") is not None and candidate.get("secondary_fitness") is not None
        ]
        descriptor_histograms = []
        for index, label in enumerate(run.descriptor_labels):
            values = [
                candidate.get("descriptor_raw", [None] * len(run.descriptor_labels))[index]
                for candidate in latest_snapshot.get("occupied_cells", [])
                if candidate.get("descriptor_raw") and len(candidate["descriptor_raw"]) > index
            ]
            descriptor_histograms.append({"label": label, "values": values})
        remaps = [event for event in run.events if event.get("type") == "remap"]
        return {
            "secondary_scatter": secondary_pairs,
            "descriptor_histograms": descriptor_histograms,
            "feature_extremes": _feature_extremes_by_feature(run, latest_snapshot),
            "record_timeline": _record_timeline(run),
            "remaps": remaps,
            "primary_metric_label": run.primary_metric_label,
            "primary_validation_metric_label": run.primary_validation_metric_label,
            "secondary_metric_label": run.secondary_metric_label,
            "secondary_validation_metric_label": run.secondary_validation_metric_label,
        }

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str) -> FileResponse:
        if not FRONTEND_DIR.exists():
            raise HTTPException(
                status_code=503,
                detail="Frontend build not found. Run the viewer build before starting the backend.",
            )
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"Unknown API path: /{full_path}")
        target = FRONTEND_DIR / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIR / "index.html")

    return app


def _sort_value(value: Any) -> Any:
    if value is None:
        return float("-inf")
    return value


def _strip_candidate_code(candidate: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(candidate)
    stripped["code"] = ""
    return stripped


def _add_descriptor_values(candidate: dict[str, Any]) -> dict[str, Any]:
    """Add a descriptor_values dict keyed by label name.

    Sources: stats.all_descriptors.value (full set stored at fitness time),
    falling back to island descriptor_raw + island labels.
    """
    stats = candidate.get("stats") or {}
    stored: dict[str, float] = (stats.get("all_descriptors") or {}).get("value") or {}

    if not stored:
        island_labels: list[str] = candidate.get("_island_labels") or []
        island_raw: list[float] = candidate.get("descriptor_raw") or []
        stored = {lbl: island_raw[i] for i, lbl in enumerate(island_labels) if i < len(island_raw)}

    enriched = dict(candidate)
    enriched["descriptor_values"] = stored
    return enriched


def _find_candidate(run: ImportedRun, candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    return run.candidates.latest_by_id.get(candidate_id)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _timeline_badge(label: str, value: Any) -> dict[str, Any]:
    return {"label": label, "value": value}


def _timeline_field(label: str, value: Any, *, kind: str = "text") -> dict[str, Any]:
    return {"label": label, "value": value, "kind": kind}


def _build_timeline_events(run: ImportedRun) -> list[dict[str, Any]]:
    # Build a lookup: llm_call_id -> call record (for timestamps)
    llm_call_by_id = {c["llm_call_id"]: c for c in run.llm_calls if c.get("llm_call_id")}

    # Build step -> best timestamp (finished_at of last LLM call at that step)
    step_to_timestamp: dict[int, str] = {}
    for call in run.llm_calls:
        step = call.get("step")
        ts = call.get("finished_at") or call.get("started_at")
        if step is not None and ts:
            # Keep the latest timestamp for each step
            existing = step_to_timestamp.get(int(step), "")
            if ts > existing:
                step_to_timestamp[int(step)] = ts

    timeline: list[dict[str, Any]] = []
    for event in run.events:
        built = _build_timeline_event(run, event, llm_call_by_id, step_to_timestamp)
        if built is not None:
            timeline.append(built)

    # Add LLM call events (errors + completions not already covered by candidate events)
    candidate_llm_ids = {e.get("llm_call_id") for e in run.events if e.get("llm_call_id")}
    for call in run.llm_calls:
        call_id = call.get("llm_call_id")
        if call_id in candidate_llm_ids:
            continue  # already represented by candidate events
        status = call.get("status", "")
        if status == "error":
            call_step = call.get("step")
            timeline.append({
                "type": "llm_error",
                "step": call_step,
                "timestamp": call.get("finished_at") or call.get("started_at") or (step_to_timestamp.get(int(call_step)) if call_step is not None else None),
                "title": "LLM Error",
                "summary": str(call.get("error") or "LLM call failed"),
                "tone": "danger",
                "badges": [
                    _timeline_badge("Step", call.get("step")),
                    _timeline_badge("Emitter", call.get("emitter_name")),
                ],
                "fields": [
                    _timeline_field("Step", call.get("step")),
                    _timeline_field("Emitter", call.get("emitter_name")),
                    _timeline_field("Error", call.get("error")),
                    _timeline_field("Latency", f"{call.get('latency_seconds', 0):.1f}s" if call.get("latency_seconds") else None),
                ],
            })

    # Sort by step, with timestamp as tiebreaker
    timeline.sort(key=lambda e: (e.get("step") or 0, e.get("timestamp") or ""))
    return timeline


def _build_timeline_event(run: ImportedRun, event: dict[str, Any], llm_call_by_id: dict | None = None, step_to_timestamp: dict | None = None) -> dict[str, Any] | None:
    event_type = str(event.get("type") or "")
    if event_type in {"worker_task_submitted", "worker_task_completed"}:
        return None

    step = event.get("step")
    candidate_id = event.get("candidate_id")

    # Attach timestamp: 1) from event itself, 2) from LLM call, 3) from step lookup
    timestamp = event.get("timestamp")
    if not timestamp and llm_call_by_id:
        llm_call_id = event.get("llm_call_id")
        if llm_call_id and llm_call_id in llm_call_by_id:
            call = llm_call_by_id[llm_call_id]
            timestamp = call.get("finished_at") or call.get("started_at")
    if not timestamp and step_to_timestamp and step is not None:
        timestamp = step_to_timestamp.get(int(step))

    if event_type in {"candidate_inserted", "candidate_not_inserted"}:
        candidate = _find_candidate(run, candidate_id if isinstance(candidate_id, str) else None)
        parent_ids = candidate.get("parent_ids", []) if isinstance(candidate, dict) else []
        replaced_candidate_id = event.get("replaced_candidate_id")
        accepted = event_type == "candidate_inserted"
        title = f"{'Accepted' if accepted else 'Rejected'} {candidate_id or 'candidate'}"
        summary_parts = [
            f"cell {event.get('cell_id', '-')}",
            f"primary {_format_metric_value(candidate.get('primary_fitness') if isinstance(candidate, dict) else None)}",
        ]
        if replaced_candidate_id:
            summary_parts.append(f"replaced {replaced_candidate_id}")
        # Get LLM latency for this step
        llm_latency = None
        if llm_call_by_id and event.get("llm_call_id") in llm_call_by_id:
            llm_latency = llm_call_by_id[event["llm_call_id"]].get("latency_seconds")
        return {
            **event,
            "timestamp": timestamp,
            "title": title,
            "summary": " · ".join(summary_parts),
            "tone": "success" if accepted else "muted",
            "badges": [
                _timeline_badge("Decision", "Accepted" if accepted else "Rejected"),
                _timeline_badge("Cell", event.get("cell_id")),
                _timeline_badge("Primary", candidate.get("primary_fitness") if isinstance(candidate, dict) else None),
            ],
            "fields": [
                _timeline_field("Candidate", candidate_id, kind="candidate"),
                _timeline_field("Decision", "Accepted" if accepted else "Rejected"),
                _timeline_field("Cell", event.get("cell_id")),
                _timeline_field("Primary Fitness", candidate.get("primary_fitness") if isinstance(candidate, dict) else None),
                _timeline_field("Secondary Fitness", candidate.get("secondary_fitness") if isinstance(candidate, dict) else None),
                _timeline_field("LLM Latency", f"{llm_latency:.1f}s" if llm_latency else None),
                _timeline_field("Emitter", event.get("emitter_name")),
                _timeline_field("LLM Call", event.get("llm_call_id")),
                _timeline_field("Replaced Candidate", replaced_candidate_id, kind="candidate"),
                _timeline_field("Parents", parent_ids, kind="candidate_list"),
            ],
        }

    if event_type == "candidate_rejected":
        return {
            **event,
            "timestamp": timestamp,
            "title": f"Rejected {candidate_id or 'candidate'}",
            "summary": str(event.get("reason") or "failed validation"),
            "tone": "danger",
            "badges": [
                _timeline_badge("Step", step),
                _timeline_badge("Reason", event.get("reason")),
            ],
            "fields": [
                _timeline_field("Candidate", candidate_id, kind="candidate"),
                _timeline_field("Reason", event.get("reason")),
                _timeline_field("Emitter", event.get("emitter_name")),
            ],
        }

    if event_type == "candidate_pruned":
        return {
            **event,
            "timestamp": timestamp,
            "title": f"Pruned {candidate_id or 'candidate'}",
            "summary": str(event.get("reason") or "candidate removed from archive"),
            "tone": "warning",
            "badges": [
                _timeline_badge("Reason", event.get("reason")),
                _timeline_badge("Candidate", candidate_id),
            ],
            "fields": [
                _timeline_field("Candidate", candidate_id, kind="candidate"),
                _timeline_field("Reason", event.get("reason")),
            ],
        }

    if event_type == "remap":
        lost_candidate_ids = event.get("lost_candidate_ids") if isinstance(event.get("lost_candidate_ids"), list) else []
        cell_changes = event.get("cell_changes") if isinstance(event.get("cell_changes"), dict) else {}
        return {
            **event,
            "timestamp": timestamp,
            "title": "Archive Remap",
            "summary": (
                f"{event.get('before_occupied_cells', '-')} -> {event.get('after_occupied_cells', '-')} occupied"
                f" · moved {event.get('moved_count', '-')}"
                f" · collisions {event.get('collision_count', '-')}"
                f" · lost {len(lost_candidate_ids)}"
            ),
            "tone": "info",
            "badges": [
                _timeline_badge("Occupied", f"{event.get('before_occupied_cells', '-')} -> {event.get('after_occupied_cells', '-')}"),
                _timeline_badge("Moved", event.get("moved_count")),
                _timeline_badge("Lost", len(lost_candidate_ids)),
            ],
            "fields": [
                _timeline_field("Occupied Cells Before", event.get("before_occupied_cells")),
                _timeline_field("Occupied Cells After", event.get("after_occupied_cells")),
                _timeline_field("Moved Candidates", event.get("moved_count")),
                _timeline_field("Collisions", event.get("collision_count")),
                _timeline_field("Lost Candidates", lost_candidate_ids, kind="candidate_list"),
                _timeline_field("Changed Cells", len(cell_changes)),
                _timeline_field("Cell Changes", cell_changes, kind="json"),
            ],
        }

    if event_type == "secondary_eval":
        pruned_candidate_ids = event.get("pruned_candidate_ids") if isinstance(event.get("pruned_candidate_ids"), list) else []
        return {
            **event,
            "timestamp": timestamp,
            "title": "Secondary Evaluation",
            "summary": (
                f"evaluated {event.get('count', '-')}"
                f" · pruned {event.get('pruned_count', len(pruned_candidate_ids))}"
                f" · best secondary {_format_metric_value(event.get('best_secondary_fitness_after'))}"
            ),
            "tone": "success",
            "badges": [
                _timeline_badge("Evaluated", event.get("count")),
                _timeline_badge("Pruned", event.get("pruned_count", len(pruned_candidate_ids))),
                _timeline_badge("Best Secondary", event.get("best_secondary_fitness_after")),
            ],
            "fields": [
                _timeline_field("Evaluated", event.get("count")),
                _timeline_field("Pruned", event.get("pruned_count", len(pruned_candidate_ids))),
                _timeline_field("Pruned Candidates", pruned_candidate_ids, kind="candidate_list"),
                _timeline_field("Archive Occupancy After", event.get("archive_occupancy_after")),
                _timeline_field("Best Primary After", event.get("best_primary_fitness_after")),
                _timeline_field("Best Secondary After", event.get("best_secondary_fitness_after")),
                _timeline_field("Secondary Eval Total After", event.get("secondary_eval_total_after")),
            ],
        }

    if event_type == "checkpoint_saved":
        return {
            **event,
            "timestamp": timestamp,
            "title": "Checkpoint Saved",
            "summary": "state and archive snapshot written",
            "tone": "warning",
            "badges": [_timeline_badge("Step", step)],
            "fields": [_timeline_field("Step", step)],
        }

    if event_type == "stage_start":
        stage_num = event.get("stage")
        return {
            **event,
            "timestamp": timestamp,
            "title": f"Stage {stage_num} Started",
            "summary": f"beginning evolution stage {stage_num}",
            "tone": "info",
            "badges": [_timeline_badge("Stage", stage_num), _timeline_badge("Step", step)],
            "fields": [_timeline_field("Stage", stage_num), _timeline_field("Step", step)],
        }

    if event_type == "stage_end":
        stage_num = event.get("stage")
        best_fit = event.get("best_primary_fitness")
        return {
            **event,
            "timestamp": timestamp,
            "title": f"Stage {stage_num} Complete",
            "summary": f"best primary {_format_metric_value(best_fit)} · {event.get('archive_occupancy', '-')} cells",
            "tone": "success",
            "badges": [
                _timeline_badge("Stage", stage_num),
                _timeline_badge("Best Primary", best_fit),
                _timeline_badge("Cells", event.get("archive_occupancy")),
            ],
            "fields": [
                _timeline_field("Stage", stage_num),
                _timeline_field("Best Primary Fitness", best_fit),
                _timeline_field("Archive Occupancy", event.get("archive_occupancy")),
                _timeline_field("Seeds Selected", event.get("seeds_selected")),
            ],
        }

    if event_type == "parse_failure":
        return {
            **event,
            "timestamp": timestamp,
            "title": "Parse Failure",
            "summary": str(event.get("error") or "candidate parser failed"),
            "tone": "danger",
            "badges": [_timeline_badge("Step", step)],
            "fields": [
                _timeline_field("Step", step),
                _timeline_field("Error", event.get("error")),
                _timeline_field("Emitter", event.get("emitter_name")),
            ],
        }

    if event_type == "llm_error":
        return {
            **event,
            "timestamp": timestamp,
            "title": "LLM Error",
            "summary": str(event.get("error") or "LLM call failed"),
            "tone": "danger",
            "badges": [_timeline_badge("Step", step), _timeline_badge("Emitter", event.get("emitter_name"))],
            "fields": [
                _timeline_field("Step", step),
                _timeline_field("Emitter", event.get("emitter_name")),
                _timeline_field("Error", event.get("error")),
            ],
        }

    return None


def _enrich_secondary_events(run: ImportedRun) -> list[dict[str, Any]]:
    stats_by_step = {
        int(record.get("step", -1)): record
        for record in run.stats
        if record.get("step") is not None
    }
    events_by_step: dict[int, list[dict[str, Any]]] = {}
    for event in run.events:
        if event.get("step") is None:
            continue
        step = int(event["step"])
        events_by_step.setdefault(step, []).append(event)
    enriched: list[dict[str, Any]] = []
    for event in run.events:
        if event.get("type") != "secondary_eval":
            continue
        step = int(event.get("step", -1))
        sibling_events = events_by_step.get(step, [])
        pruned_candidate_ids = [
            sibling.get("candidate_id")
            for sibling in sibling_events
            if sibling.get("type") == "candidate_pruned" and sibling.get("candidate_id")
        ]
        step_stats = stats_by_step.get(step, {})
        enriched.append(
            {
                **event,
                "pruned_candidate_ids": pruned_candidate_ids,
                "pruned_count": len(pruned_candidate_ids),
                "archive_occupancy_after": step_stats.get("archive_occupancy"),
                "best_primary_fitness_after": step_stats.get("best_primary_fitness"),
                "best_secondary_fitness_after": step_stats.get("best_secondary_fitness"),
                "secondary_eval_total_after": step_stats.get("secondary_eval_count"),
            }
        )
    return enriched


def _feature_extremes_by_feature(run: ImportedRun, latest_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    occupied = latest_snapshot.get("occupied_cells", [])
    entries: list[dict[str, Any]] = []
    for index, label in enumerate(run.descriptor_labels):
        min_candidate: dict[str, Any] | None = None
        max_candidate: dict[str, Any] | None = None
        min_value: float | None = None
        max_value: float | None = None
        for cell in occupied:
            raw = cell.get("descriptor_raw")
            if not isinstance(raw, list) or len(raw) <= index or raw[index] is None:
                continue
            value = float(raw[index])
            candidate_id = cell.get("candidate_id")
            if not candidate_id:
                continue
            candidate = run.candidates.latest_by_id.get(candidate_id)
            if candidate is None:
                continue
            if min_value is None or value < min_value:
                min_candidate = candidate
                min_value = value
            if max_value is None or value > max_value:
                max_candidate = candidate
                max_value = value
        if min_candidate is None or max_candidate is None or min_value is None or max_value is None:
            entries.append(
                {
                    "label": label,
                    "min": {
                        "candidate_id": None,
                        "feature_value": None,
                        "primary_fitness": None,
                        "secondary_fitness": None,
                        "cell_id": None,
                    },
                    "max": {
                        "candidate_id": None,
                        "feature_value": None,
                        "primary_fitness": None,
                        "secondary_fitness": None,
                        "cell_id": None,
                    },
                }
            )
            continue
        entries.append(
            {
                "label": label,
                "min": {
                    "candidate_id": min_candidate.get("id"),
                    "feature_value": min_value,
                    "primary_fitness": min_candidate.get("primary_fitness"),
                    "secondary_fitness": min_candidate.get("secondary_fitness"),
                    "cell_id": min_candidate.get("cell_id"),
                },
                "max": {
                    "candidate_id": max_candidate.get("id"),
                    "feature_value": max_value,
                    "primary_fitness": max_candidate.get("primary_fitness"),
                    "secondary_fitness": max_candidate.get("secondary_fitness"),
                    "cell_id": max_candidate.get("cell_id"),
                },
            }
        )
    return entries


def _record_timeline(run: ImportedRun) -> dict[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = [
        candidate
        for history in run.candidates.history_by_id.values()
        for candidate in history
        if candidate.get("created_at_step") is not None
    ]
    records.sort(
        key=lambda candidate: (
            int(candidate.get("created_at_step", -1)),
            str(candidate.get("id", "")),
            int(candidate.get("record_version", 1)),
        )
    )

    primary_records: list[dict[str, Any]] = []
    primary_validation_records: list[dict[str, Any]] = []
    secondary_records: list[dict[str, Any]] = []
    secondary_validation_records: list[dict[str, Any]] = []
    best_primary: float | None = None
    best_primary_validation: float | None = None
    best_secondary: float | None = None
    best_secondary_validation: float | None = None

    for candidate in records:
        candidate_id = candidate.get("id")
        step = candidate.get("created_at_step")
        primary = candidate.get("primary_fitness")
        primary_validation = candidate.get("primary_validation_fitness")
        secondary = candidate.get("secondary_fitness")
        secondary_validation = candidate.get("secondary_validation_fitness")

        if candidate_id is None or step is None:
            continue

        if primary is not None:
            primary_value = float(primary)
            if best_primary is None or primary_value > best_primary:
                best_primary = primary_value
                primary_records.append(
                    {
                        "candidate_id": candidate_id,
                        "created_at_step": int(step),
                        "fitness": primary_value,
                    }
                )

        if primary_validation is not None:
            primary_validation_value = float(primary_validation)
            if best_primary_validation is None or primary_validation_value > best_primary_validation:
                best_primary_validation = primary_validation_value
                primary_validation_records.append(
                    {
                        "candidate_id": candidate_id,
                        "created_at_step": int(step),
                        "fitness": primary_validation_value,
                    }
                )

        if secondary is not None:
            secondary_value = float(secondary)
            if best_secondary is None or secondary_value > best_secondary:
                best_secondary = secondary_value
                secondary_records.append(
                    {
                        "candidate_id": candidate_id,
                        "created_at_step": int(step),
                        "fitness": secondary_value,
                    }
                )

        if secondary_validation is not None:
            secondary_validation_value = float(secondary_validation)
            if best_secondary_validation is None or secondary_validation_value > best_secondary_validation:
                best_secondary_validation = secondary_validation_value
                secondary_validation_records.append(
                    {
                        "candidate_id": candidate_id,
                        "created_at_step": int(step),
                        "fitness": secondary_validation_value,
                    }
                )

    return {
        "primary": primary_records,
        "primary_validation": primary_validation_records,
        "secondary": secondary_records,
        "secondary_validation": secondary_validation_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Adaptive CVT-MAP-Elites web viewer.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("viewer.backend.main:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
