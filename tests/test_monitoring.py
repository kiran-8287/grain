"""
Tests for monitoring log parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monitoring.log_parser import (
    AnalysisRecord,
    LogRecord,
    ParseResult,
    RequestRecord,
    StageTiming,
    _parse_line,
    compute_analysis_stats,
    compute_endpoint_stats,
    compute_error_summary,
    compute_stage_timings,
    get_last_log_time,
    get_memory_metrics,
    parse_file,
    parse_logs,
    reconstruct_analyses,
    reconstruct_requests,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "monitoring_sample.log"


def test_parse_line_valid() -> None:
    line = '{"timestamp": "2026-10-07T12:00:00.000Z", "event": "analysis_started"}'
    rec = _parse_line(line)
    assert not rec.parse_error
    assert rec.event == "analysis_started"


def test_parse_line_malformed() -> None:
    rec = _parse_line("not valid json {{{")
    assert rec.parse_error
    assert rec.timestamp_str == ""


def test_parse_line_empty() -> None:
    rec = _parse_line("")
    assert rec.parse_error


def test_parse_fixture_file() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    assert result.malformed_lines >= 1
    assert len(result.records) >= 1


def test_reconstruct_requests_counts_once() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    finished = [r for r in requests if r.finished_at is not None or r.error_type]
    assert len(finished) >= 1


def test_reconstruct_analysis_single_job() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    analyses = reconstruct_analyses(result.records)
    job_ids = [a.job_id for a in analyses if a.job_id]
    assert len(set(job_ids)) == len(job_ids)


def test_request_id_linked_to_job_id() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    linked = [r for r in requests if r.job_id]
    assert len(linked) >= 1
    assert linked[0].job_id == "job-001"


def test_failed_analysis_produces_failed_result() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    analyses = reconstruct_analyses(result.records)
    failed = [a for a in analyses if a.status == "failed"]
    assert len(failed) >= 1
    assert failed[0].failed_stage == "segmentation"


def test_duplicate_lifecycle_events_not_double_counted() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    assert len(requests) >= 4


def test_duration_calculation_from_explicit_ms() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    req = next((r for r in requests if r.request_id == "req-001"), None)
    assert req is not None
    assert req.duration_ms == 1120


def test_endpoint_aggregation() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    stats = compute_endpoint_stats(requests)
    routes = {s["route"]: s for s in stats}
    assert "/analyze" in routes
    assert "/health" in routes


def test_failure_aggregation() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    es = compute_error_summary(result.records)
    assert es["total_errors"] >= 1


def test_stage_timing_aggregation() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    timings = compute_stage_timings(result.records)
    assert len(timings) >= 1


def test_last_log_time_found() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    last = get_last_log_time(result.records)
    assert last is not None


def test_memory_metrics_not_available() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    mem = get_memory_metrics(result.records)
    assert not mem["available"]


def test_analysis_stats() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    analyses = reconstruct_analyses(result.records)
    stats = compute_analysis_stats(analyses)
    assert stats["total_analyses"] >= 1
    assert stats["successful_analyses"] >= 1
    assert stats["failed_analyses"] >= 1


def test_request_started_finished_equals_one_request() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    requests = reconstruct_requests(result.records)
    completed = [r for r in requests if r.finished_at is not None]
    assert len(completed) >= 4


def test_malformed_line_does_not_crash_parser() -> None:
    lines = [
        '{"timestamp": "2026-10-07T12:00:00.000Z", "event": "analysis_started"}',
        "MALFORMED!!!",
        '{"timestamp": "2026-10-07T12:00:01.000Z", "event": "analysis_completed"}',
    ]
    records: list[Any] = []
    for line in lines:
        rec = _parse_line(line)
        if not rec.parse_error:
            records.append(rec)
    assert len(records) == 2


def test_sorted_chronologically() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    timestamps = [r.timestamp for r in result.records if r.timestamp is not None]
    assert timestamps == sorted(timestamps)


def test_analysis_stats_reflects_success_and_failure() -> None:
    result = parse_file(FIXTURE_PATH, max_lines=100_000)
    analyses = reconstruct_analyses(result.records)
    stats = compute_analysis_stats(analyses)
    assert stats["successful_analyses"] >= 1
    assert stats["failed_analyses"] >= 1
    assert stats["total_analyses"] == stats["successful_analyses"] + stats["failed_analyses"]
