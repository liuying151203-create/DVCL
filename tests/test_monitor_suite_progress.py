from datetime import datetime, timedelta, timezone

import pytest

from scripts.monitor_suite_progress import add_eta, estimate_historical_eta


def test_eta_uses_observed_completion_throughput():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [{
        "timestamp": start.isoformat(),
        "physical": {"completed": 40},
    }]
    snapshot = {
        "timestamp": (start + timedelta(minutes=30)).isoformat(),
        "physical": {"completed": 45},
        "remaining": 15,
    }
    result = add_eta(snapshot, history)
    assert result["throughput_per_hour"] == pytest.approx(10)
    assert result["eta_hours"] == pytest.approx(1.5)


def test_eta_is_unknown_without_progress_history():
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "physical": {"completed": 5},
        "remaining": 10,
    }
    result = add_eta(snapshot, [])
    assert result["throughput_per_hour"] is None
    assert result["eta_hours"] is None


def test_eta_falls_back_to_historical_run_durations():
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "physical": {"completed": 5},
        "remaining": 10,
        "historical_eta_hours": 2.25,
    }
    result = add_eta(snapshot, [])
    assert result["throughput_per_hour"] is None
    assert result["eta_hours"] == pytest.approx(2.25)
    assert result["eta_source"] == "historical_run_durations"


def test_partitioned_eta_uses_slowest_attack_queue():
    pending = [("dblp", "adaptive_query")] * 4 + [
        ("dblp", "hg_baseline")
    ] * 8
    durations = {
        ("dblp", "adaptive_query"): [3600],
        ("dblp", "hg_baseline"): [600],
    }
    pooled = estimate_historical_eta(pending, durations, workers=2)
    partitioned = estimate_historical_eta(
        pending, durations, workers=2, partition_by_attack=True
    )
    assert pooled == pytest.approx(9600)
    assert partitioned == pytest.approx(14400)


def test_partitioned_eta_is_not_overridden_by_aggregate_throughput():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [{
        "timestamp": start.isoformat(),
        "physical": {"completed": 40},
    }]
    snapshot = {
        "timestamp": (start + timedelta(minutes=30)).isoformat(),
        "physical": {"completed": 45},
        "remaining": 15,
        "eta_partition": "attack",
        "historical_eta_hours": 4.0,
    }
    result = add_eta(snapshot, history)
    assert result["throughput_per_hour"] == pytest.approx(10)
    assert result["eta_hours"] == pytest.approx(4.0)
    assert result["eta_source"] == "historical_attack_partition"
