from datetime import datetime, timedelta, timezone

import pytest

from scripts.monitor_suite_progress import add_eta


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
