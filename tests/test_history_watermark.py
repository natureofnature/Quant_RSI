"""A run reads cross-run history through one recorded watermark (#159 D8.2).

Snapshot plus overlay: a study registered after the run's snapshot, by another process
or before a resume, must stay invisible to that run's cross-run reads.
"""

import json
from pathlib import Path

from test_study_history import prices, result_for

from quant_rsi.research import study_history
from quant_rsi.research.study_resume import StudyCache


def two_studies(tmp_path, monkeypatch):
    _, snapshot = prices(tmp_path, monkeypatch, "2026-10-23")
    history = tmp_path / "history"
    first = study_history.register_study(
        history, tmp_path / "first", result_for(snapshot), snapshot
    )
    return history, snapshot, first


def register_later(history, tmp_path, snapshot):
    return study_history.register_study(history, tmp_path / "later", result_for(snapshot), snapshot)


def ids(view):
    return [entry["study_id"] for entry in view["entries"]]


def test_reads_through_the_watermark_exclude_later_studies(tmp_path, monkeypatch):
    history, snapshot, first = two_studies(tmp_path, monkeypatch)
    watermark = study_history.history_watermark(history)
    later = register_later(history, tmp_path, snapshot)
    assert ids(study_history.history_view(history, "KO")) == [
        first["study_id"],
        later["study_id"],
    ]
    assert ids(study_history.history_view(history, "KO", through=watermark)) == [first["study_id"]]
    feedback = study_history.mature_feedback(history, "KO", through=watermark)
    assert [e["study_id"] for e in feedback["history"]["entries"]] == [first["study_id"]]
    assert [r["study_id"] for r in feedback["factor_memory"]] == [first["study_id"]]
    assert feedback["factor_ledger"]["study_count"] == 1
    assert study_history.factor_ledger(history, "KO")["study_count"] == 2


def test_missing_history_has_watermark_zero(tmp_path):
    assert study_history.history_watermark(tmp_path / "absent") == 0
    assert study_history.history_watermark(None) == 0


def test_run_records_its_watermark_once_and_resume_keeps_it(tmp_path, monkeypatch):
    history, snapshot, first = two_studies(tmp_path, monkeypatch)
    instruments = Path(__file__).resolve().parents[1] / "instruments"
    run = tmp_path / "run"
    configuration = {"symbol": "KO", "question": "", "factor_root": str(instruments)}
    run.mkdir()
    cache = StudyCache(run, configuration)
    recorded = study_history.run_history_watermark(cache, history, "KO")
    later = register_later(history, tmp_path, snapshot)
    # Later in the same run, and after a resume, reads stay at the recorded snapshot.
    assert study_history.run_history_watermark(cache, history, "KO") == recorded
    runtime = {"status": "failed", "counts": {}, "token_usage": {}, "total_elapsed_seconds": 1}
    (run / "runtime.json").write_text(json.dumps(runtime))
    resumed = StudyCache(run, configuration, resume=True)
    through = study_history.run_history_watermark(resumed, history, "KO")
    assert through == recorded < study_history.history_watermark(history)
    visible = ids(study_history.history_view(history, "KO", through=through))
    assert visible == [first["study_id"]] and later["study_id"] not in visible
