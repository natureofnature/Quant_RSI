"""Read-only history views share the ledger; they never make a running study fail."""

import multiprocessing
import threading
import time

import pytest

from quant_rsi.research.study_history import _ledger, history_view


def _poll(directory, stop):
    # What the live UI and workbench do while a study runs, as a tight loop.
    while not stop.is_set():
        history_view(directory, "KO")


@pytest.fixture
def ledger(tmp_path):
    directory = tmp_path / "history"
    with _ledger(directory, write=True):
        pass
    return directory


def test_polling_viewer_never_fails_a_study_writer(ledger):
    context = multiprocessing.get_context("fork")
    stop = context.Event()
    viewer = context.Process(target=_poll, args=(ledger, stop))
    viewer.start()
    try:
        time.sleep(0.2)
        for _ in range(100):
            # register_study and observe_prices take this exclusive access.
            with _ledger(ledger, write=True):
                pass
    finally:
        stop.set()
        viewer.join(timeout=10)
    assert viewer.exitcode == 0


def test_readers_share_the_ledger(ledger):
    with _ledger(ledger):
        with _ledger(ledger, wait_seconds=0.2) as (journal, _):
            assert journal.connection.execute("SELECT 1").fetchone()[0] == 1


def test_writer_blocks_readers_for_a_bounded_time_only(ledger):
    with _ledger(ledger, write=True):
        started = time.monotonic()
        with pytest.raises(ValueError, match="another controller"):
            with _ledger(ledger, wait_seconds=0.2):
                pass
        assert time.monotonic() - started < 5


def test_writer_waits_for_a_short_read_instead_of_failing(ledger):
    reading, release = threading.Event(), threading.Event()

    def reader():
        with _ledger(ledger):
            reading.set()
            release.wait(5)

    thread = threading.Thread(target=reader)
    thread.start()
    reading.wait(5)
    threading.Timer(0.3, release.set).start()
    try:
        with _ledger(ledger, write=True, wait_seconds=5):
            assert release.is_set()
    finally:
        release.set()
        thread.join(5)
