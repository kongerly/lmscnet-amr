from __future__ import annotations

import pytest

from na_lmscnet.training.progress import ProgressReporter


def _epoch_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "epoch": 12,
        "max_epochs": 100,
        "learning_rate": 0.0003,
        "train_loss": 1.2345,
        "train_samples": 154000,
        "validation_loss": 1.1,
        "validation": {"accuracy": 0.5232, "macro_f1": 0.522},
    }
    record.update(overrides)
    return record


def test_epoch_line_is_parseable_and_readable(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = ProgressReporter(interactive=False)
    reporter.on_epoch(_epoch_record(), run_id="lr-0p0003_dropout-0_seed-13")
    line = capsys.readouterr().out.strip()
    assert line.startswith("epoch 12/100")
    assert "run_id=lr-0p0003_dropout-0_seed-13" in line
    assert "lr=3.000e-04" in line
    assert "val_macro_f1=0.5220" in line


def test_batch_callback_is_noop_when_not_interactive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reporter = ProgressReporter(interactive=False)
    reporter.on_batch(
        {
            "event": "batch_complete",
            "epoch": 1,
            "batch": 1,
            "total_batches": 5,
            "max_epochs": 100,
            "train_loss": 2.0,
        }
    )
    reporter.finish()
    assert capsys.readouterr().out == ""


def test_batch_callback_updates_bar_when_interactive() -> None:
    reporter = ProgressReporter(interactive=True)
    reporter.on_batch(
        {
            "event": "batch_complete",
            "epoch": 1,
            "batch": 1,
            "total_batches": 5,
            "max_epochs": 100,
            "train_loss": 2.0,
        }
    )
    reporter.on_epoch(_epoch_record())
    reporter.finish()
