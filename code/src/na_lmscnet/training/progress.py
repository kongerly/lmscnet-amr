"""Human-readable terminal progress rendering for training CLIs."""

from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm


class ProgressReporter:
    """Render a per-batch tqdm bar and per-epoch summary lines to the terminal.

    The per-batch bar is only drawn when stdout is an interactive terminal so that
    redirected logs stay free of carriage-return noise. The per-epoch line is always
    printed and uses parseable ``key=value`` fields.
    """

    def __init__(self, *, interactive: bool | None = None) -> None:
        self.interactive = sys.stdout.isatty() if interactive is None else interactive
        self._bar: tqdm[Any] | None = None
        self._bar_epoch = 0

    def on_batch(self, record: dict[str, object]) -> None:
        if not self.interactive:
            return
        epoch = int(record["epoch"])
        if self._bar is None or self._bar_epoch != epoch:
            self._close_bar()
            self._bar_epoch = epoch
            self._bar = tqdm(
                total=int(record["total_batches"]),
                desc=f"epoch {epoch}/{int(record['max_epochs'])}",
                unit="batch",
                leave=False,
            )
        assert self._bar is not None
        self._bar.update(1)
        self._bar.set_postfix(train_loss=f"{float(record['train_loss']):.4f}")

    def on_epoch(self, record: dict[str, object], *, run_id: str | None = None) -> None:
        self._close_bar()
        validation = record.get("validation")
        validation_mapping = validation if isinstance(validation, dict) else {}
        parts = [f"epoch {int(record['epoch'])}/{int(record['max_epochs'])}"]
        if run_id is not None:
            parts.append(f"run_id={run_id}")
        parts.append(f"lr={float(record['learning_rate']):.3e}")
        parts.append(f"train_loss={float(record['train_loss']):.4f}")
        parts.append(f"val_loss={float(record['validation_loss']):.4f}")
        parts.append(f"val_acc={float(validation_mapping.get('accuracy', float('nan'))):.4f}")
        parts.append(f"val_macro_f1={float(validation_mapping.get('macro_f1', float('nan'))):.4f}")
        print(" | ".join(parts), flush=True)

    def finish(self) -> None:
        self._close_bar()

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
