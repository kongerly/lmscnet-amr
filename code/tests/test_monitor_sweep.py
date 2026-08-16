from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code/scripts/monitor_sweep.py"


def _load_monitor_module() -> object:
    spec = importlib.util.spec_from_file_location("monitor_sweep_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_epoch_counts_parse_json_and_human_lines(tmp_path: Path) -> None:
    module = _load_monitor_module()
    log = tmp_path / "runner.log"
    log.write_text(
        "\n".join(
            [
                json.dumps({"event": "epoch_complete", "run_id": "a", "epoch": 3}),
                "epoch 5/100 | run_id=b | lr=3.000e-04 | val_macro_f1=0.5000",
                "epoch 7/100 | run_id=a | lr=3.000e-04 | val_macro_f1=0.6000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    counts = module._epoch_counts(log)  # type: ignore[attr-defined]
    assert counts == {"a": 7, "b": 5}
