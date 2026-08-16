"""Run Phase R6 fixed-epoch validation contrasts without replacing Phase R2."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_r2_primary_contrasts import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LOW_SNR_VALUES,
    PERMUTATION_SEEDS,
    _alignment_seed,
    _contrast_row,
    _mean_accuracy,
    _replays_by_seed,
    _shuffled_contrast_row,
    _shuffled_replays_by_permutation,
)


class R6ContrastError(ValueError):
    """Raised when the R6 contrast evidence is incomplete."""


def _build_report(historical: Path, intervention: Path, model_replay: Path) -> dict[str, object]:
    s2_aligned = _replays_by_seed(intervention, "lmscnet_s2_aligned")
    s2_mean = _replays_by_seed(intervention, "lmscnet_s2_mean")
    s2_shuffled_by_perm = _shuffled_replays_by_permutation(intervention)
    s1_static = _replays_by_seed(model_replay, "lmscnet_s1_static")
    s1_wide_static = _replays_by_seed(model_replay, "lmscnet_s1_wide_static")
    sknet = _replays_by_seed(model_replay, "sknet_1d_adaptation")
    afnet = _replays_by_seed(model_replay, "afnet_adaptation")
    s1_equal_historical = _replays_by_seed(historical, "lmscnet_s1")

    for name, variant in (
        ("s2_mean", s2_mean),
        ("s1_static", s1_static),
        ("s1_wide_static", s1_wide_static),
        ("sknet_1d_adaptation", sknet),
        ("afnet_adaptation", afnet),
        ("s1_equal_historical", s1_equal_historical),
    ):
        try:
            _alignment_seed(s2_aligned, variant)
        except ValueError as error:
            raise R6ContrastError(f"Replay alignment failed for {name}: {error}") from error

    contrasts = [
        _contrast_row("R6_C1_s2_aligned_vs_s1_static", s2_aligned, s1_static),
        _contrast_row("R6_C2a_s2_aligned_vs_s2_mean", s2_aligned, s2_mean),
        _shuffled_contrast_row("R6_C2b_s2_aligned_vs_s2_shuffled", s2_aligned, s2_shuffled_by_perm),
        _contrast_row("R6_C3_s2_aligned_vs_s1_wide_static", s2_aligned, s1_wide_static),
        _contrast_row("R6_C4_s2_aligned_vs_sknet", s2_aligned, sknet),
        _contrast_row("R6_C4_s2_aligned_vs_afnet", s2_aligned, afnet),
        _contrast_row("R6_background_s2_aligned_vs_s1_equal_historical", s2_aligned, s1_equal_historical),
    ]
    accuracy_rows: list[dict[str, object]] = []
    for name, replays in (
        ("lmscnet_s2_aligned", s2_aligned),
        ("lmscnet_s2_mean", s2_mean),
        ("lmscnet_s1_static", s1_static),
        ("lmscnet_s1_wide_static", s1_wide_static),
        ("sknet_1d_adaptation", sknet),
        ("afnet_adaptation", afnet),
        ("lmscnet_s1_equal_historical", s1_equal_historical),
    ):
        accuracy_rows.append(
            {
                "model": name,
                "overall_accuracy_mean": _mean_accuracy(replays),
                "low_snr_accuracy_mean": _mean_accuracy(replays, LOW_SNR_VALUES),
            }
        )
    shuffled = {"overall": [], "low_snr": []}
    for permutation_seed in PERMUTATION_SEEDS:
        replays = s2_shuffled_by_perm[permutation_seed]
        shuffled["overall"].append(_mean_accuracy(replays))
        shuffled["low_snr"].append(_mean_accuracy(replays, LOW_SNR_VALUES))
    accuracy_rows.append(
        {
            "model": "lmscnet_s2_shuffled",
            "overall_accuracy_mean": float(statistics.fmean(shuffled["overall"])),
            "low_snr_accuracy_mean": float(statistics.fmean(shuffled["low_snr"])),
        }
    )
    return {
        "schema_version": 1,
        "purpose": "phase_r6_fixed_epoch_validation_contrast_analysis",
        "test_accessed": False,
        "selection_metric": "fixed_epoch",
        "checkpoint_epoch": 100,
        "validation_role": "post_training_assessment_only",
        "does_not_replace_phase_r2_gatekeeping": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "stratification": ["seed", "modulation", "snr_db"],
        "low_snr_values_db": list(LOW_SNR_VALUES),
        "permutation_seeds": PERMUTATION_SEEDS,
        "reference_model": "lmscnet_s2_aligned",
        "contrasts": contrasts,
        "accuracy_rows": accuracy_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--intervention-replay", type=Path, required=True)
    parser.add_argument("--model-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    historical = args.historical_predictions.resolve(strict=True)
    intervention = args.intervention_replay.resolve(strict=True)
    model_replay = args.model_replay.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise R6ContrastError("Output must remain outside the repository")
    report = _build_report(historical, intervention, model_replay)
    output_dir.mkdir(parents=True)
    (output_dir / "r6-fixed-epoch-contrasts.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = report["accuracy_rows"]
    if not isinstance(rows, list) or not rows:
        raise R6ContrastError("Accuracy rows are empty")
    with (output_dir / "r6-fixed-epoch-contrast-accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output_dir), "contrast_count": 7, "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
