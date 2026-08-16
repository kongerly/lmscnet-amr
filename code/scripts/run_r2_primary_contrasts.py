"""Phase R2 primary contrast analysis on frozen validation predictions.

Runs the PLAN section 8 gatekeeping contrasts on validation replays:
  C1  S2-aligned vs S1-static
  C2a S2-aligned vs S2-mean
  C2b S2-aligned vs S2-shuffled (mean over the frozen permutation seeds)
  C3  S2-aligned vs S1-wide-static
  C4  S2-aligned vs strongest direct neighbor (SKNet-1D / AFNet adaptation)

All replays must share identical validation sample ordering. The historical
S1/S2 predictions are the frozen five-seed files from the original queue;
the R2 model predictions are replayed from the new queue checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation.core_ablation_multiseed_report import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    paired_hierarchical_bootstrap,
)

SEEDS = (13, 37, 73, 101, 137)
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
PERMUTATION_SEEDS = [13, 37, 73, 101, 137, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511, 1601]


class ContrastError(ValueError):
    """Raised when the R2 contrast evidence is incomplete or misaligned."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_replay(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        replay = {
            "sample_ids": np.asarray(archive["sample_ids"]),
            "predictions": np.asarray(archive["predictions"]),
            "targets": np.asarray(archive["targets"]),
            "modulation": np.asarray(archive["modulation"]),
            "snr_db": np.asarray(archive["snr_db"]),
        }
    return replay


def _replays_by_seed(directory: Path, prefix: str, suffix: str = "") -> dict[int, dict[str, object]]:
    replays: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        filename = f"{prefix}-seed-{seed}{suffix}.npz"
        path = directory / filename
        if not path.is_file():
            raise ContrastError(f"Missing replay file: {path}")
        replays[seed] = _load_replay(path)
    return replays


def _alignment_seed(reference: dict[int, dict[str, object]], variant: dict[int, dict[str, object]]) -> str:
    for seed in SEEDS:
        ref = reference[seed]
        var = variant[seed]
        for field in ("sample_ids", "targets", "modulation", "snr_db"):
            if not np.array_equal(np.asarray(ref[field]), np.asarray(var[field])):
                raise ContrastError(f"Sample alignment differs for seed {seed} field {field}")
    return "aligned"


def _mean_accuracy(replays: dict[int, dict[str, object]], snr_values: list[int] | None = None) -> float:
    values = []
    for seed in SEEDS:
        replay = replays[seed]
        mask = np.isin(replay["snr_db"], snr_values) if snr_values else np.ones(len(replay["targets"]), dtype=bool)
        values.append(float(np.mean((replay["predictions"][mask] == replay["targets"][mask]).astype(np.int8))))
    return statistics.fmean(values)


def _shuffled_replays_by_permutation(directory: Path) -> dict[int, dict[int, dict[str, object]]]:
    """Load all frozen-permutation replays as {permutation_seed: {seed: replay}}."""
    replays: dict[int, dict[int, dict[str, object]]] = {}
    first_replay: dict[str, object] | None = None
    for permutation_seed in PERMUTATION_SEEDS:
        per_seed: dict[int, dict[str, object]] = {}
        for seed in SEEDS:
            path = directory / f"lmscnet_s2_shuffled-seed-{seed}-perm-{permutation_seed}.npz"
            if not path.is_file():
                raise ContrastError(f"Missing shuffled replay: {path}")
            replay = _load_replay(path)
            if first_replay is None:
                first_replay = replay
            else:
                for field in ("sample_ids", "targets", "modulation", "snr_db"):
                    if not np.array_equal(np.asarray(first_replay[field]), np.asarray(replay[field])):
                        raise ContrastError(
                            f"Shuffled replay alignment differs for perm {permutation_seed} "
                            f"seed {seed} field {field}"
                        )
            per_seed[seed] = replay
        replays[permutation_seed] = per_seed
    return replays


def _observed_difference(
    reference: dict[int, dict[str, object]], variant: dict[int, dict[str, object]], snr_values: list[int] | None
) -> float:
    differences = []
    for seed in SEEDS:
        ref = np.asarray(reference[seed]["predictions"])
        var = np.asarray(variant[seed]["predictions"])
        targets = np.asarray(reference[seed]["targets"])
        if snr_values is not None:
            mask = np.isin(np.asarray(reference[seed]["snr_db"]), snr_values)
            ref, var, targets = ref[mask], var[mask], targets[mask]
        differences.append(
            float(np.mean((ref == targets).astype(np.int8) - (var == targets).astype(np.int8)))
        )
    return statistics.fmean(differences)


def _shuffled_contrast_row(
    name: str, reference: dict[int, dict[str, object]], shuffled: dict[int, dict[int, dict[str, object]]]
) -> dict[str, object]:
    """Aggregate the 20 frozen permutations: report mean and percentile interval."""
    per_permutation: dict[str, dict[str, object]] = {}
    aggregate: dict[str, object] = {}
    for scope, snr_values in (("overall", None), ("low_snr", LOW_SNR_VALUES)):
        values = [
            _observed_difference(reference, shuffled[permutation_seed], snr_values)
            for permutation_seed in PERMUTATION_SEEDS
        ]
        per_permutation[scope] = {"values": values}
        aggregate[scope] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
            "p2_5": float(np.percentile(values, 2.5)),
            "p97_5": float(np.percentile(values, 97.5)),
            "positive_permutation_count": int(sum(value > 0 for value in values)),
        }
    return {
        "contrast": name,
        "reference_model": "lmscnet_s2_aligned",
        "method": (
            "per-permutation five-seed mean difference, aggregated over 20 frozen permutations; "
            "percentile interval is across permutations, not a bootstrap CI"
        ),
        "permutation_seed_count": len(PERMUTATION_SEEDS),
        "aggregate": aggregate,
        "per_permutation": per_permutation,
    }


def _contrast_row(
    name: str, reference: dict[int, dict[str, object]], variant: dict[int, dict[str, object]]
) -> dict[str, object]:
    rows = []
    for metric in ("accuracy", "macro_f1"):
        bootstrap = paired_hierarchical_bootstrap(
            reference_replays=reference,
            variant_replays=variant,
            metric=metric,
            snr_values=None,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        low_snr_bootstrap = paired_hierarchical_bootstrap(
            reference_replays=reference,
            variant_replays=variant,
            metric=metric,
            snr_values=LOW_SNR_VALUES,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        rows.append(
            {
                "metric": metric,
                "overall": bootstrap,
                "low_snr": low_snr_bootstrap,
            }
        )
    seed_differences = []
    for seed in SEEDS:
        reference_accuracy = float(
            np.mean((np.asarray(reference[seed]["predictions"]) == np.asarray(reference[seed]["targets"])).astype(np.int8))
        )
        variant_accuracy = float(
            np.mean((np.asarray(variant[seed]["predictions"]) == np.asarray(variant[seed]["targets"])).astype(np.int8))
        )
        seed_differences.append(float(reference_accuracy - variant_accuracy))
    return {
        "contrast": name,
        "reference_model": "lmscnet_s2_aligned",
        "seed_differences": seed_differences,
        "positive_seed_count": int(sum(difference > 0 for difference in seed_differences)),
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--r2-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    historical = args.historical_predictions.resolve(strict=True)
    replay_dir = args.replay_dir.resolve(strict=True)
    r2_predictions = args.r2_predictions.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise ContrastError("Output must remain outside the repository")

    s2_aligned = _replays_by_seed(replay_dir, "lmscnet_s2_aligned")
    s2_mean = _replays_by_seed(replay_dir, "lmscnet_s2_mean")
    s2_shuffled_by_perm = _shuffled_replays_by_permutation(replay_dir)
    s1_static = _replays_by_seed(r2_predictions, "lmscnet_s1_static")
    s1_wide_static = _replays_by_seed(r2_predictions, "lmscnet_s1_wide_static")
    sknet = _replays_by_seed(r2_predictions, "sknet_1d_adaptation")
    afnet = _replays_by_seed(r2_predictions, "afnet_adaptation")
    s1_equal_historical = _replays_by_seed(historical, "lmscnet_s1")

    for _name, variant in (
        ("s2_mean", s2_mean),
        ("s1_static", s1_static),
        ("s1_wide_static", s1_wide_static),
        ("sknet_1d_adaptation", sknet),
        ("afnet_adaptation", afnet),
        ("s1_equal", s1_equal_historical),
    ):
        _alignment_seed(s2_aligned, variant)

    contrasts = [
        _contrast_row("C1_s2_aligned_vs_s1_static", s2_aligned, s1_static),
        _contrast_row("C2a_s2_aligned_vs_s2_mean", s2_aligned, s2_mean),
        _shuffled_contrast_row("C2b_s2_aligned_vs_s2_shuffled", s2_aligned, s2_shuffled_by_perm),
        _contrast_row("C3_s2_aligned_vs_s1_wide_static", s2_aligned, s1_wide_static),
        _contrast_row("C4_s2_aligned_vs_sknet", s2_aligned, sknet),
        _contrast_row("C4_s2_aligned_vs_afnet", s2_aligned, afnet),
        _contrast_row("background_s2_aligned_vs_s1_equal", s2_aligned, s1_equal_historical),
    ]

    accuracy_rows = []
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
    shuffled_accuracies: dict[str, list[float]] = {"overall": [], "low_snr": []}
    for permutation_seed in PERMUTATION_SEEDS:
        replays = s2_shuffled_by_perm[permutation_seed]
        shuffled_accuracies["overall"].append(_mean_accuracy(replays))
        shuffled_accuracies["low_snr"].append(_mean_accuracy(replays, LOW_SNR_VALUES))
    accuracy_rows.append(
        {
            "model": "lmscnet_s2_shuffled",
            "overall_accuracy_mean": float(statistics.fmean(shuffled_accuracies["overall"])),
            "low_snr_accuracy_mean": float(statistics.fmean(shuffled_accuracies["low_snr"])),
        }
    )

    report = {
        "schema_version": 1,
        "purpose": "phase_r2_primary_contrast_analysis",
        "test_accessed": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "stratification": ["seed", "modulation", "snr_db"],
        "low_snr_values_db": list(LOW_SNR_VALUES),
        "permutation_seeds": PERMUTATION_SEEDS,
        "reference_model": "lmscnet_s2_aligned",
        "contrasts": contrasts,
        "accuracy_rows": accuracy_rows,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "r2-primary-contrasts.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "r2-contrast-accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accuracy_rows[0]))
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(json.dumps({"output_dir": str(output_dir), "contrast_count": len(contrasts), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
