"""Generate the two final major-revision figures from frozen R2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAMESPACE = Path(r"D:\Datasets\RadioML\revision-controlled-fusion-r0-20260814-v3")
MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
BRANCHES = ["k=3", "k=7", "k=15"]
COLORS = ["#2374AB", "#E07A5F", "#3D9970"]
INK = "#172033"
MUTED = "#526174"
PAPER = "#FFFFFF"


class FigureGenerationError(ValueError):
    """Raised when figure generation would violate the revision boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_test_isolated(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("test_accessed") is not False:
        raise FigureGenerationError(f"Figure input is not explicitly test-isolated: {path}")
    return value


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def _centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, fill: str) -> None:
    draw.multiline_text(xy, text, font=font, fill=fill, anchor="mm", align="center", spacing=8)


def _box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    color: str,
    *,
    font_size: int = 38,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PAPER, outline=color, width=5)
    _centered(
        draw,
        ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
        text,
        _font(font_size),
        INK,
    )


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = MUTED) -> None:
    draw.line([start, end], fill=color, width=6)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 24
    wing = 12
    points = [
        end,
        (
            int(end[0] - length * math.cos(angle) + wing * math.sin(angle)),
            int(end[1] - length * math.sin(angle) - wing * math.cos(angle)),
        ),
        (
            int(end[0] - length * math.cos(angle) - wing * math.sin(angle)),
            int(end[1] - length * math.sin(angle) + wing * math.cos(angle)),
        ),
    ]
    draw.polygon(points, fill=color)


def _save(image: Image.Image, base_path: Path) -> list[Path]:
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    image.save(png, dpi=(300, 300), optimize=True)
    image.convert("RGB").save(pdf, "PDF", resolution=300.0)
    return [png, pdf]


def _figure_one(output_dir: Path) -> list[Path]:
    image = Image.new("RGB", (3600, 2160), PAPER)
    draw = ImageDraw.Draw(image)
    _centered(draw, (1800, 125), "Controlled evidence design", _font(76, bold=True), INK)
    _centered(
        draw,
        (1800, 225),
        "Shared six-block 1-D backbone | depthwise branches k={3, 7, 15} | common validation protocol",
        _font(34),
        MUTED,
    )
    draw.text((260, 390), "Retrained model contrasts", font=_font(44, bold=True), fill="#0F766E")
    draw.text((2620, 390), "Post-training interventions", font=_font(44, bold=True), fill="#B45309")

    reference = (1390, 715, 2210, 1110)
    _box(
        draw,
        reference,
        "S2-aligned\ninput-conditioned scalar gate\nreference checkpoint",
        "#1D4ED8",
        font_size=46,
    )
    left = [
        ((150, 560, 1040, 805), "S1-equal\nbackground only", "#64748B"),
        ((150, 880, 1040, 1125), "S1-static\nC1: global preference", "#0F766E"),
        ((150, 1200, 1040, 1445), "S1-wide-static\nC3: capacity matched", "#0F766E"),
        ((150, 1520, 1040, 1765), "SKNet-1D / AFNet\nC4: direct neighbors", "#0F766E"),
    ]
    for box, text, color in left:
        _box(draw, box, text, color)
        _arrow(draw, (box[2], (box[1] + box[3]) // 2), (reference[0], 910))

    mean_box = (2560, 670, 3450, 930)
    shuffle_box = (2560, 1050, 3450, 1390)
    _box(draw, mean_box, "S2-mean\nC2a: replace with train means", "#B45309")
    _box(
        draw,
        shuffle_box,
        "S2-shuffled\nC2b: batch-local reassignment\n95.8% same modulation",
        "#B45309",
    )
    _arrow(draw, (reference[2], 850), (mean_box[0], 800), "#B45309")
    _arrow(draw, (reference[2], 980), (shuffle_box[0], 1220), "#B45309")

    boundary = (1325, 1375, 2275, 1815)
    _box(
        draw,
        boundary,
        "Evidence boundary\nC1 unresolved\nC3 excludes capacity-only account\nC2 supports frozen-checkpoint dependence\nC4 unresolved",
        "#7C3AED",
        font_size=40,
    )
    _arrow(draw, (1800, reference[3]), (1800, boundary[1]), "#7C3AED")
    _centered(
        draw,
        (1800, 2030),
        "Arrows separate estimands; they are not an ordered proof chain. No new test evidence is used.",
        _font(34),
        MUTED,
    )
    return _save(image, output_dir / "figure1-controlled-evidence-design")


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    draw.rounded_rectangle(box, radius=16, fill="#FBFCFE", outline="#CBD5E1", width=3)
    draw.text((box[0] + 28, box[1] + 22), title, font=_font(38, bold=True), fill=INK)
    return box[0] + 80, box[1] + 110, box[2] - 50, box[3] - 70


def _axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    *,
    x_label: str = "",
    y_label: str = "",
) -> None:
    left, top, right, bottom = plot
    draw.line([(left, top), (left, bottom), (right, bottom)], fill=INK, width=4)
    if x_label:
        _centered(draw, ((left + right) // 2, bottom + 45), x_label, _font(25), MUTED)
    if y_label:
        draw.text((left + 8, top + 3), y_label, font=_font(24), fill=MUTED)


def _mapping_aggregate(audit: dict[str, Any]) -> dict[str, float]:
    rows = [row for seed_rows in audit["permutation_mapping"].values() for row in seed_rows.values()]
    total = sum(int(row["sample_count"]) for row in rows)
    return {
        "Same modulation": sum(int(row["same_modulation_count"]) for row in rows) / total,
        "Same SNR": sum(int(row["same_snr_count"]) for row in rows) / total,
        "Gate changed": sum(float(row["gate_changed_fraction"]) * int(row["sample_count"]) for row in rows) / total,
        "Argmax changed": sum(float(row["argmax_flip_fraction"]) * int(row["sample_count"]) for row in rows) / total,
    }


def _figure_two(
    gate: dict[str, Any], contrasts: dict[str, Any], audit: dict[str, Any], output_dir: Path
) -> list[Path]:
    s2 = gate["models"]["s2_aligned"]
    image = Image.new("RGB", (4500, 2900), PAPER)
    draw = ImageDraw.Draw(image)
    _centered(
        draw,
        (2250, 95),
        "Descriptive S2 gate behavior and audited intervention effects",
        _font(68, bold=True),
        INK,
    )
    boxes = [
        (100, 190, 1465, 1480), (1565, 190, 2930, 1480), (3030, 190, 4395, 1480),
        (100, 1570, 1465, 2800), (1565, 1570, 2930, 2800), (3030, 1570, 4395, 2800),
    ]

    plot = _panel(draw, boxes[0], "(a) Overall branch weights")
    _axes(draw, plot, y_label="Gate weight")
    left, top, right, bottom = plot
    scale = (bottom - top) / 0.75
    centers = [left + (right - left) * (index + 0.5) / 3 for index in range(3)]
    for index, center in enumerate(centers):
        mean = float(s2["overall_mean_weight"][index])
        std = float(s2["overall_std_weight"][index])
        width = 180
        y = bottom - mean * scale
        draw.rectangle((int(center - width / 2), int(y), int(center + width / 2), bottom), fill=COLORS[index])
        low = bottom - max(0.0, mean - std) * scale
        high = bottom - min(0.75, mean + std) * scale
        draw.line([(center, high), (center, low)], fill=INK, width=5)
        draw.line([(center - 32, high), (center + 32, high)], fill=INK, width=5)
        draw.line([(center - 32, low), (center + 32, low)], fill=INK, width=5)
        _centered(draw, (int(center), bottom + 40), BRANCHES[index], _font(27), INK)
        _centered(draw, (int(center), int(y - 35)), f"{mean:.3f}", _font(25, bold=True), INK)

    plot = _panel(draw, boxes[1], "(b) SNR-stratified means")
    _axes(draw, plot, x_label="SNR (dB)")
    left, top, right, bottom = plot
    snrs = sorted(int(key) for key in s2["by_snr"])
    for branch_index in range(3):
        points = []
        for snr in snrs:
            x = left + (snr - snrs[0]) / (snrs[-1] - snrs[0]) * (right - left)
            value = float(s2["by_snr"][str(snr)]["mean_weight"][branch_index])
            y = bottom - (value - 0.15) / 0.40 * (bottom - top)
            points.append((int(x), int(y)))
        draw.line(points, fill=COLORS[branch_index], width=7)
        for point in points:
            draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=COLORS[branch_index])
        draw.text(
            (left + 210 * branch_index, top + 8),
            BRANCHES[branch_index],
            font=_font(24),
            fill=COLORS[branch_index],
        )
    for snr in (-20, -10, 0, 10, 18):
        x = left + (snr - snrs[0]) / (snrs[-1] - snrs[0]) * (right - left)
        _centered(draw, (int(x), bottom + 38), str(snr), _font(23), MUTED)

    plot = _panel(draw, boxes[2], "(c) Modulation-stratified means")
    left, top, right, bottom = plot
    label_width = 210
    grid_left = left + label_width
    cell_width = (right - grid_left) / 3
    cell_height = (bottom - top) / 11
    values = [
        float(value)
        for index in range(11)
        for value in s2["by_modulation"][str(index)]["mean_weight"]
    ]
    minimum, maximum = min(values), max(values)
    for row in range(11):
        draw.text((left, int(top + row * cell_height + 10)), MODULATIONS[row], font=_font(24), fill=INK)
        for column in range(3):
            value = float(s2["by_modulation"][str(row)]["mean_weight"][column])
            ratio = (value - minimum) / (maximum - minimum)
            color = (int(42 + ratio * 165), int(65 + ratio * 130), int(125 - ratio * 70))
            box = (
                int(grid_left + column * cell_width), int(top + row * cell_height),
                int(grid_left + (column + 1) * cell_width - 3),
                int(top + (row + 1) * cell_height - 3),
            )
            draw.rectangle(box, fill=color)
            _centered(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), f"{value:.2f}", _font(22), PAPER)
    for column, label in enumerate(BRANCHES):
        _centered(draw, (int(grid_left + (column + 0.5) * cell_width), bottom + 35), label, _font(25), INK)

    plot = _panel(draw, boxes[3], "(d) Block entropy and collapse")
    _axes(draw, plot, x_label="Block")
    left, top, right, bottom = plot
    block_width = (right - left) / 6
    for index, row in enumerate(s2["per_block"]):
        center = left + (index + 0.5) * block_width
        entropy = float(row["entropy_bits"])
        collapse = 100 * float(row["collapse_fraction"])
        y = bottom - entropy / 1.65 * (bottom - top)
        draw.rectangle((int(center - 55), int(y), int(center + 55), bottom), fill="#2374AB")
        cy = bottom - collapse / 24 * (bottom - top)
        draw.ellipse((int(center - 14), int(cy - 14), int(center + 14), int(cy + 14)), fill="#C2410C")
        if index:
            previous = s2["per_block"][index - 1]
            px = left + (index - 0.5) * block_width
            py = bottom - (100 * float(previous["collapse_fraction"])) / 24 * (bottom - top)
            draw.line([(int(px), int(py)), (int(center), int(cy))], fill="#C2410C", width=6)
        _centered(draw, (int(center), bottom + 38), str(index + 1), _font(24), INK)
    draw.text((left + 5, top + 8), "bars: entropy", font=_font(23), fill="#2374AB")
    draw.text((left + 230, top + 8), "line: collapse", font=_font(23), fill="#C2410C")

    plot = _panel(draw, boxes[4], "(e) Frozen-checkpoint interventions")
    _axes(draw, plot, y_label="Low-SNR accuracy")
    left, top, right, bottom = plot
    accuracy = {row["model"]: row["low_snr_accuracy_mean"] for row in contrasts["accuracy_rows"]}
    conditions = [
        ("Aligned", accuracy["lmscnet_s2_aligned"], "#2374AB"),
        ("Train mean", accuracy["lmscnet_s2_mean"], "#E07A5F"),
        ("Shuffled", accuracy["lmscnet_s2_shuffled"], "#3D9970"),
    ]
    centers = [left + (right - left) * (index + 0.5) / 3 for index in range(3)]
    for center, (label, value, color) in zip(centers, conditions, strict=True):
        y = bottom - float(value) / 0.72 * (bottom - top)
        draw.rectangle((int(center - 105), int(y), int(center + 105), bottom), fill=color)
        _centered(draw, (int(center), int(y - 34)), f"{value:.3f}", _font(27, bold=True), INK)
        _centered(draw, (int(center), bottom + 42), label, _font(25), INK)
    _centered(draw, ((left + right) // 2, bottom - 45), "mean/shuffled are not retrained baselines", _font(23), MUTED)

    plot = _panel(draw, boxes[5], "(f) Shuffled-mapping audit")
    left, top, right, bottom = plot
    mapping = _mapping_aggregate(audit)
    row_height = (bottom - top) / 4
    bar_colors = ["#7C3AED", "#D97706", "#0F766E", "#BE123C"]
    for index, (label, value) in enumerate(mapping.items()):
        y = top + (index + 0.5) * row_height
        draw.text((left, int(y - 18)), label, font=_font(26), fill=INK)
        bar_left = left + 315
        bar_right = bar_left + value * (right - bar_left - 80)
        draw.rounded_rectangle((bar_left, int(y - 24), int(bar_right), int(y + 24)), radius=12, fill=bar_colors[index])
        draw.text((int(bar_right + 18), int(y - 18)), f"{100 * value:.1f}%", font=_font(26, bold=True), fill=INK)

    return _save(image, output_dir / "figure2-s2-gate-and-interventions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", type=Path, default=DEFAULT_NAMESPACE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    namespace = args.namespace.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if PROJECT_ROOT.resolve() == output_dir or PROJECT_ROOT.resolve() in output_dir.parents:
        raise FigureGenerationError("Generated figures must remain outside the repository")

    gate_path = namespace / "reports/r2-gate-mechanism-b0310ec/r2-gate-mechanism.json"
    contrast_path = namespace / "reports/r2-primary-contrasts-b0310ec/r2-primary-contrasts.json"
    audit_path = namespace / "audits/r25-intervention-validity-8fa0562/r25-intervention-validity-report.json"
    gate = _load_test_isolated(gate_path)
    contrasts = _load_test_isolated(contrast_path)
    audit = _load_test_isolated(audit_path)
    if not audit.get("passed"):
        raise FigureGenerationError("R2.5 intervention-validity audit did not pass")

    output_dir.mkdir(parents=True)
    outputs = _figure_one(output_dir) + _figure_two(gate, contrasts, audit, output_dir)
    manifest = {
        "schema_version": 1,
        "purpose": "major_revision_final_figure_assets",
        "test_accessed": False,
        "neighbor_gate_curves_included": False,
        "renderer": "Pillow",
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (gate_path, contrast_path, audit_path)
        ],
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in outputs],
    }
    manifest_path = output_dir / "figure-assets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = [f"{_sha256(path)}  {path.name}" for path in [*outputs, manifest_path]]
    (output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="ascii")
    print(json.dumps({"output_dir": str(output_dir), "outputs": len(outputs), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
