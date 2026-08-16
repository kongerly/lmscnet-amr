"""Prune references.bib to manuscript citations and archive unused entries."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BibliographyPruneError(ValueError):
    """Raised when a BibTeX file cannot be pruned without ambiguity."""


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\[(@[^\]]+)\]", text):
        for item in group.split(";"):
            match = re.match(r"\s*@([A-Za-z0-9_:-]+)", item)
            if match:
                keys.add(match.group(1))
    return keys


def split_entries(text: str) -> list[tuple[str, str]]:
    starts = list(re.finditer(r"(?m)^@(\w+)\{([^,]+),", text))
    entries: list[tuple[str, str]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        entries.append((match.group(2), text[match.start() : end].strip()))
    return entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=PROJECT_ROOT / "paper/manuscript_major_revision_2026-08-15.md",
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=PROJECT_ROOT / "literature/bibliography/references.bib",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT
        / "literature/bibliography/references_unused_major_revision_2026-08-15.bib",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manuscript = args.manuscript.read_text(encoding="utf-8")
    bibliography = args.bibliography.read_text(encoding="utf-8")
    used = citation_keys(manuscript)
    entries = split_entries(bibliography)
    available = {key for key, _ in entries}
    missing = used - available
    if missing:
        raise BibliographyPruneError(f"Missing citation keys: {sorted(missing)}")
    if args.archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {args.archive}")

    retained = [entry for key, entry in entries if key in used]
    unused = [entry for key, entry in entries if key not in used]
    args.archive.write_text("\n\n".join(unused) + "\n", encoding="utf-8")
    args.bibliography.write_text("\n\n".join(retained) + "\n", encoding="utf-8")
    print(
        {
            "used_count": len(retained),
            "unused_archived_count": len(unused),
            "archive": str(args.archive),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
