"""Build a self-contained AEU CAS submission package from the frozen manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

AUTHOR_METADATA_FIELDS = (
    "name",
    "short_name",
    "email",
    "organization",
    "address_line",
    "city",
    "postal_code",
    "state",
    "country",
    "credit_statement",
    "submission_date",
)

HIGHLIGHTS = [
    "Five-seed and fixed-epoch analyses separate preference, capacity, and assignment.",
    "Learned-static fusion leaves the independent benefit of sample gating unresolved.",
    "A parameter-matched control rules out parameter count as a sufficient explanation.",
    "Frozen checkpoints depend on gate assignment under audited interventions.",
    "SKNet-1D and AFNet show no stable difference under the common protocol.",
]

TABLE_LABELS = {
    1: "tab:rq-control-estimand",
    2: "tab:neighbor-adaptation-mapping",
    3: "tab:validation-means",
    4: "tab:gate-interventions",
    5: "tab:primary-contrasts",
    6: "tab:efficiency",
    7: "tab:historical-test",
    8: "tab:seed-level-differences",
}

FIGURE_FILES = {
    1: "figure1-controlled-evidence-design.pdf",
    2: "figure2-s2-gate-and-interventions.pdf",
}


class AEUSubmissionError(ValueError):
    """Raised when the AEU package cannot be generated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    display_path = path.relative_to(base.resolve()).as_posix() if base is not None else path.name
    return {"path": display_path, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load_author_metadata(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AEUSubmissionError("Author metadata must be a JSON object")
    missing = [field for field in AUTHOR_METADATA_FIELDS if not str(value.get(field, "")).strip()]
    if missing:
        raise AEUSubmissionError(f"Author metadata is missing required fields: {missing}")
    extra = sorted(set(value) - set(AUTHOR_METADATA_FIELDS))
    if extra:
        raise AEUSubmissionError(f"Author metadata contains unsupported fields: {extra}")
    metadata = {field: str(value[field]).strip() for field in AUTHOR_METADATA_FIELDS}
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", metadata["email"]):
        raise AEUSubmissionError("Author metadata contains an invalid email address")
    return metadata


def _escape_plain(text: str) -> str:
    text = text.replace("\u2013", "--").replace("\u2014", "---")
    text = text.replace("\u2212", "-").replace("\u00a0", " ")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


INLINE_TOKEN = re.compile(r"(`[^`]+`|\$[^$]+\$|\[@[^\]]+\]|\*\*[^*]+\*\*)")


def _inline(text: str) -> str:
    output: list[str] = []
    position = 0
    for match in INLINE_TOKEN.finditer(text):
        output.append(_escape_plain(text[position : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            output.append(r"\texttt{" + _escape_plain(token[1:-1]) + "}")
        elif token.startswith("$"):
            output.append(token)
        elif token.startswith("[@"):
            keys = re.findall(r"@([A-Za-z0-9_:-]+)", token)
            output.append(r"\citep{" + ",".join(keys) + "}")
        else:
            output.append(r"\textbf{" + _inline(token[2:-2]) + "}")
        position = match.end()
    output.append(_escape_plain(text[position:]))
    return "".join(output)


def _section_title(raw: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", raw).strip()


def _table(lines: list[str], caption: str, number: int) -> str:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) < 3:
        raise AEUSubmissionError(f"Table {number} is incomplete")
    header = rows[0]
    body = rows[2:]
    columns = "l" + "c" * (len(header) - 1)
    environment = "table*" if len(header) >= 4 else "table"
    if number in {1, 2}:
        wrapped_columns = "@{}l" + "Y" * (len(header) - 1) + "@{}"
        rendered = [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            rf"\caption{{{_inline(caption)}}}",
            rf"\label{{{TABLE_LABELS[number]}}}",
            rf"\begin{{tabularx}}{{\linewidth}}{{{wrapped_columns}}}",
            r"\toprule",
            " & ".join(_inline(cell) for cell in header) + r" \\",
            r"\midrule",
        ]
        rendered.extend(" & ".join(_inline(cell) for cell in row) + r" \\" for row in body)
        rendered.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
        return "\n".join(rendered)
    position = "!htbp" if number == 8 else "t"
    rendered = [
        rf"\begin{{{environment}}}[{position}]",
        r"\centering",
        r"\small",
        rf"\caption{{{_inline(caption)}}}",
        rf"\label{{{TABLE_LABELS[number]}}}",
        rf"\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(_inline(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    rendered.extend(" & ".join(_inline(cell) for cell in row) + r" \\" for row in body)
    rendered.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            rf"\end{{{environment}}}",
        ]
    )
    if number == 8:
        rendered.insert(0, r"\FloatBarrier")
        rendered.append(r"\FloatBarrier")
    return "\n".join(rendered)


def _figure(caption: str, number: int) -> str:
    lines = [
        r"\begin{figure}",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{figures/{FIGURE_FILES[number]}}}",
        rf"\caption{{{_inline(caption)}}}",
        rf"\label{{fig:{number}}}",
        r"\end{figure}",
    ]
    if number == max(FIGURE_FILES):
        lines.append(r"\FloatBarrier")
    return "\n".join(lines)


def _convert_body(markdown: str) -> tuple[str, list[str]]:
    body_start = markdown.index("## 1. Introduction")
    body_end = markdown.index("## References")
    lines = markdown[body_start:body_end].splitlines()
    output: list[str] = []
    figure_captions: list[str] = []
    index = 0
    pending_table: tuple[int, str] | None = None

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            output.append("")
            index += 1
            continue

        if line == "$$":
            math_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                math_lines.append(lines[index].rstrip())
                index += 1
            if index >= len(lines):
                raise AEUSubmissionError("Unclosed display-math block")
            output.extend([r"\begin{equation}", *math_lines, r"\end{equation}"])
            index += 1
            continue

        heading = re.match(r"^(#{2,3})\s+(.+)$", line)
        if heading:
            level, title = heading.groups()
            title = _section_title(title)
            appendix = title.startswith("Appendix A.")
            if appendix:
                output.append(r"\appendix")
                title = title.removeprefix("Appendix A.").strip()
            starred = title in {
                "Data and Code Availability",
                "Statements",
                "Declaration of Generative AI and AI-Assisted Technologies",
            }
            command = "section" if level == "##" else "subsection"
            output.append(rf"\{command}{'*' if starred else ''}{{{_inline(title)}}}")
            index += 1
            continue

        table_caption = re.match(r"^\*\*Table (\d+)\. (.+)\*\*$", line)
        if table_caption:
            pending_table = (int(table_caption.group(1)), table_caption.group(2))
            index += 1
            continue

        if line.startswith("|"):
            if pending_table is None:
                raise AEUSubmissionError("Markdown table has no numbered caption")
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            number, caption = pending_table
            output.append(_table(table_lines, caption, number))
            pending_table = None
            continue

        figure_caption = re.match(r"^\*\*Figure (\d+)\. ([^*]+)\*\*\s*(.*)$", line)
        if figure_caption:
            number = int(figure_caption.group(1))
            caption = " ".join(
                part.strip()
                for part in (figure_caption.group(2), figure_caption.group(3))
                if part.strip()
            )
            figure_captions.append(f"Figure {number}. {caption}")
            output.append(_figure(caption, number))
            index += 1
            continue

        labeled = re.match(r"^\*\*([^*]+)\*\*\s*(.*)$", line)
        if labeled:
            label, remainder = labeled.groups()
            output.append(rf"\paragraph{{{_inline(label)}}} {_inline(remainder)}")
            index += 1
            continue

        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            next_line = lines[index].strip()
            if next_line.startswith(("##", "###", "|", "$$", "**Table", "**Figure")):
                break
            paragraph.append(next_line)
            index += 1
        output.append(_inline(" ".join(paragraph)))

    if pending_table is not None:
        raise AEUSubmissionError("Final table caption has no table")
    return "\n".join(output).strip() + "\n", figure_captions


def _extract_frontmatter(markdown: str) -> dict[str, Any]:
    title = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    abstract = re.search(
        r"^## Abstract\s+(.+?)\s+\*\*Keywords:\*\* ([^\r\n]+)$",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    if title is None or abstract is None:
        raise AEUSubmissionError("Could not parse title, abstract, or keywords")
    return {
        "title": title.group(1).strip(),
        "abstract": abstract.group(1).strip(),
        "keywords": [item.strip() for item in abstract.group(2).split(";")],
    }


def _manuscript_tex(
    frontmatter: dict[str, Any], body: str, author: dict[str, str]
) -> str:
    highlights = "\n".join(rf"\item {_inline(item)}" for item in HIGHLIGHTS)
    keywords = r" \sep ".join(_inline(item) for item in frontmatter["keywords"])
    return rf"""\documentclass[a4paper,fleqn]{{cas-sc}}

\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{placeins}}
\usepackage{{textcomp}}
\usepackage{{url}}
\usepackage[authoryear]{{natbib}}
\graphicspath{{{{figures/}}}}
\emergencystretch=2em
\RenewDocumentCommand{{\printorcid}}{{}}{{}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}

\hypersetup{{
  pdftitle={{{_inline(frontmatter['title'])}}},
  pdfauthor={{{_inline(author['name'])}}},
  pdfsubject={{Submission to AEU - International Journal of Electronics and Communications}},
  pdfkeywords={{{'; '.join(frontmatter['keywords'])}}}
}}

\begin{{document}}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{{1}}
\def\textpagefraction{{.001}}

\shorttitle{{Controlled multi-scale gating for low-SNR AMR}}
\shortauthors{{{_inline(author['short_name'])}}}
\title [mode = title]{{{_inline(frontmatter['title'])}}}

\author[1]{{{_inline(author['name'])}}}
\cormark[1]
\ead{{{_inline(author['email'])}}}
\credit{{{_inline(author['credit_statement'])}}}

\affiliation[1]{{organization={{{_inline(author['organization'])}}},
            addressline={{{_inline(author['address_line'])}}},
            city={{{_inline(author['city'])}}},
            postcode={{{_inline(author['postal_code'])}}},
            state={{{_inline(author['state'])}}},
            country={{{_inline(author['country'])}}}}}

\cortext[1]{{Corresponding author}}

\begin{{abstract}}
{_inline(frontmatter['abstract'])}
\end{{abstract}}

\begin{{highlights}}
{highlights}
\end{{highlights}}

\begin{{keywords}}
{keywords}
\end{{keywords}}

\maketitle

{body}

\printcredits

\bibliographystyle{{cas-model2-names}}
\bibliography{{references}}

\end{{document}}
"""


def _cover_letter(title: str, author: dict[str, str]) -> str:
    return f"""{author['submission_date']}

Editor-in-Chief
AEU - International Journal of Electronics and Communications

Dear Editor-in-Chief,

I am pleased to submit the manuscript entitled \"{title}\" for consideration as a Research Article in AEU - International Journal of Electronics and Communications.

The manuscript presents a controlled empirical study of multi-scale gating for low-SNR automatic modulation recognition. Under one five-seed RadioML 2016.10A validation protocol, it compares equal fusion, learned-static fusion, a parameter-matched static control, audited post-training gate interventions, and source-informed SKNet-1D and AFNet adaptations. A pre-specified fixed-epoch sensitivity route adds 25 validation runs and reproduces the same evidence boundary. The results show that parameter count alone was insufficient under the evaluated matched control and that frozen checkpoints depend on their input-conditioned gate assignments. They do not establish an independent advantage over learned-static fusion or a performance advantage over the direct adaptive-fusion neighbors. This boundary is reported explicitly.

The submission fits the journal's interest in communication systems and related signal-processing methods. Its contribution is methodological and empirical: it separates competing explanations that are often conflated in adaptive-fusion comparisons, reports null findings alongside positive findings, and provides auditable intervention and reproducibility records. The study is limited to synthetic benchmarks and makes no over-the-air, real-time, or deployment claim.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. I am the sole author and approve its submission. The study involved no human participants or animals. I declare no competing interests and received no external funding. Data and code availability and the use of generative AI and AI-assisted technologies are disclosed in the manuscript.

Thank you for considering this submission.

Sincerely,

{author['name']}
{author['organization']}
{author['city']}, {author['state']}, {author['country']}
{author['email']}
"""


def _author_metadata_text(author: dict[str, str]) -> str:
    return f"""Author: {author['name']}
Affiliation: {author['organization']}
Address: {author['address_line']}
City: {author['city']}
Province or state: {author['state']}
Postcode: {author['postal_code']}
Country: {author['country']}
Corresponding author: {author['name']}
Email: {author['email']}
ORCID: provide or omit in the submission system
"""


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        required=True,
    )
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--author-metadata",
        type=Path,
        required=True,
        help="Untracked external JSON containing the required author metadata fields.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manuscript_path = args.manuscript.resolve()
    bibliography_path = args.bibliography.resolve()
    template_dir = args.template_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    author_metadata_path = args.author_metadata.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    markdown = manuscript_path.read_text(encoding="utf-8")
    bibliography = bibliography_path.read_text(encoding="utf-8")
    author_metadata = _load_author_metadata(author_metadata_path)
    frontmatter = _extract_frontmatter(markdown)
    body, figure_captions = _convert_body(markdown)

    citation_keys = {
        key
        for cluster in re.findall(r"\[(@[^\]]+)\]", markdown)
        for key in re.findall(r"@([A-Za-z0-9_:-]+)", cluster)
    }
    bibliography_keys = set(re.findall(r"^@\w+\{([^,]+),", bibliography, flags=re.MULTILINE))
    if citation_keys != bibliography_keys or len(citation_keys) != 30:
        raise AEUSubmissionError(
            f"Citation closure failed: used={len(citation_keys)}, bib={len(bibliography_keys)}, "
            f"missing={sorted(citation_keys - bibliography_keys)}, "
            f"unused={sorted(bibliography_keys - citation_keys)}"
        )
    if len(figure_captions) != 2:
        raise AEUSubmissionError(f"Expected two figure captions, found {len(figure_captions)}")
    if any(len(item) > 85 for item in HIGHLIGHTS):
        raise AEUSubmissionError("At least one highlight exceeds 85 characters")

    output_dir.mkdir(parents=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir()

    input_paths = [manuscript_path, bibliography_path, author_metadata_path]
    for filename in ("cas-sc.cls", "cas-common.sty", "stfloats.sty"):
        source = template_dir / filename
        shutil.copy2(source, output_dir / filename)
        input_paths.append(source)
    for source in sorted((template_dir / "thumbnails").glob("*.jpeg")):
        shutil.copy2(source, thumbnails_dir / source.name)
        input_paths.append(source)
    for filename in FIGURE_FILES.values():
        source = figure_dir / filename
        shutil.copy2(source, figures_dir / filename)
        input_paths.append(source)

    shutil.copy2(bibliography_path, output_dir / "references.bib")
    _write_text(
        output_dir / "manuscript.tex",
        _manuscript_tex(frontmatter, body, author_metadata),
    )
    _write_text(output_dir / "highlights.txt", "\n".join(f"- {item}" for item in HIGHLIGHTS))
    _write_text(
        output_dir / "cover_letter.txt",
        _cover_letter(frontmatter["title"], author_metadata),
    )
    _write_text(output_dir / "figure_captions.txt", "\n\n".join(figure_captions))
    _write_text(
        output_dir / "declaration_of_interest.txt",
        "Declaration of interests\n\nThe author declares no competing interests.",
    )
    _write_text(
        output_dir / "author_metadata.txt",
        _author_metadata_text(author_metadata),
    )
    _write_text(
        output_dir / "submission_checklist.md",
        """# AEU Submission Checklist

- [x] CAS `cas-sc` editable LaTeX source created from the frozen manuscript.
- [x] Two final PDF figures copied from the audited revision namespace.
- [x] Thirty cited bibliography entries copied with no missing or unused keys.
- [x] Five Highlights prepared; every item is at most 85 characters.
- [x] Cover letter and declaration of interests drafted.
- [ ] Confirm ORCID and all submission-system author metadata.
- [ ] Recheck the AEU Guide for Authors on the actual submission date.
- [ ] Verify current WoS/JCR, institutional ranking, fees, and OA policy.
- [ ] Confirm whether a graphical abstract is requested or optional.
- [ ] Perform the final PDF visual inspection and upload-file audit.
- [ ] Generate the final submission ZIP only after all unchecked items are resolved.

No training, validation replay, bootstrap, new metric, or test access is authorized.
""",
    )
    _write_text(
        output_dir / "change_log.md",
        """# AEU Format-Conversion Change Log

- Converted the frozen Markdown manuscript to the Elsevier CAS `cas-sc` structure.
- Preserved the title, abstract, keywords, equations, numerical values, tables, captions, citations, claims, and limitations.
- Replaced Markdown tables and figure-caption placeholders with LaTeX floats.
- Added journal-facing Highlights and a cover letter without adding a superiority, confirmatory-test, OTA, real-time, or deployment claim.
- Reused only the CAS class/style files from the archived DSP package; no obsolete DSP manuscript text, figure, result, or submission claim was reused.
""",
    )

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "purpose": "aeu_submission_source_package",
        "journal": "AEU - International Journal of Electronics and Communications",
        "template": "Elsevier CAS cas-sc, reused from archived official CAS Bundle v2.4",
        "test_accessed": False,
        "inputs": [_binding(path) for path in input_paths],
        "outputs": [_binding(path, base=output_dir) for path in output_files],
        "checks": {
            "citation_key_count": len(citation_keys),
            "citation_closure": True,
            "figure_count": len(figure_captions),
            "table_count": len(TABLE_LABELS),
            "highlight_count": len(HIGHLIGHTS),
            "highlight_max_characters": max(map(len, HIGHLIGHTS)),
            "scientific_claims_changed": False,
            "historical_test_reopened": False,
        },
    }
    manifest_path = output_dir / "submission-source-manifest.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))

    checksum_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    checksums = [f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in checksum_files]
    _write_text(output_dir / "SHA256SUMS", "\n".join(checksums))

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "citation_keys": len(citation_keys),
                "figures": len(figure_captions),
                "tables": len(TABLE_LABELS),
                "test_accessed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
