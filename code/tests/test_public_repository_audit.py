from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code/scripts/audit_public_repository.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("public_repository_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()


def _load_build_module():
    script = PROJECT_ROOT / "code/scripts/build_aeu_submission.py"
    spec = importlib.util.spec_from_file_location("build_aeu_submission_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load_build_module()


def _example_author_metadata() -> dict[str, str]:
    return {
        "name": "Public Author",
        "short_name": "P. Author",
        "email": "author" + "@" + "example.invalid",
        "organization": "Example University",
        "address_line": "Example Address",
        "city": "Example City",
        "postal_code": "000000",
        "state": "Example State",
        "country": "Example Country",
        "credit_statement": "Conceptualization, Software, Writing -- original draft",
        "submission_date": "August 16, 2026",
    }


def test_audit_file_accepts_clean_utf8(tmp_path: Path) -> None:
    path = tmp_path / "clean.md"
    path.write_text("# Public documentation\n", encoding="utf-8")

    result = AUDIT.audit_file(tmp_path, path.name)

    assert result.utf8_text is True
    assert result.issues == ()


def test_audit_file_detects_language_and_privacy_issues(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.txt"
    cjk_text = chr(0x6D4B) + chr(0x8BD5)
    email = "author" + "@" + "example.com"
    path.write_text(
        f"{cjk_text}\n{email}\nC:\\Users\\private-user\\artifact.txt\n",
        encoding="utf-8",
    )

    result = AUDIT.audit_file(tmp_path, path.name)

    assert "non-English CJK text" in result.issues
    assert "email address in tracked content" in result.issues
    assert "user-specific home path in tracked content" in result.issues


def test_current_tracked_files_pass_content_audit() -> None:
    files = AUDIT.audit_files(PROJECT_ROOT, AUDIT.tracked_files(PROJECT_ROOT))

    assert not {item.path: item.issues for item in files if item.issues}


def test_submission_builder_uses_external_author_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "author.json"
    metadata_path.write_text(json.dumps(_example_author_metadata()), encoding="utf-8")

    metadata = BUILD._load_author_metadata(metadata_path)
    rendered = BUILD._manuscript_tex(
        {"title": "Public Title", "abstract": "Abstract", "keywords": ["AMR"]},
        "Body",
        metadata,
    )

    assert r"\author[1]{Public Author}" in rendered
    assert (r"\ead{" + metadata["email"] + "}") in rendered
    assert r"organization={Example University}" in rendered
