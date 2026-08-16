"""Audit every tracked file and Git commit for public-release hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISALLOWED_BASENAMES = {
    ".env",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
DISALLOWED_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
DISALLOWED_PATH_PARTS = {
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "checkpoints",
    "datasets",
    "outputs",
    "runs",
    "secrets",
    "wandb",
}
MAX_PUBLIC_FILE_BYTES = 1024 * 1024

CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
WINDOWS_HOME_PATTERN = re.compile(
    r"(?i)\b[A-Z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s]+"
)
POSIX_HOME_PATTERN = re.compile(r"/(?:Users|home)/[^/\s]+")
PRIVATE_KEY_PATTERN = re.compile(
    "-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
TOKEN_PATTERNS = (
    re.compile("A" + r"KIA[0-9A-Z]{16}"),
    re.compile("gh" + r"[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}"),
    re.compile("sk" + r"-[A-Za-z0-9_-]{20,}"),
)


@dataclass(frozen=True)
class FileAudit:
    path: str
    size_bytes: int
    sha256: str
    utf8_text: bool
    issues: tuple[str, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def tracked_files(root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item)


def audit_file(root: Path, relative_path: str) -> FileAudit:
    path = root / relative_path
    data = path.read_bytes()
    issues: list[str] = []
    pure_path = Path(relative_path)
    path_parts = {part.lower() for part in pure_path.parts}

    if pure_path.name.lower() in DISALLOWED_BASENAMES:
        issues.append("disallowed sensitive filename")
    if pure_path.suffix.lower() in DISALLOWED_SUFFIXES:
        issues.append("disallowed dataset, model, or credential suffix")
    if path_parts & DISALLOWED_PATH_PARTS:
        issues.append("disallowed artifact directory")
    if len(data) > MAX_PUBLIC_FILE_BYTES:
        issues.append(f"file exceeds {MAX_PUBLIC_FILE_BYTES} bytes")
    if b"\0" in data:
        issues.append("binary content requires an explicit public-release exception")
        return FileAudit(relative_path, len(data), _sha256(data), False, tuple(issues))

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        issues.append("file is not valid UTF-8")
        return FileAudit(relative_path, len(data), _sha256(data), False, tuple(issues))

    if CJK_PATTERN.search(text):
        issues.append("non-English CJK text")
    if EMAIL_PATTERN.search(text):
        issues.append("email address in tracked content")
    if WINDOWS_HOME_PATTERN.search(text) or POSIX_HOME_PATTERN.search(text):
        issues.append("user-specific home path in tracked content")
    if PRIVATE_KEY_PATTERN.search(text):
        issues.append("private-key material")
    if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
        issues.append("credential-like token")

    return FileAudit(relative_path, len(data), _sha256(data), True, tuple(issues))


def audit_files(root: Path, paths: Iterable[str]) -> list[FileAudit]:
    return [audit_file(root, path) for path in paths]


def audit_history(root: Path, *, as_of: date) -> list[dict[str, str]]:
    format_string = "%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%aI%x1e"
    raw = _git(root, "log", "HEAD", f"--format={format_string}")
    issues: list[dict[str, str]] = []
    for record in raw.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        fields = record.split("\x1f")
        if len(fields) != 6:
            issues.append({"commit": "unknown", "issue": "unparseable commit metadata"})
            continue
        commit, author_name, author_email, committer_name, committer_email, authored_at = fields
        if not author_email.lower().endswith("@users.noreply.github.com"):
            issues.append({"commit": commit, "issue": "author email is not a GitHub noreply address"})
        if not committer_email.lower().endswith("@users.noreply.github.com"):
            issues.append(
                {"commit": commit, "issue": "committer email is not a GitHub noreply address"}
            )
        if datetime.fromisoformat(authored_at).date() > as_of:
            issues.append({"commit": commit, "issue": f"author date is after {as_of.isoformat()}"})
        if not author_name.strip() or not committer_name.strip():
            issues.append({"commit": commit, "issue": "empty author or committer name"})
    return issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument("--skip-history", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve(strict=True)
    paths = tracked_files(root)
    files = audit_files(root, paths)
    history_issues = [] if args.skip_history else audit_history(root, as_of=args.as_of_date)
    failed_files = [item for item in files if item.issues]
    report = {
        "schema_version": 1,
        "purpose": "public_repository_file_by_file_audit",
        "repository_root_name": root.name,
        "as_of_date": args.as_of_date.isoformat(),
        "tracked_file_count": len(files),
        "files": [asdict(item) for item in files],
        "history_issues": history_issues,
        "passed": not failed_files and not history_issues,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if output == root or root in output.parents:
            raise ValueError("Audit output must remain outside the repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "tracked_file_count": len(files),
                "failed_file_count": len(failed_files),
                "history_issue_count": len(history_issues),
                "passed": report["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
