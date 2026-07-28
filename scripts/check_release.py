"""Run lightweight checks before publishing the repository."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SENSITIVE_PATTERNS = [
    re.compile(r"202\.120\.54\.218"),
    re.compile(r"C:\\Users\\Admin", re.IGNORECASE),
    re.compile(r"password\s*=", re.IGNORECASE),
    re.compile(r"passwd\s*=", re.IGNORECASE),
    re.compile(r"HostName\s+\d+\.\d+\.\d+\.\d+", re.IGNORECASE),
]

BLOCKED_EXTENSIONS = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".pkl", ".npz", ".h5"}
MAX_TRACKED_FILE_MB = 10


def check_python_syntax() -> None:
    errors: list[str] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            compile(path.read_text(encoding="utf-8", errors="replace"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(PROJECT_ROOT)}:{exc.lineno}: {exc.msg}")
    if errors:
        raise SystemExit("Syntax errors:\n" + "\n".join(errors))
    print("Syntax check passed.")


def check_sensitive_text() -> None:
    hits: list[str] = []
    checker = Path(__file__).resolve()
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.resolve() == checker:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".docx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path.relative_to(PROJECT_ROOT)}: {pattern.pattern}")
    if hits:
        raise SystemExit("Sensitive strings found:\n" + "\n".join(hits))
    print("Sensitive-string check passed.")


def check_large_or_blocked_files() -> None:
    problems: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PROJECT_ROOT)
        if path.suffix.lower() in BLOCKED_EXTENSIONS:
            problems.append(f"blocked extension: {rel}")
        if path.stat().st_size > MAX_TRACKED_FILE_MB * 1024 * 1024:
            problems.append(f"large file > {MAX_TRACKED_FILE_MB} MB: {rel}")
        if rel.parts[:2] in {("data", "raw"), ("data", "processed")} and path.name != ".gitkeep":
            problems.append(f"data file should not be tracked: {rel}")
    if problems:
        raise SystemExit("Release file-policy violations:\n" + "\n".join(problems))
    print("File-policy check passed.")


def check_smoke() -> None:
    subprocess.run([sys.executable, "-m", "rnadgg.smoke"], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    check_python_syntax()
    check_sensitive_text()
    check_large_or_blocked_files()
    check_smoke()
    print("Release checks passed.")


if __name__ == "__main__":
    main()
