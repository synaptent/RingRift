"""Contract checks for the supported documentation path.

The repository still keeps historical docs, archived plans, and diagnostic
notes. This test intentionally covers only the public/supported docs a fresh
reader is expected to follow first, so archival churn does not block unrelated
work while credibility-critical links remain ratcheted.
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

SUPPORTED_DOCS = (
    "README.md",
    "QUICKSTART.md",
    "DOCUMENTATION_INDEX.md",
    "docs/INDEX.md",
    "docs/PROJECT_BRIEF.md",
    "docs/RESEARCH_SNAPSHOT.md",
    "docs/RESULTS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ARCHITECTURE_OVERVIEW.md",
    "docs/CODEBASE_QUALITY_PROGRAM.md",
    "docs/security/SECURITY_THREAT_MODEL.md",
    "docs/security/SUPPLY_CHAIN_AND_CI_SECURITY.md",
    "ai-service/app/README.md",
    "ai-service/models/README.md",
    "ai-service/scripts/README.md",
)

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _iter_local_link_targets(doc_path: Path) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    for match in MARKDOWN_LINK_RE.finditer(doc_path.read_text(encoding="utf-8")):
        raw_target = match.group(1).strip()
        if not raw_target or raw_target.startswith("#"):
            continue

        if raw_target.startswith("<") and raw_target.endswith(">"):
            raw_target = raw_target[1:-1]
        elif " " in raw_target:
            raw_target = raw_target.split()[0]

        parsed = urllib.parse.urlparse(raw_target)
        if parsed.scheme or raw_target.startswith(("mailto:", "tel:")):
            continue

        link_path = urllib.parse.unquote(parsed.path)
        if not link_path:
            continue

        if link_path.startswith("/"):
            resolved = REPO_ROOT / link_path.lstrip("/")
        else:
            resolved = doc_path.parent / link_path
        targets.append((raw_target, resolved))

    return targets


def test_supported_docs_exist() -> None:
    missing_docs = [doc for doc in SUPPORTED_DOCS if not (REPO_ROOT / doc).exists()]
    assert not missing_docs, "Supported docs are missing:\n" + "\n".join(missing_docs)


def test_supported_docs_have_no_local_absolute_paths() -> None:
    offenders: list[str] = []
    for doc in SUPPORTED_DOCS:
        doc_path = REPO_ROOT / doc
        text = doc_path.read_text(encoding="utf-8")
        if str(REPO_ROOT) in text:
            offenders.append(doc)

    assert not offenders, "Supported docs contain local machine paths:\n" + "\n".join(offenders)


def test_supported_docs_local_links_resolve() -> None:
    missing_links: list[str] = []
    for doc in SUPPORTED_DOCS:
        doc_path = REPO_ROOT / doc
        for raw_target, resolved in _iter_local_link_targets(doc_path):
            if not resolved.exists():
                try:
                    display = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    display = resolved
                missing_links.append(f"{doc}: {raw_target} -> {display}")

    assert not missing_links, "Supported docs contain broken local links:\n" + "\n".join(missing_links)
