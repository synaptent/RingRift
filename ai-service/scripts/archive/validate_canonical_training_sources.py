#!/usr/bin/env python
from __future__ import annotations

"""
Utilities for validating canonical training data sources.

This module serves two roles:

- CLI: scan ai-service/scripts for obvious *.db literals and list any
  non-canonical references plus legacy DBs under data/games.
- Library helper: validate_canonical_sources(...), used by
  training_preflight_check.py to ensure training pipelines only consume
  canonical, parity-gated GameReplayDBs listed in TRAINING_DATA_REGISTRY.md.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


DB_LITERAL_RE = re.compile(r"['\"]([^'\"]+\\.db)['\"]")


def _scan_script_for_dbs(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    return [m.group(1) for m in DB_LITERAL_RE.finditer(text)]


def find_noncanonical_db_references() -> Dict[str, List[Tuple[str, str]]]:
    """
    Scan ai-service/scripts for .db string literals and return a mapping:
      script_path -> [(literal, basename), ...] for non-canonical basenames.
    """
    scripts_dir = AI_SERVICE_ROOT / "scripts"
    results: Dict[str, List[Tuple[str, str]]] = {}

    for path in sorted(scripts_dir.glob("*.py")):
        db_literals = _scan_script_for_dbs(path)
        bad: List[Tuple[str, str]] = []
        for lit in db_literals:
            basename = os.path.basename(lit)
            # Allow canonical_*.db and obvious test/fixture DBs.
            if basename.startswith("canonical_"):
                continue
            if basename in ("minimal_test.db", "training.db"):
                continue
            bad.append((lit, basename))
        if bad:
            results[str(path.relative_to(AI_SERVICE_ROOT))] = bad

    return results


def list_legacy_db_files() -> List[str]:
    """List non-canonical DB files under data/games for awareness."""
    games_dir = AI_SERVICE_ROOT / "data" / "games"
    if not games_dir.exists():
        return []
    paths: List[str] = []
    for db_path in sorted(games_dir.glob("*.db")):
        basename = db_path.name
        if basename.startswith("canonical_"):
            continue
        paths.append(str(db_path))
    return paths


# -----------------------------------------------------------------------------
# Canonical training source validation (registry + gate artefacts)
# -----------------------------------------------------------------------------


def _normalise_status(raw: str) -> str:
    """Normalise a Markdown-formatted status cell to a plain identifier.

    Examples
    --------
    "**canonical**"        -> "canonical"
    "**pending_gate**"     -> "pending_gate"
    " ⚠️ **DEPRECATED** "  -> "deprecated"
    """
    text = raw.strip()
    # Strip basic Markdown formatting. Keep underscores because they are part
    # of canonical identifiers like "pending_gate" and "legacy_noncanonical".
    for ch in ("*", "`"):
        text = text.replace(ch, "")

    # Prefer the first token that looks like an identifier, skipping emoji.
    tokens = text.strip().split()
    for token in tokens:
        cleaned = re.sub(r"[^\w]+", "", token, flags=re.UNICODE)
        if re.search(r"[A-Za-z]", cleaned):
            return cleaned.lower()

    return ""


def _parse_registry_game_dbs(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse TRAINING_DATA_REGISTRY.md game DB tables.

    Returns a mapping:

        {
          "canonical_square8.db": {
              "status": "pending_gate",
              "gate_summary": "db_health.canonical_square8.json",
          },
          ...
        }

    The parser is intentionally simple and tailored to the current Markdown
    table layout in TRAINING_DATA_REGISTRY.md; it does not try to be fully
    generic Markdown.
    """
    text = registry_path.read_text(encoding="utf-8")
    rows: Dict[str, Dict[str, Any]] = {}

    in_games_section = False
    for line in text.splitlines():
        if not in_games_section:
            if line.strip().startswith("## Game Replay Databases"):
                in_games_section = True
            continue

        # Stop once we leave the game DB section.
        if (
            in_games_section
            and line.startswith("## ")
            and "Game Replay Databases" not in line
        ):
            break

        # Only consider Markdown table rows with a backticked *.db cell.
        line_stripped = line.strip()
        if not (
            line_stripped.startswith("|")
            and ".db" in line_stripped
            and "`" in line_stripped
        ):
            continue

        # Split on '|' and trim cells.
        parts = [p.strip() for p in line_stripped.split("|")]
        if len(parts) < 6:
            continue

        # Columns: Database, Board Type, Players, Status, Gate Summary, Notes.
        db_cell = parts[1]
        status_cell = parts[4] if len(parts) > 4 else ""
        gate_cell = parts[5] if len(parts) > 5 else ""

        # Extract backticked database name, e.g. `canonical_square8.db`.
        if "`" not in db_cell:
            continue
        db_name = db_cell.strip(" `")
        if not db_name.endswith(".db"):
            continue

        status_norm = _normalise_status(status_cell)
        gate_summary = gate_cell.strip()
        # Some tables may use "-" when no gate artefact exists yet.
        if gate_summary == "-":
            gate_summary = ""

        rows[db_name] = {
            "status": status_norm,
            "gate_summary": gate_summary,
        }

    return rows


def _resolve_gate_summary_path(
    registry_path: Path,
    gate_ref: str,
) -> Path | None:
    """Resolve a gate summary reference from the registry to an on-disk JSON.

    The *gate_ref* string typically comes from the "Gate Summary" column, for
    example::

        db_health.canonical_square8.json
        parity_gate.square8.json

    Resolution strategy:

    - If *gate_ref* is an absolute path, use it directly.
    - Otherwise, try relative to the registry directory.
    - Finally, try ``<registry_dir>/data/games/<gate_ref>``.
    """
    if not gate_ref:
        return None

    gate_ref = gate_ref.strip().strip("`")
    candidate_paths = []

    if os.path.isabs(gate_ref):
        candidate_paths.append(Path(gate_ref))
    else:
        root = registry_path.resolve().parent
        candidate_paths.append(root / gate_ref)
        candidate_paths.append(root / "data" / "games" / gate_ref)

    for path in candidate_paths:
        if path.exists():
            return path

    return None


def _load_gate_summary(path: Path) -> Dict[str, Any]:
    """Load a JSON gate/health summary, returning an empty dict on failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _extract_parity_ok(meta: Dict[str, Any]) -> bool:
    """Return True when the parity gate is reported as passing.

    Handles both v1 parity summaries (top-level
    ``passed_canonical_parity_gate``) and v2 ``parity_gate`` wrappers.
    """
    # v2: summary from generate_canonical_selfplay.py
    parity_gate = meta.get("parity_gate")
    if isinstance(parity_gate, dict):
        if parity_gate.get("passed_canonical_parity_gate"):
            return True

    # v1: summary from run_parity_and_history_gate.py or parity-only scripts
    if meta.get("passed_canonical_parity_gate"):
        return True

    return False


def _is_canonical_from_summary(meta: Dict[str, Any]) -> bool:
    """Determine whether a health summary marks a DB as canonical.

    Policy (from TRAINING_DATA_REGISTRY.md):

    - canonical_ok must be True, and
    - the canonical parity gate must have passed.
    """
    canonical_ok = bool(meta.get("canonical_ok"))
    parity_ok = _extract_parity_ok(meta)
    return canonical_ok and parity_ok


def validate_canonical_sources(
    registry_path: Path,
    db_paths: List[Path],
    *,
    allowed_statuses: List[str] | None = None,
) -> Dict[str, Any]:
    """Validate that *db_paths* are canonical training sources.

    This is the central helper used by training preflight scripts. It enforces:

    - DB must be present in TRAINING_DATA_REGISTRY.md under the game DB tables.
    - Registry status for the DB must be ``canonical``.
    - The associated gate/health JSON (from the "Gate Summary" column) must
      exist and report both:

        canonical_ok == true
        and a passing canonical parity gate.

    Parameters
    ----------
    registry_path:
        Path to TRAINING_DATA_REGISTRY.md.
    db_paths:
        List of absolute or relative filesystem paths to candidate DB files.
    allowed_statuses:
        Registry Status values that are permitted for *db_paths*. Defaults to
        ``["canonical"]``. This is intentionally strict; callers should only
        relax this (e.g. to include ``pending_gate``) behind an explicit flag.

    Returns
    -------
    dict
        {
          "ok": bool,
          "problems": [<str>, ...],
          "checked": [<str-path>, ...],
        }
    """
    problems: List[str] = []
    checked: List[str] = []

    registry_path = registry_path.resolve()
    registry_entries = _parse_registry_game_dbs(registry_path)
    allowed = [s.strip().lower() for s in (allowed_statuses or ["canonical"]) if s.strip()]
    if not allowed:
        allowed = ["canonical"]

    for raw_path in db_paths:
        db_path = raw_path.resolve()
        checked.append(str(db_path))

        basename = db_path.name
        prefix = f"{db_path}"

        entry = registry_entries.get(basename)
        if entry is None:
            problems.append(
                f"{prefix}: not listed in TRAINING_DATA_REGISTRY.md "
                "game DB tables; add a row with Status=canonical once "
                "the DB has passed the canonical parity and history "
                "gates."
            )
            continue

        status = str(entry.get("status") or "").lower()
        if status not in allowed:
            problems.append(
                f"{prefix}: registry Status={status!r} "
                f"(expected one of: {', '.join(allowed)}); this DB is not approved "
                "for training use."
            )

        gate_ref = str(entry.get("gate_summary") or "").strip()
        if not gate_ref:
            problems.append(
                f"{prefix}: registry Gate Summary column is empty or "
                "'-' so there is no health summary JSON to prove "
                "canonical_ok/parity gate."
            )
            continue

        gate_path = _resolve_gate_summary_path(registry_path, gate_ref)
        if gate_path is None:
            problems.append(
                f"{prefix}: health summary JSON '{gate_ref}' not found "
                f"relative to {registry_path.parent}; cannot confirm "
                "canonical status."
            )
            continue

        meta = _load_gate_summary(gate_path)
        if not meta:
            problems.append(
                f"{prefix}: failed to load or parse health summary JSON "
                f"at {gate_path}; cannot confirm canonical status."
            )
            continue

        if not _is_canonical_from_summary(meta):
            canonical_ok = bool(meta.get("canonical_ok"))
            parity_ok = _extract_parity_ok(meta)
            problems.append(
                f"{prefix}: health summary {gate_path.name} reports "
                f"canonical_ok={canonical_ok!r}, "
                f"passed_canonical_parity_gate={parity_ok!r}; both must "
                "be true for canonical training sources."
            )

    return {
        "ok": not problems,
        "problems": problems,
        "checked": checked,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that training-related scripts only reference "
            "canonical_*.db GameReplayDBs and surface any legacy DBs "
            "still present under data/games."
        )
    )
    parser.add_argument(
        "--list-legacy-dbs",
        action="store_true",
        help="Also list non-canonical *.db files under ai-service/data/games.",
    )
    args = parser.parse_args(argv)

    noncanonical_refs = find_noncanonical_db_references()
    legacy_dbs = list_legacy_db_files() if args.list_legacy_dbs else []

    if noncanonical_refs:
        print(
            "[canonical-sources] Non-canonical DB references found "
            "in scripts:\n"
        )
        for script, refs in noncanonical_refs.items():
            print(f"  {script}:")
            for lit, basename in refs:
                print(f"    literal={lit!r} (basename={basename})")
        print()
        print(
            "Update these scripts to:\n"
            "  - Use canonical_<board>.db for production training paths, or\n"
            "  - Clearly mark non-canonical DBs as test/legacy only."
        )

    if legacy_dbs:
        print(
            "\n[canonical-sources] Non-canonical DB files under "
            "ai-service/data/games:"
        )
        for p in legacy_dbs:
            print(f"  {p}")
        print(
            "\nUse scripts/scan_canonical_phase_dbs.py or targeted "
            "rm/mv commands to delete or archive these once you are "
            "confident they are no longer needed."
        )

    return 0 if not noncanonical_refs else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
