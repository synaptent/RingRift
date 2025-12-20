from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]


def resolve_registry_path(registry_path: Path | None = None) -> Path:
    """Resolve TRAINING_DATA_REGISTRY.md path with a repo-local default."""
    return registry_path or (AI_SERVICE_ROOT / "TRAINING_DATA_REGISTRY.md")


def parse_registry(registry_path: Path) -> dict[str, dict[str, str]]:
    """Parse TRAINING_DATA_REGISTRY.md and return DB info keyed by filename."""
    if not registry_path.exists():
        return {}

    content = registry_path.read_text(encoding="utf-8")
    result: dict[str, dict[str, str]] = {}

    table_row_pattern = re.compile(
        r"^\s*\|\s*`?([^`|]+\.db)`?\s*\|"
        r"\s*([^|]*)\|"
        r"\s*([^|]*)\|"
        r"\s*\*?\*?([^|*]+)\*?\*?\s*\|"
        r"\s*([^|]*)\|"
        r"\s*([^|]*)\|",
        re.MULTILINE,
    )

    for match in table_row_pattern.finditer(content):
        db_name = match.group(1).strip()
        status = match.group(4).strip().lower()
        gate_summary = match.group(5).strip()

        if db_name.lower() == "database" or status == "status":
            continue

        result[db_name] = {
            "status": status,
            "gate_summary": gate_summary,
        }

    return result


def load_gate_summary(registry_dir: Path, gate_summary_name: str) -> dict | None:
    """Load a gate summary JSON file relative to the registry directory."""
    if not gate_summary_name or gate_summary_name == "-":
        return None

    summary_path = registry_dir / gate_summary_name
    if not summary_path.exists():
        candidate = Path(gate_summary_name)
        if candidate.parent == Path("."):
            summary_path = registry_dir / "data" / "games" / gate_summary_name
        if not summary_path.exists():
            return None

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def validate_canonical_sources(
    registry_path: Path,
    db_paths: Iterable[Path],
    *,
    allowed_statuses: list[str] | None = None,
) -> dict:
    """Validate that all referenced DBs have allowed canonical status."""
    if allowed_statuses is None:
        allowed_statuses = ["canonical"]

    allowed_statuses = [s.lower() for s in allowed_statuses]

    problems: list[str] = []
    checked: list[str] = []

    registry_info = parse_registry(registry_path)
    registry_dir = registry_path.parent

    for db_path in db_paths:
        db_name = Path(db_path).name

        if db_name not in registry_info:
            problems.append(
                f"Database '{db_name}' not found in registry {registry_path}"
            )
            continue

        info = registry_info[db_name]
        status = info.get("status", "").lower()

        if status not in allowed_statuses:
            problems.append(
                f"Database '{db_name}' has status '{status}' "
                f"(allowed: {allowed_statuses})"
            )
            continue

        gate_summary_name = info.get("gate_summary", "")
        if gate_summary_name and gate_summary_name != "-":
            gate_data = load_gate_summary(registry_dir, gate_summary_name)
            if gate_data:
                if "canonical_ok" in gate_data:
                    if not bool(gate_data.get("canonical_ok")):
                        problems.append(
                            f"Database '{db_name}' failed canonical gate "
                            f"(gate_summary: {gate_summary_name})"
                        )
                        continue
                else:
                    parity_gate = gate_data.get("parity_gate", {})
                    if parity_gate:
                        passed = bool(parity_gate.get("passed_canonical_parity_gate", True))
                    else:
                        passed = bool(gate_data.get("passed_canonical_parity_gate", True))
                    if not passed:
                        problems.append(
                            f"Database '{db_name}' failed parity gate "
                            f"(gate_summary: {gate_summary_name})"
                        )
                        continue

        checked.append(db_name)

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "checked": checked,
    }


def enforce_canonical_sources(
    db_paths: Iterable[Path],
    *,
    registry_path: Path | None = None,
    allowed_statuses: list[str] | None = None,
    allow_noncanonical: bool = False,
    error_prefix: str = "canonical-source",
) -> None:
    """Raise SystemExit if any DBs are not canonical by registry policy."""
    if allow_noncanonical:
        return

    registry = resolve_registry_path(registry_path)
    result = validate_canonical_sources(
        registry_path=registry,
        db_paths=list(db_paths),
        allowed_statuses=allowed_statuses,
    )
    if result.get("ok"):
        return

    issues = result.get("problems", [])
    details = "\n".join(f"- {issue}" for issue in issues) if issues else "Unknown issue"
    raise SystemExit(f"[{error_prefix}] Canonical source validation failed:\n{details}")
