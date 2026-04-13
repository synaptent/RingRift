from __future__ import annotations

from pathlib import Path

from scripts.audit_import_graph import build_import_graph


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_import_graph_ignores_local_imports_by_default(tmp_path: Path) -> None:
    _write(
        tmp_path / "app" / "pkg" / "__init__.py",
        "from app.pkg.alpha import Alpha\n",
    )
    _write(
        tmp_path / "app" / "pkg" / "alpha.py",
        """
from app.pkg.beta import Beta

def lazy_only():
    from app.pkg.gamma import Gamma
    return Gamma
""".strip()
        + "\n",
    )
    _write(tmp_path / "app" / "pkg" / "beta.py", "class Beta: ...\n")
    _write(tmp_path / "app" / "pkg" / "gamma.py", "class Gamma: ...\n")

    graph, _ = build_import_graph(tmp_path, ("app",))

    assert "app.pkg.beta" in graph["app.pkg.alpha"]
    assert "app.pkg.gamma" not in graph["app.pkg.alpha"]


def test_build_import_graph_can_include_local_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "app" / "pkg" / "alpha.py",
        """
def lazy_only():
    from app.pkg.gamma import Gamma
    return Gamma
""".strip()
        + "\n",
    )
    _write(tmp_path / "app" / "pkg" / "gamma.py", "class Gamma: ...\n")

    graph, _ = build_import_graph(tmp_path, ("app",), include_local_imports=True)

    assert "app.pkg.gamma" in graph["app.pkg.alpha"]
