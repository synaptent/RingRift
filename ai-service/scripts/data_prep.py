#!/usr/bin/env python
"""Deprecated compatibility wrapper for replay dataset export.

The historical ``data_prep.py`` entry point now delegates to
``export_replay_dataset.py``. Keep this as a real file rather than a symlink so
fresh clones and CI runners do not depend on developer-machine paths.
"""

from __future__ import annotations

from export_replay_dataset import main


if __name__ == "__main__":  # pragma: no cover - compatibility CLI
    raise SystemExit(main())
