#!/usr/bin/env python3
"""Apply cull manifest to archive low-performing models."""
import json
import os
import shutil
from pathlib import Path

def main():
    manifest_path = Path.home() / "ringrift/ai-service/data/models/cull_manifest.json"
    models_dir = Path.home() / "ringrift/ai-service/models"
    archive_dir = models_dir / "archived" / "square8_2p"

    if not manifest_path.exists():
        print("NO_MANIFEST")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    archived_ids = set(m["model_id"] for m in manifest.get("archived_models", []))
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for model_id in archived_ids:
        src = models_dir / f"{model_id}.pth"
        if src.exists():
            dst = archive_dir / f"{model_id}.pth"
            shutil.move(str(src), str(dst))
            moved += 1

    remaining = len(list(models_dir.glob("*.pth")))
    print(f"MOVED:{moved} REMAINING:{remaining}")

if __name__ == "__main__":
    main()
