"""Conftest for scripts library tests.

Sets up Python path to ensure scripts.lib can be imported.
"""

import sys
from pathlib import Path

# Ensure the ai-service root is on the path so scripts.lib can be imported
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))
