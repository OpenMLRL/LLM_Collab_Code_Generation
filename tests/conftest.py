"""Make this repository and a sibling CoMLRL checkout importable in tests."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
