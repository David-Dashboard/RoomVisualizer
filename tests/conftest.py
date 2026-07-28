import sys
from pathlib import Path

# Make both the package and the synthetic-scene helper importable regardless of
# where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
