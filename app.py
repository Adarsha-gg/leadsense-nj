from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
