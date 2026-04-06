from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "uvicorn", "app.api_server:app", "--host", "127.0.0.1", "--port", "8000"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
