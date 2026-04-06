from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.ingestion import build_real_data_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real NJ ACS + EPA/SDWIS data into local cache.")
    parser.add_argument("--cache-dir", default="data/cache", help="Directory for cached CSV/JSON artifacts.")
    parser.add_argument("--acs-year", type=int, default=2022, help="ACS 5-year dataset year.")
    parser.add_argument(
        "--max-violation-rows",
        type=int,
        default=30000,
        help="Maximum lead violation rows to fetch from EPA efservice.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60, help="HTTP timeout per request.")
    args = parser.parse_args()

    artifacts = build_real_data_cache(
        cache_dir=Path(args.cache_dir),
        acs_year=args.acs_year,
        max_violation_rows=args.max_violation_rows,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"Wrote: {artifacts.acs_path}")
    print(f"Wrote: {artifacts.epa_lcr_samples_path}")
    print(f"Wrote: {artifacts.epa_lcr_sample_results_path}")
    print(f"Wrote: {artifacts.epa_violations_path}")
    print(f"Wrote: {artifacts.epa_pws_summary_path}")
    print(f"Wrote: {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
