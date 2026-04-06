from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.satellite import build_sentinel_feature_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 tile metadata features for block groups.")
    parser.add_argument(
        "--feature-table",
        default="data/processed/block_group_features_sample.csv",
        help="Feature table containing geoid/lat/lon.",
    )
    parser.add_argument("--cache-dir", default="data/cache", help="Output cache directory.")
    parser.add_argument("--start-date", default="2024-04-01", help="Search start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2024-10-31", help="Search end date (YYYY-MM-DD).")
    parser.add_argument("--items-per-block", type=int, default=2, help="Maximum STAC items fetched per block.")
    parser.add_argument("--max-cloud-cover", type=float, default=50.0, help="Maximum allowed cloud cover.")
    parser.add_argument("--bbox-half-size-deg", type=float, default=0.01, help="Half-size of point bbox in degrees.")
    args = parser.parse_args()

    df = build_feature_table(Path(args.feature_table))
    artifacts = build_sentinel_feature_cache(
        df,
        cache_dir=Path(args.cache_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        items_per_block=args.items_per_block,
        max_cloud_cover=args.max_cloud_cover,
        bbox_half_size_deg=args.bbox_half_size_deg,
    )
    print(f"Wrote: {artifacts.features_path}")
    print(f"Wrote: {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
