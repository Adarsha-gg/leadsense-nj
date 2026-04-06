from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.infrastructure import build_county_proxy_edge_list, validate_edge_list
from leadsense_nj.preprocessing import build_feature_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build infrastructure proxy edge list (hook for NJ DEP/NJGIN network edges)."
    )
    parser.add_argument(
        "--feature-table",
        default="data/processed/block_group_features_sample.csv",
        help="CSV with at least geoid/county and optional lat/lon.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/graph_edges_sample.csv",
        help="Output edge list CSV path.",
    )
    parser.add_argument(
        "--k-within-county",
        type=int,
        default=2,
        help="K nearest neighbors within each county for proxy edges.",
    )
    args = parser.parse_args()

    df = build_feature_table(Path(args.feature_table))
    edges = build_county_proxy_edge_list(df, k_within_county=args.k_within_county)
    validate_edge_list(edges)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(f"Rows: {len(edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
