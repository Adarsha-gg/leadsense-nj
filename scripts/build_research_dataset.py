from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.research_data import build_research_dataset_from_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NJ-wide research feature dataset from cached ACS inputs.")
    parser.add_argument("--acs-cache", default="data/cache/acs_nj_block_groups_2022.csv")
    parser.add_argument("--out", default="data/processed/nj_research_features.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = build_research_dataset_from_cache(
        acs_cache_path=Path(args.acs_cache),
        out_path=Path(args.out),
        seed=args.seed,
    )
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
