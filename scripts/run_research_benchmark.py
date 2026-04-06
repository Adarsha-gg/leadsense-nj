from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark


def _to_markdown(report: dict) -> str:
    h = report["historical"]["accuracy"]["mean"]
    f = report["fusion"]["accuracy"]["mean"]
    g = report["graph"]["accuracy"]["mean"]
    lines = [
        "# LeadSense Research Benchmark",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Folds: `{report['n_folds']}`",
        "",
        "## Accuracy (Mean +/- Std)",
        "",
        f"- Historical: `{h:.3f} +/- {report['historical']['accuracy']['std']:.3f}`",
        f"- Fusion: `{f:.3f} +/- {report['fusion']['accuracy']['std']:.3f}`",
        f"- Graph: `{g:.3f} +/- {report['graph']['accuracy']['std']:.3f}`",
        "",
        "## Improvement",
        "",
        f"- Graph - Historical accuracy: `{report['improvement_graph_over_historical_accuracy']:.3f}`",
        f"- Graph - Fusion accuracy: `{report['improvement_graph_over_fusion_accuracy']:.3f}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    df = build_feature_table()
    report = run_model_research_benchmark(df, n_splits=3, threshold=0.5, random_state=42)

    out_dir = Path("artifacts") / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_results.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
