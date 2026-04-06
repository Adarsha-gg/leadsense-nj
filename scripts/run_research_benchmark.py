from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leadsense_nj.preprocessing import build_feature_table
from leadsense_nj.research import run_model_research_benchmark


def _to_markdown(report: dict, *, dataset_path: str) -> str:
    h = report["historical"]["accuracy"]["mean"]
    f = report["fusion"]["accuracy"]["mean"]
    g = report["graph"]["accuracy"]["mean"]
    g_knn = report.get("graph_knn", {}).get("accuracy", {}).get("mean", g)
    g_infra = report.get("graph_infrastructure", {}).get("accuracy", {}).get("mean", g)
    ablation_rows = [
        "| Model | Accuracy (mean +/- std) | AUROC | AUPRC |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("ablation_accuracy_table", []):
        ablation_rows.append(
            "| {model} | {acc_mean:.3f} +/- {acc_std:.3f} | {auroc:.3f} | {auprc:.3f} |".format(
                model=row["model"],
                acc_mean=row["accuracy_mean"],
                acc_std=row["accuracy_std"],
                auroc=row["auroc_mean"],
                auprc=row["auprc_mean"],
            )
        )

    lines = [
        "# LeadSense Research Benchmark",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Dataset: `{dataset_path}`",
        f"- Rows used: `{report.get('n_rows')}`",
        f"- Folds: `{report['n_folds']}`",
        f"- Fold overlap count: `{report.get('split_integrity', {}).get('fold_overlap_count', 'n/a')}`",
        "",
        "## Accuracy (Mean +/- Std)",
        "",
        f"- Historical: `{h:.3f} +/- {report['historical']['accuracy']['std']:.3f}`",
        f"- Fusion: `{f:.3f} +/- {report['fusion']['accuracy']['std']:.3f}`",
        f"- Graph: `{g:.3f} +/- {report['graph']['accuracy']['std']:.3f}`",
        f"- Graph (KNN): `{g_knn:.3f}`",
        f"- Graph (Infrastructure): `{g_infra:.3f}`",
        "",
        "## Graph Model Ranking Metrics",
        "",
        f"- Graph AUROC: `{report['graph']['auroc']['mean']:.3f} +/- {report['graph']['auroc']['std']:.3f}`",
        f"- Graph AUPRC: `{report['graph']['auprc']['mean']:.3f} +/- {report['graph']['auprc']['std']:.3f}`",
        "",
        "## Improvement",
        "",
        f"- Graph - Historical accuracy: `{report['improvement_graph_over_historical_accuracy']:.3f}`",
        f"- Graph - Fusion accuracy: `{report['improvement_graph_over_fusion_accuracy']:.3f}`",
        f"- Infrastructure Graph - KNN Graph accuracy: `{report.get('improvement_graph_infrastructure_over_graph_knn_accuracy', 0.0):.3f}`",
        "",
        "## Ablation Table",
        "",
        *ablation_rows,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LeadSense benchmark and write JSON/Markdown artifacts.")
    parser.add_argument(
        "--dataset",
        default="data/processed/nj_research_features.csv",
        help="Input CSV dataset for benchmark.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2500,
        help="Optional subsample size for compute-bounded runs; set <=0 for full dataset.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = Path("data/processed/block_group_features_sample.csv")
    df = build_feature_table(dataset_path)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    report = run_model_research_benchmark(
        df,
        n_splits=args.n_splits,
        threshold=args.threshold,
        random_state=args.random_state,
        max_rows=max_rows,
    )

    out_dir = Path("artifacts") / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_results.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report, dataset_path=str(dataset_path)), encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
