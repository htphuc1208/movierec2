from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import leaderboard_row, write_leaderboard


def rows_from_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = str(manifest.get("dataset", ""))
    model = str(manifest.get("model_name", manifest_path.parent.name))
    source = str(manifest.get("model_source", "artifact"))
    family = model_family(model, source)
    metrics = manifest.get("metrics", {})
    return [
        leaderboard_row(
            dataset=dataset,
            model=model,
            model_family=family,
            source=source,
            metrics=metrics,
            artifact_dir=str(manifest_path.parent),
            tuned_weights=manifest.get("weights", {}),
        )
    ]


def rows_from_recbole_report(path: str | Path, dataset: str, artifact_root: str | Path = "") -> list[dict[str, Any]]:
    report_path = Path(path)
    if not report_path.exists():
        return []
    results = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for result in results:
        model = str(result.get("model", ""))
        artifact_dir = str(result.get("artifact_dir") or "")
        if not artifact_dir and artifact_root:
            inferred = Path(artifact_root) / f"recbole-{model.lower()}"
            if (inferred / "manifest.json").exists():
                artifact_dir = str(inferred)
        rows.append(
            leaderboard_row(
                dataset=dataset,
                model=f"recbole-{model}",
                model_family=model.lower(),
                source="recbole",
                metrics=result.get("test_result", {}),
                artifact_dir=artifact_dir,
                command=str(result.get("command", "")),
            )
        )
    return rows


def model_family(model: str, source: str) -> str:
    text = f"{model} {source}".lower()
    if "lightgcn" in text:
        return "lightgcn"
    if "two" in text and "tower" in text:
        return "two_tower"
    if "bpr" in text:
        return "bpr"
    if "svd" in text:
        return "svd"
    if "itemknn" in text:
        return "itemknn"
    if "pop" in text:
        return "popularity"
    return "hybrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized recommender leaderboard from artifacts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--recbole-report", default="")
    parser.add_argument("--recbole-artifact-root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for manifest in args.manifest:
        rows.extend(rows_from_manifest(manifest))
    if args.recbole_report:
        rows.extend(rows_from_recbole_report(args.recbole_report, args.dataset, args.recbole_artifact_root))
    write_leaderboard(rows, args.output_prefix)


if __name__ == "__main__":
    main()
