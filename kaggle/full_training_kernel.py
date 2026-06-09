"""Kaggle script kernel for full GPU training pipeline.

Runs the complete run_kaggle_full_artifacts.sh pipeline:
- Train PDF-clean artifacts (SBERT + LightGCN + Two-Tower) for MovieLens + Letterboxd
- Train strong ranker artifacts (LightGBM) for both datasets
- Run comparison suite (core + full) for both datasets
- Audit and zip all outputs

Submitted by scripts/submit_kaggle_training.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


WORKDIR = Path("/kaggle/working")


def run(command: list[str] | str, shell: bool = False, **kwargs) -> None:
    if isinstance(command, list):
        print("+", " ".join(command))
    else:
        print("+", command)
    subprocess.run(command, check=True, shell=shell, **kwargs)


def main() -> None:
    project_dir = WORKDIR / "movierec3"

    # Detect if running from Kaggle dataset input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("Kaggle input contents (first 30):")
        for p in sorted(kaggle_input.rglob("*"))[:30]:
            print(f"  {p}")

        # Find the directory that contains src/ (could be nested)
        src_dir = None
        for p in kaggle_input.rglob("src"):
            if p.is_dir() and (p / "recommender").exists():
                src_dir = p.parent
                break

        if src_dir and not project_dir.exists():
            print(f"Found project at: {src_dir}")
            run(["cp", "-r", str(src_dir), str(project_dir)])
        elif not project_dir.exists():
            # Fallback: just copy the first dataset dir
            datasets = [d for d in kaggle_input.iterdir() if d.is_dir()]
            if datasets:
                print(f"Fallback: copying {datasets[0]} to {project_dir}")
                run(["cp", "-r", str(datasets[0]), str(project_dir)])

    if not project_dir.exists():
        print("ERROR: project_dir not found at", project_dir)
        print("Working dir contents:")
        for p in sorted(WORKDIR.rglob("*"))[:30]:
            print(f"  {p}")
        sys.exit(1)

    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    print(f"Contents: {sorted(os.listdir('.'))}")
    os.environ["PYTHONPATH"] = f"{project_dir / 'src'}:{project_dir}"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Install dependencies
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    if Path("requirements-optional.txt").exists():
        # Install optional deps one-by-one (some may fail, e.g. lightfm needs C compiler)
        with open("requirements-optional.txt") as f:
            for line in f:
                pkg = line.strip()
                if pkg and not pkg.startswith("#"):
                    try:
                        run([sys.executable, "-m", "pip", "install", "-q", pkg])
                    except Exception as e:
                        print(f"WARNING: Failed to install optional dep '{pkg}': {e}")

    # Check CUDA
    run([sys.executable, "-c", """
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
else:
    print('WARNING: No GPU, will use CPU (slower)')
"""])

    device = "cuda" if _has_cuda() else "cpu"
    epochs = 100 if device == "cuda" else 20
    batch_size = 8192 if device == "cuda" else 4096
    dim = 128
    grid_step = 0.05
    # Use tfidf to avoid CUDA/SBERT kernel mismatch on Kaggle
    content_backend = "tfidf"

    # ---- PDF-Clean Artifacts ----
    print("\n" + "=" * 60)
    print("PHASE 1: Train PDF-Clean Artifacts")
    print("=" * 60)

    # MovieLens PDF-clean
    try:
        run([
            sys.executable, "-u", "scripts/train.py",
            "--raw-dir", "data/raw/ml-latest-small",
            "--enriched-catalog", "data/processed/movie_catalog_enriched.parquet",
            "--artifacts-dir", "artifacts/movielens_pdf_clean",
            "--content-backend", content_backend,
            "--train-lightgcn", "--train-two-tower",
            "--lightgcn-dim", str(dim), "--lightgcn-layers", "3",
            "--epochs", str(epochs), "--batch-size", str(batch_size),
            "--device", device, "--hybrid-grid-step", str(grid_step),
            "--min-rating", "4.0",
        ])
    except Exception as e:
        print(f"WARNING: MovieLens PDF-clean failed: {e}")

    # Letterboxd PDF-clean
    try:
        run([
            sys.executable, "-u", "scripts/train.py",
            "--raw-dir", "data/processed/letterboxd",
            "--enriched-catalog", "data/processed/letterboxd/movie_catalog_enriched.parquet",
            "--artifacts-dir", "artifacts/letterboxd_pdf_clean",
            "--content-backend", content_backend,
            "--train-lightgcn", "--train-two-tower",
            "--lightgcn-dim", str(dim), "--lightgcn-layers", "3",
            "--epochs", str(epochs), "--batch-size", str(batch_size),
            "--device", device, "--hybrid-grid-step", str(grid_step),
            "--min-rating", "4.0",
        ])
    except Exception as e:
        print(f"WARNING: Letterboxd PDF-clean failed: {e}")

    # ---- Strong Ranker Artifacts ----
    print("\n" + "=" * 60)
    print("PHASE 2: Train Strong Ranker Artifacts")
    print("=" * 60)

    for ds_name, raw_dir, catalog in [
        ("movielens", "data/raw/ml-latest-small", "data/processed/movie_catalog_enriched.parquet"),
        ("letterboxd", "data/processed/letterboxd", "data/processed/letterboxd/movie_catalog_enriched.parquet"),
    ]:
        try:
            run([
                sys.executable, "-u", "scripts/train_strong_hybrid.py",
                "--dataset", ds_name,
                "--raw-dir", raw_dir,
                "--enriched-catalog", catalog,
                "--artifacts-dir", f"artifacts/{ds_name}_strong",
                "--content-backend", content_backend,
                "--ranker", "lightgbm",
                "--lightgcn-dim", str(dim), "--lightgcn-layers", "3",
                "--lightgcn-epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--device", device,
                "--max-ease-items", "5000",
                "--max-ranker-samples", "500000",
                "--min-rating", "4.0",
            ])
        except Exception as e:
            print(f"WARNING: {ds_name} strong ranker failed: {e}")

    # ---- Comparison Suite ----
    print("\n" + "=" * 60)
    print("PHASE 3: Comparison Suite")
    print("=" * 60)

    # PDF-clean core comparison (both datasets)
    try:
        run([
            sys.executable, "-u", "scripts/compare_models.py",
            "--dataset", "both",
            "--movielens-dir", "data/raw/ml-latest-small",
            "--movielens-enriched-catalog", "data/processed/movie_catalog_enriched.parquet",
            "--letterboxd-dir", "data/processed/letterboxd",
            "--letterboxd-enriched-catalog", "data/processed/letterboxd/movie_catalog_enriched.parquet",
            "--content-backend", content_backend,
            "--models", "core",
            "--k", "10",
            "--epochs", str(epochs),
            "--mf-dim", str(dim),
            "--batch-size", str(batch_size),
            "--device", device,
            "--hybrid-grid-step", str(grid_step),
            "--output-dir", "reports/comparison_gpu_core",
        ])
    except Exception as e:
        print(f"WARNING: Core comparison failed: {e}")

    # Strong full comparison (both datasets)
    try:
        run([
            sys.executable, "-u", "scripts/compare_models.py",
            "--dataset", "both",
            "--movielens-dir", "data/raw/ml-latest-small",
            "--movielens-enriched-catalog", "data/processed/movie_catalog_enriched.parquet",
            "--letterboxd-dir", "data/processed/letterboxd",
            "--letterboxd-enriched-catalog", "data/processed/letterboxd/movie_catalog_enriched.parquet",
            "--content-backend", content_backend,
            "--models", "full",
            "--k", "10",
            "--epochs", str(epochs),
            "--mf-dim", str(dim),
            "--batch-size", str(batch_size),
            "--device", device,
            "--max-ease-items", "5000",
            "--max-slim-items", "3000",
            "--max-ranker-samples", "500000",
            "--hybrid-grid-step", str(grid_step),
            "--output-dir", "reports/comparison_gpu_full",
        ])
    except Exception as e:
        print(f"WARNING: Full comparison failed: {e}")

    # ---- Visualization ----
    print("\n" + "=" * 60)
    print("PHASE 4: Embedding Visualization")
    print("=" * 60)

    for art_dir, out_dir in [
        ("artifacts/movielens_pdf_clean", "reports/embedding_visualization_movielens"),
        ("artifacts/letterboxd_pdf_clean", "reports/embedding_visualization_letterboxd"),
    ]:
        if Path(art_dir).exists():
            run([
                sys.executable, "-u", "scripts/visualize_embeddings.py",
                "--artifacts-dir", art_dir,
                "--output-dir", out_dir,
                "--method", "tsne",
                "--sample-size", "2500",
                "--top-genres", "8",
            ])

    # ---- Audit ----
    print("\n" + "=" * 60)
    print("PHASE 5: Audit Artifacts")
    print("=" * 60)
    run([sys.executable, "-u", "scripts/audit_artifacts.py"])

    # ---- Zip outputs ----
    print("\n" + "=" * 60)
    print("PHASE 6: Zip Outputs")
    print("=" * 60)
    zip_paths = [
        "artifacts/movielens_pdf_clean",
        "artifacts/letterboxd_pdf_clean",
        "artifacts/movielens_strong",
        "artifacts/letterboxd_strong",
        "reports/comparison_sbert_pdf_clean_both",
        "reports/comparison_sbert_strong_both",
        "reports/embedding_visualization_movielens",
        "reports/embedding_visualization_letterboxd",
    ]
    output_zip = WORKDIR / "full_training_outputs.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base_path in zip_paths:
            p = Path(base_path)
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        zf.write(f, str(f))
                print(f"  Zipped: {base_path}")
            else:
                print(f"  Skipped (missing): {base_path}")

    print(f"\nAll outputs zipped to: {output_zip}")
    print("DONE!")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
