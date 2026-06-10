#!/usr/bin/env python3
"""Submit full GPU training pipeline as a Kaggle script kernel.

Usage:
    PYTHONPATH=src:. python scripts/submit_kaggle_training.py --username <kaggle_username>

This uploads the project code + data as a Kaggle dataset (if not already
uploaded) and submits a script kernel that runs the complete training pipeline.

After the kernel finishes (~1-2 hours on GPU), download outputs with:
    kaggle kernels output <username>/movierec3-full-training -p kaggle_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from recommender.config import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit full training pipeline to Kaggle GPU")
    parser.add_argument("--username", default=os.getenv("KAGGLE_USERNAME"),
                        help="Kaggle username/owner")
    parser.add_argument("--kernel-slug", default="movierec3-full-training")
    parser.add_argument("--kernel-title", default="MovieRec3 Full Training")
    parser.add_argument(
        "--kernel-file",
        type=Path,
        default=PROJECT_ROOT / "kaggle" / "full_training_kernel.py",
        help="Kernel script to submit. Defaults to the full training kernel.",
    )
    parser.add_argument("--dataset-slug", default="movierec3-input")
    parser.add_argument("--skip-dataset-upload", action="store_true",
                        help="Skip dataset upload, assume already uploaded")
    parser.add_argument("--no-submit", action="store_true",
                        help="Only create staging directory, don't submit")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Submit without GPU (for testing)")
    return parser.parse_args()


def kaggle_command() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def check_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    has_file = (
        (kaggle_dir / "kaggle.json").exists()
        or (kaggle_dir / "credentials.json").exists()
        or (kaggle_dir / "access_token").exists()
    )
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    has_token_env = bool(os.getenv("KAGGLE_API_TOKEN"))
    if not has_file and not has_env and not has_token_env:
        raise RuntimeError("Missing Kaggle credentials.")


def upload_dataset(username: str, dataset_slug: str) -> str:
    """Upload or update the movierec3 input dataset to Kaggle."""
    zip_path = PROJECT_ROOT / "movierec3_kaggle_input.zip"
    if not zip_path.exists():
        raise RuntimeError(f"Missing {zip_path}. Create it first.")

    staging = Path(tempfile.mkdtemp(prefix="kaggle_ds_"))
    try:
        shutil.copy2(zip_path, staging / "movierec3_kaggle_input.zip")

        metadata = {
            "title": dataset_slug,
            "id": f"{username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        cmd = [*kaggle_command(), "datasets", "create", "-p", str(staging), "-q"]
        print("+", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        output = result.stderr + result.stdout
        output_lower = output.lower()
        if result.returncode == 0 and "dataset creation error" not in output_lower:
            print(result.stdout)
        elif any(text in output_lower for text in ("already exists", "already in use", "requested title")):
            print("Dataset already exists, updating version...")
            cmd = [*kaggle_command(), "datasets", "version", "-p", str(staging),
                   "-m", "Updated from submit_kaggle_training", "-q"]
            print("+", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            result.check_returncode()

        return f"{username}/{dataset_slug}"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def stage_kernel(args: argparse.Namespace, staging_dir: Path, dataset_source: str) -> None:
    if not args.username:
        raise RuntimeError("Missing --username or KAGGLE_USERNAME")

    kernel_file = args.kernel_file
    if not kernel_file.is_absolute():
        kernel_file = PROJECT_ROOT / kernel_file
    if not kernel_file.exists():
        raise RuntimeError(f"Missing kernel file: {kernel_file}")
    shutil.copy2(kernel_file, staging_dir / kernel_file.name)

    metadata = {
        "id": f"{args.username}/{args.kernel_slug}",
        "title": args.kernel_title,
        "code_file": kernel_file.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        "enable_internet": True,
        "dataset_sources": [dataset_source],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not args.username:
        raise SystemExit("ERROR: --username required or set KAGGLE_USERNAME env var")

    check_credentials()

    # Step 1: Upload dataset
    dataset_source = f"{args.username}/{args.dataset_slug}"
    if not args.skip_dataset_upload:
        print("=" * 60)
        print("Step 1: Upload dataset to Kaggle")
        print("=" * 60)
        dataset_source = upload_dataset(args.username, args.dataset_slug)
        print(f"Dataset: {dataset_source}")
    else:
        print(f"Skipping dataset upload, using: {dataset_source}")

    # Step 2: Stage and submit kernel
    print()
    print("=" * 60)
    print("Step 2: Submit training kernel")
    print("=" * 60)

    staging_parent = PROJECT_ROOT / "build" if args.keep_staging or args.no_submit else None
    if staging_parent:
        staging_parent.mkdir(exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix="kaggle_train_", dir=staging_parent))
    else:
        staging_dir = Path(tempfile.mkdtemp(prefix="kaggle_train_"))

    try:
        stage_kernel(args, staging_dir, dataset_source)
        print(f"Staged kernel at: {staging_dir}")

        if args.no_submit:
            print("--no-submit: stopping here")
            return

        cmd = [*kaggle_command(), "kernels", "push", "-p", str(staging_dir)]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)

        print()
        print("=" * 60)
        print(f"✅ Submitted kernel: {args.username}/{args.kernel_slug}")
        print()
        print("Monitor progress:")
        print(f"  kaggle kernels status {args.username}/{args.kernel_slug}")
        print()
        print("Download outputs after completion:")
        print(f"  kaggle kernels output {args.username}/{args.kernel_slug} -p kaggle_outputs")
        print()
        print("Then extract locally:")
        print("  unzip kaggle_outputs/full_training_outputs.zip -d .")
        print("=" * 60)
    finally:
        if not args.keep_staging and not args.no_submit:
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
