"""Kaggle script kernel for the full GPU training pipeline.

This kernel is intentionally thin: it locates the uploaded project dataset,
copies it to /kaggle/working, then delegates all training/comparison/audit
work to scripts/run_kaggle_full_artifacts.sh. The shell script owns the actual
pipeline parameters, including SBERT, LightGCN, Two-Tower and strong ranker
training.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


WORKDIR = Path("/kaggle/working")
OUTPUT_ZIP = WORKDIR / "full_training_outputs.zip"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def patch_runtime_project(project_dir: Path) -> None:
    """Patch copied inputs so stale Kaggle dataset versions still run safely."""
    requirements_path = project_dir / "requirements.txt"
    if requirements_path.exists():
        text = requirements_path.read_text(encoding="utf-8")
        lines = []
        changed = False
        for line in text.splitlines():
            if line.startswith("torch>=") and "<2.6.0" not in line:
                lines.append("torch>=2.3.0,<2.6.0")
                changed = True
            elif line.startswith("sentence-transformers") and "<3.5.0" not in line:
                lines.append("sentence-transformers>=3.0.0,<3.5.0")
                changed = True
            elif line.startswith("transformers") and "<5.0.0" not in line:
                lines.append("transformers>=4.41.0,<5.0.0")
                changed = True
            else:
                lines.append(line)
        if changed:
            requirements_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print("Patched requirements.txt Kaggle-safe ML dependency pins", flush=True)

    script_path = project_dir / "scripts" / "run_kaggle_full_artifacts.sh"
    if script_path.exists():
        text = script_path.read_text(encoding="utf-8")
        install_line = '  run_step install_requirements "$PYTHON_BIN" -m pip install -q -r requirements.txt\n'
        uninstall_line = (
            '  run_step uninstall_incompatible_torch_extras "$PYTHON_BIN" '
            "-m pip uninstall -y torchvision torchaudio torchcodec\n"
        )
        if uninstall_line not in text and install_line in text:
            script_path.write_text(text.replace(install_line, install_line + uninstall_line, 1), encoding="utf-8")
            print("Patched run_kaggle_full_artifacts.sh to remove incompatible torchvision/torchaudio", flush=True)


def find_uploaded_project(kaggle_input: Path) -> Path | None:
    """Return the directory containing the recommender source tree."""
    for src_dir in sorted(kaggle_input.rglob("src")):
        if src_dir.is_dir() and (src_dir / "recommender").exists():
            return src_dir.parent
    return None


def find_project_zip(kaggle_input: Path) -> Path | None:
    """Return a zip archive that contains the project source tree."""
    for zip_path in sorted(kaggle_input.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                if any(name.startswith("src/recommender/") for name in archive.namelist()):
                    return zip_path
        except zipfile.BadZipFile:
            continue
    return None


def copy_project() -> Path:
    project_dir = WORKDIR / "movierec3"
    if project_dir.exists():
        return project_dir

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("Kaggle input contents (first 40):", flush=True)
        for path in sorted(kaggle_input.rglob("*"))[:40]:
            print(f"  {path}", flush=True)

        uploaded_project = find_uploaded_project(kaggle_input)
        if uploaded_project is not None:
            print(f"Found uploaded project at: {uploaded_project}", flush=True)
            run(["cp", "-r", str(uploaded_project), str(project_dir)])
            return project_dir

        project_zip = find_project_zip(kaggle_input)
        if project_zip is not None:
            print(f"Found uploaded project zip at: {project_zip}", flush=True)
            project_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(project_zip) as archive:
                archive.extractall(project_dir)
            return project_dir

        datasets = [path for path in sorted(kaggle_input.iterdir()) if path.is_dir()]
        if datasets:
            print(f"Fallback: copying {datasets[0]} to {project_dir}", flush=True)
            run(["cp", "-r", str(datasets[0]), str(project_dir)])
            return project_dir

    print("ERROR: project_dir not found and no uploaded project could be located.", flush=True)
    print("Working dir contents:", flush=True)
    for path in sorted(WORKDIR.rglob("*"))[:40]:
        print(f"  {path}", flush=True)
    raise SystemExit(1)


def ensure_download_zip(project_dir: Path) -> None:
    """Expose the shell pipeline zip under the submit script's documented name."""
    shell_zip = project_dir / "artifacts_and_reports_full.zip"
    if shell_zip.exists():
        shutil.copy2(shell_zip, OUTPUT_ZIP)
        print(f"Copied {shell_zip} -> {OUTPUT_ZIP}", flush=True)
        return

    print("WARNING: artifacts_and_reports_full.zip was not produced; creating fallback zip.", flush=True)
    zip_paths = [
        project_dir / "artifacts" / "movielens_pdf_clean",
        project_dir / "artifacts" / "letterboxd_pdf_clean",
        project_dir / "artifacts" / "movielens_strong",
        project_dir / "artifacts" / "letterboxd_strong",
        project_dir / "reports" / "comparison_sbert_pdf_clean_both",
        project_dir / "reports" / "comparison_sbert_strong_both",
        project_dir / "logs",
    ]
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for base_path in zip_paths:
            if not base_path.exists():
                print(f"  Skipped missing path: {base_path}", flush=True)
                continue
            for file_path in base_path.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(project_dir))
            print(f"  Zipped: {base_path.relative_to(project_dir)}", flush=True)
    print(f"Created fallback output zip: {OUTPUT_ZIP}", flush=True)


def main() -> None:
    project_dir = copy_project()
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}", flush=True)
    print(f"Contents: {sorted(os.listdir('.'))}", flush=True)

    os.environ["PYTHONPATH"] = f"{project_dir / 'src'}:{project_dir}"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("PYTHON_BIN", sys.executable)
    os.environ.setdefault("INSTALL_DEPS", "1")

    patch_runtime_project(project_dir)
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", "torchcodec"])

    script_path = project_dir / "scripts" / "run_kaggle_full_artifacts.sh"
    if not script_path.exists():
        raise SystemExit(f"Missing pipeline script: {script_path}")

    run(["bash", str(script_path)])
    ensure_download_zip(project_dir)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
