"""Build a submission zip of the project, excluding environments and caches."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "heart_disease_project.zip"

EXCLUDED_DIRS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".git", "node_modules", ".idea", ".vscode",
    "build", "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".DS_Store", ".zip"}
EXCLUDED_FILES = {OUTPUT.name}


def iter_project_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name in EXCLUDED_FILES:
            continue
        yield relative, path


def package_project(output: Path = OUTPUT) -> Path:
    output = Path(output)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, path in iter_project_files(ROOT):
            archive.write(path, arcname=str(relative))
    return output


def main() -> None:
    output = package_project()
    count = len(zipfile.ZipFile(output).namelist())
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Packaged {count} files into {output} ({size_mb:.1f} MB)")
    print("Excluded: virtual environments, caches, git metadata, zip archives.")


if __name__ == "__main__":
    main()
