"""Thin CLI wrapper: download the UCI Cleveland dataset and validate its schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heart_disease.data import download_data, load_raw, validate_schema


def main() -> None:
    path = download_data()
    df = load_raw(path)
    validate_schema(df)
    print(f"Dataset contract passed: {len(df)} rows, {len(df.columns) - 1} features.")


if __name__ == "__main__":
    main()
