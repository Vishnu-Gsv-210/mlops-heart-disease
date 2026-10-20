"""End-to-end pipeline: download -> validate -> EDA -> train -> evaluate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heart_disease.config import DATA_RAW, FIGURES_DIR
from src.heart_disease.data import download_data, generate_eda, load_raw, validate_schema
from src.heart_disease.train import train_models


def main() -> None:
    try:
        raw_path = download_data()
        print(f"Downloaded fresh copy to {raw_path}")
    except Exception as exc:  # offline-friendly: reuse the existing local copy
        local = DATA_RAW / "processed.cleveland.data"
        if not local.exists():
            raise
        print(f"Download failed ({exc}); using existing local copy {local}")
        raw_path = local

    df = load_raw(raw_path)
    validate_schema(df)
    print(f"Dataset contract passed: {len(df)} rows, {len(df.columns) - 1} features.")

    eda_summary = generate_eda(df, FIGURES_DIR)
    print(f"EDA complete: {eda_summary['rows']} rows, "
          f"{eda_summary['positive_class_count']} positive cases.")

    output = train_models()
    print(f"Pipeline complete. Selected model: {output['selected_model']}")


if __name__ == "__main__":
    main()
