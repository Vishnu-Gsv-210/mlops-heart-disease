"""Data acquisition, cleaning, schema validation, target binarization, and EDA."""
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.heart_disease.config import (  # noqa: E402
    ARTIFACTS_DIR,
    DATA_PROCESSED,
    DATA_RAW,
    FEATURES,
    TARGET,
)

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMN_NAMES = FEATURES + ["num"]

sns.set_theme(style="whitegrid", context="notebook")


def download_data(output_path: Path | None = None) -> Path:
    """Download the official processed Cleveland subset from UCI."""
    output_path = Path(output_path or (DATA_RAW / "processed.cleveland.data"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES, na_values="?")
    if df.empty or len(df.columns) != 14:
        raise ValueError("Downloaded UCI data has an unexpected shape.")
    df.to_csv(output_path, index=False)
    return output_path


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the UCI `num` column to a binary target: 0 stays 0, anything above 0 is 1."""
    df = df.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    return df


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load local data or download it if the local copy does not exist, then clean it."""
    path = Path(path or (DATA_RAW / "processed.cleveland.data"))
    if not path.exists():
        path = download_data(path)
    df = pd.read_csv(path, na_values=["?", ""])
    expected = set(COLUMN_NAMES)
    if not expected.issubset(df.columns):
        # Support the original headerless UCI format.
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    df = df[COLUMN_NAMES].copy()
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = binarize_target(df)
    df = df.drop(columns=["num"])
    # Remove impossible target rows; feature missingness is imputed in the pipeline.
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Lightweight dataset contract check; raises ValueError on any violation."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target must be binary.")
    return True


def save_cleaned_dataset(df: pd.DataFrame) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / "heart_disease_cleaned.csv"
    df.to_csv(path, index=False)
    return path


def load_clean_dataframe(
    processed_path: Path | str | None = None,
    raw_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the cleaned dataset, rebuilding it from raw data when absent."""
    path = Path(processed_path or (DATA_PROCESSED / "heart_disease_cleaned.csv"))
    if path.exists():
        return pd.read_csv(path)
    df = load_raw(raw_path)
    save_cleaned_dataset(df)
    return df


def generate_eda(
    df: pd.DataFrame,
    out_dir: Path | str,
    summary_dir: Path | str | None = None,
) -> dict:
    """Generate reproducible exploratory plots and a JSON summary.

    Figures are written to ``out_dir``; the summary JSON is written to
    ``summary_dir`` (default: the shared artifacts directory).
    """
    out_dir = Path(out_dir)
    summary_dir = Path(summary_dir) if summary_dir is not None else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_cleaned_dataset(df)

    # Feature distributions
    df[FEATURES].hist(figsize=(14, 10), bins=18, edgecolor="white")
    plt.suptitle("Heart Disease Dataset: Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    # Class balance
    ax = sns.countplot(data=df, x=TARGET)
    ax.set(title="Binary Target Distribution", xlabel="Heart disease (0=no, 1=yes)",
           ylabel="Records")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(out_dir / "class_balance.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    # Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(df[FEATURES + [TARGET]].corr(numeric_only=True),
                cmap="vlag", center=0, annot=True, fmt=".2f", square=False)
    plt.title("Feature and Target Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    summary = {
        "rows": len(df),
        "features": len(FEATURES),
        "positive_class_count": int(df[TARGET].sum()),
        "negative_class_count": int((df[TARGET] == 0).sum()),
        "missing_values_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows_after_cleaning": int(df.duplicated().sum()),
        "target_definition": "num == 0 -> 0; num in {1,2,3,4} -> 1",
    }
    (summary_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
