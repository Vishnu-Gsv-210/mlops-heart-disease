"""Tests for data acquisition, cleaning, schema validation, and EDA."""
import json

import pandas as pd
import pytest

from src.heart_disease.config import FEATURES, TARGET
from src.heart_disease.data import (
    binarize_target,
    generate_eda,
    load_clean_dataframe,
    load_raw,
    save_cleaned_dataset,
    validate_schema,
)


def _write_uci_csv(df: pd.DataFrame, path, header: bool) -> None:
    out = df.copy()
    out = out.astype(object)
    for column in out.columns:
        out[column] = out[column].where(out[column].notna(), "?")
    out.to_csv(path, index=False, header=header)


def test_load_raw_parses_headerless_uci_format(tmp_path, raw_df):
    path = tmp_path / "processed.cleveland.data"
    _write_uci_csv(raw_df, path, header=False)
    df = load_raw(path)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == len(raw_df)  # feature missingness is kept; only bad targets drop
    assert set(df[TARGET].unique()).issubset({0, 1})
    assert df[FEATURES].isna().sum().sum() > 0  # '?' became NaN and survives for imputation


def test_load_raw_parses_header_csv(tmp_path, raw_df):
    path = tmp_path / "processed.cleveland.data"
    _write_uci_csv(raw_df, path, header=True)
    df = load_raw(path)
    assert list(df.columns) == FEATURES + [TARGET]
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_binarize_target_maps_uci_num(raw_df):
    out = binarize_target(raw_df)
    assert (out.loc[out["num"] == 0, TARGET] == 0).all()
    assert (out.loc[out["num"] > 0, TARGET] == 1).all()


def test_validate_schema_accepts_clean_data(synthetic_df):
    assert validate_schema(synthetic_df) is True


def test_validate_schema_rejects_missing_column(synthetic_df):
    broken = synthetic_df.drop(columns=["chol"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(broken)


def test_validate_schema_rejects_non_binary_target(synthetic_df):
    broken = synthetic_df.copy()
    broken[TARGET] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_schema(broken)


def test_validate_schema_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        validate_schema(pd.DataFrame(columns=FEATURES + [TARGET]))


def test_clean_roundtrip(tmp_path, synthetic_df, monkeypatch):
    monkeypatch.setattr("src.heart_disease.data.DATA_PROCESSED", tmp_path)
    path = save_cleaned_dataset(synthetic_df)
    assert path.exists()
    reloaded = load_clean_dataframe(processed_path=path)
    assert len(reloaded) == len(synthetic_df)
    assert list(reloaded.columns) == list(synthetic_df.columns)


def test_load_clean_rebuilds_from_raw_when_processed_missing(tmp_path, raw_df, monkeypatch):
    monkeypatch.setattr("src.heart_disease.data.DATA_PROCESSED", tmp_path / "processed")
    raw_csv = tmp_path / "raw.csv"
    _write_uci_csv(raw_df, raw_csv, header=True)
    df = load_clean_dataframe(processed_path=tmp_path / "missing.csv", raw_path=raw_csv)
    assert list(df.columns) == FEATURES + [TARGET]
    assert (tmp_path / "processed" / "heart_disease_cleaned.csv").exists()


def test_generate_eda_writes_figures_and_summary(tmp_path, synthetic_df, monkeypatch):
    monkeypatch.setattr("src.heart_disease.data.DATA_PROCESSED", tmp_path / "processed")
    figures = tmp_path / "figures"
    summaries = tmp_path / "summaries"
    summary = generate_eda(synthetic_df, figures, summary_dir=summaries)

    for name in ("feature_distributions.png", "class_balance.png", "correlation_heatmap.png"):
        assert (figures / name).exists()
    stored = json.loads((summaries / "eda_summary.json").read_text())
    assert stored["rows"] == len(synthetic_df)
    assert stored["features"] == len(FEATURES)
    assert stored["positive_class_count"] + stored["negative_class_count"] == len(synthetic_df)
    assert summary["target_definition"] == "num == 0 -> 0; num in {1,2,3,4} -> 1"
