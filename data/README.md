# Data

## Source

- **Dataset:** Heart Disease (processed Cleveland subset)
- **Publisher:** UCI Machine Learning Repository
- **DOI:** [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)
- **Records:** 303 (after cleaning; the original file has 303 rows, of which 6 contain missing values)
- **Columns:** 13 predictor features + 1 original diagnosis column (`num`)

## Download

```bash
python scripts/download_data.py
```

This fetches the official `processed.cleveland.data` from the UCI archive into
`data/raw/` and validates the schema contract. The raw file is committed to
`data/raw/` only as a local convenience copy; the UCI archive is the source of
truth. The cleaned output below is regenerated, not hand-edited:

```bash
python -c "from src.heart_disease.config import FIGURES_DIR; from src.heart_disease.data import generate_eda, load_clean_dataframe; generate_eda(load_clean_dataframe(), FIGURES_DIR)"
```

- Raw copy: `data/raw/processed.cleveland.data`
- Cleaned output: `data/processed/heart_disease_cleaned.csv` (produced by the pipeline)

## Schema (14 columns: 13 features + `num`)

| Column | Type | Description | Values / units |
|---|---|---|---|
| `age` | numeric | Age in years | years |
| `sex` | categorical | Sex | 1 = male, 0 = female |
| `cp` | categorical | Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| `trestbps` | numeric | Resting systolic blood pressure | mm Hg on admission |
| `chol` | numeric | Serum cholesterol | mg/dl |
| `fbs` | categorical | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| `restecg` | categorical | Resting electrocardiographic result | 0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy |
| `thalach` | numeric | Maximum heart rate achieved | bpm |
| `exang` | categorical | Exercise-induced angina | 1 = yes, 0 = no |
| `oldpeak` | numeric | ST depression induced by exercise relative to rest | mm |
| `slope` | categorical | Slope of the peak exercise ST segment | 1 = upsloping, 2 = flat, 3 = downsloping |
| `ca` | numeric | Major vessels colored by fluoroscopy | 0–3 |
| `thal` | categorical | Thallium stress test result | 3 = normal, 6 = fixed defect, 7 = reversible defect |
| `num` | target (original) | Diagnosis of heart disease | 0 = absence, 1–4 = presence, graded |

## Target binarization

The pipeline converts the graded `num` column to a binary target `target`
(see `src/heart_disease/data.py::binarize_target`):

- `num == 0` -> `0` (heart disease **not** indicated)
- `num > 0` (1, 2, 3, 4) -> `1` (heart disease present)

The `num` column is dropped after binarization; `heart_disease_cleaned.csv`
therefore carries the 13 features + `target`.

## Missing-value convention

The original UCI file encodes missing values as the literal character `?`
(they occur in `ca` and `thal`). The loaders read the file with
`na_values="?"`, so these become `NaN` in pandas. Feature missingness is left in
the cleaned frame and handled inside the model pipeline by imputation (median
for numeric features, most-frequent for categorical features); rows with an
unusable target are dropped.

## Citation

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Educational use only: this is a small, historical dataset and nothing derived
from it is a medical conclusion.
