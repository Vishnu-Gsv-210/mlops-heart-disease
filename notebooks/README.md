# Notebooks

## `eda.ipynb`

Executable exploratory data analysis for the cleaned UCI Cleveland Heart Disease
dataset. The notebook imports the project package directly (`src.heart_disease.config`,
`src.heart_disease.data.load_clean_dataframe`) instead of copying pipeline code,
so it always analyses exactly what the training pipeline sees.

Contents, in order: intro with student details; `%matplotlib inline` setup;
`sys.path` bootstrap so it runs from the repo root or from `notebooks/`;
data load via `load_clean_dataframe()`; `df.head()`, `df.describe()`, dtypes;
class-balance table + bar plot; missing-values table; feature histograms;
correlation heatmap; and a findings section with a `FIXME-AFTER-EXECUTION`
marker for the observed numbers.

The notebook ships **unexecuted** — run it once and save with outputs so the
findings section records real values, then include the executed copy as
assignment evidence.

## Run it

```bash
# one-time: kernel + execution deps into the project venv
.venv/bin/pip install ipykernel nbclient nbformat

# launch Jupyter from the repository root
.venv/bin/jupyter notebook notebooks/eda.ipynb
# or, without a browser:
.venv/bin/jupyter lab
```

Select the "Python 3 (ipykernel)" kernel (it is the `.venv` interpreter when
Jupyter is launched from `.venv/bin`). "Run All Cells" loads
`data/processed/heart_disease_cleaned.csv` — or rebuilds it from
`data/raw/processed.cleveland.data` if missing (run
`python scripts/download_data.py` first if neither exists).

Headless execution (regenerates the notebook with outputs in place):

```bash
.venv/bin/jupyter nbconvert --to notebook --inplace --execute notebooks/eda.ipynb
```

## Script-based equivalent

`make eda` runs the same EDA through `src.heart_disease.data.generate_eda` and
writes figures to `reports/figures/` plus `reports/artifacts/eda_summary.json`.
Prefer the notebook when you need the narrated, cell-by-cell evidence view.
