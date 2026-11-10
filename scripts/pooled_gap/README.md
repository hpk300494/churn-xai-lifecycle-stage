# Pooled mean gap and top-three check (paper Section 3.4)

These scripts reproduce three numbers in the paper that are **not** produced by the notebook cells:

| Quantity | Value in paper |
|---|---|
| Pooled top-1 driver gap (stage-specific vs pooled, minus refit floor) | +27.8 points [7.7, 46.6] |
| Pooled flagged-customer overlap gap | +2.5 points [-7.5, 11.2] |
| Pooled top-3 driver gap (sensitivity check) | +16.9 points [4.6, 29.9] |

They use a copy of the notebook's data preparation (`common.py`: same 80/20 split, seed 42, engineered features, tenure-quartile stages) and the notebook's logistic-regression grid (`fit.py`). They are standalone scripts, not notebook cells, and have been run in a sandbox, not in Colab.

## Run
```
pip install -r requirements.txt
TELCO_XLSX="/path/to/Telco_customer_churn 2.xlsx" bash scripts/pooled_gap/run_all.sh
```
If `TELCO_XLSX` is not set, the scripts look for `Telco_customer_churn 2.xlsx` in the repository root. Intermediate files go to `scripts/pooled_gap/out/` (override with `POOLED_GAP_OUT`).

Steps: `fit.py` (fit models) -> `decision.py` (200 bootstrap refits per stage, the slow step, a few minutes) -> `pooled_gap.py` (pooled gaps). `top3.py` needs only `fit.py`.

`expected_output.txt` holds the printed results of the run that produced the paper's numbers. Seeds are fixed; small differences are possible across library versions.
