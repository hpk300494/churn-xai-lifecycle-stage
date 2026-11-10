# Lifecycle-aware explainable churn models (IBM Telco proxy)

> **Exploratory, proxy dataset.** The data are the public IBM Telco Customer Churn file, a single cross-sectional snapshot used as a proxy for B2B SaaS. Nothing here is evidence about real B2B SaaS customers, and none of it is causal.

Code for Hanaa Parvez Khan's DBA thesis at SSBM Geneva, *Predicting Customer Churn in B2B SaaS Platforms Using Explainable AI (XAI)*.

## Contents
- `notebooks/Telco_Churn_Lifecycle_XAI_Pipeline.ipynb` - the full pipeline (data prep, lifecycle stages from tenure, 5 algorithms pooled and per stage, SHAP, LIME, noise ceilings, composition checks, decision-level comparison).
- `docs/Technical_Paper_Exploratory_Proxy.pdf` - the technical paper (exploratory, proxy dataset). Tables 1-2 and Appendix G come from notebook cells 4d-14 and 4d-15, which were run in Colab on 4 November 2026; their saved output matches the paper's tables.
- `notebooks/cells/` - plain-text copies of cells 4d-14 (decision-level intervals) and 4d-15 (Kitagawa decomposition).
- `scripts/pooled_gap/` - standalone scripts for the pooled mean gap (top-1 driver +27.8, flagged overlap +2.5) and the top-three check (+16.9) in Section 3.4. Run with `bash scripts/pooled_gap/run_all.sh`; see its README. These are not notebook cells.
- `scripts/make_figures.py` - redraws the paper's Figures 1-3 from the values the notebook prints (hard-coded in the script).
- `data/README.md` - where to get the data file.

## Reproduce
1. Download the IBM Telco Customer Churn workbook (see `data/README.md`) and save it as `Telco_customer_churn 2.xlsx` next to the notebook (or edit `DATA_PATH` in the data-loading cell).
2. Open the notebook in Google Colab (or a local Jupyter with Python 3.10+). The first code cell installs `xgboost`, `shap` and `lime`; locally you can instead run `pip install -r requirements.txt`.
3. **Runtime > Run all**, top to bottom. Cells depend on earlier cells; run order matters. A full run takes a while (the matched refit bootstraps in Stage 4d-8b are the slowest).
4. Seeds are fixed in the notebook; small differences can still occur across library versions.

## Notes
- Library versions are not pinned in the notebook; `requirements.txt` gives minimum versions. Record the versions from your own run if you need exact reproduction.
- The committed notebook contains saved outputs so the numbers in the PDF can be checked without re-running.

## Citation
See `CITATION.cff`. Licence: MIT for the code (see `LICENSE`). The dataset is IBM's and is not redistributed here.

- The pooled mean gap and top-three check (Section 3.4) are not in the notebook; they are in `scripts/pooled_gap/` and were run in a sandbox, not in Colab.
