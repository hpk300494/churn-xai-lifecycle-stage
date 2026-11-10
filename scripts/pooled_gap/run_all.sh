#!/usr/bin/env bash
# Reproduces the pooled gaps in Section 3.4 (top-1 driver and flagged-customer overlap) and the top-3 sensitivity check.
# Usage:  TELCO_XLSX="/path/to/Telco_customer_churn 2.xlsx" bash scripts/pooled_gap/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
python fit.py          # pooled + per-stage logistic regressions -> out/fitted.pkl
python decision.py     # 200 bootstrap refits per stage -> out/decision_out.pkl (slowest step)
python pooled_gap.py   # pooled mean gap: top-1 driver and flagged overlap
python top3.py         # top-3 driver sensitivity check
