# Results

Tables and figures reported in the paper (`../paper/churn_lifecycle_stages_paper.pdf`).

**Provenance:** these CSVs and PNGs were transcribed/extracted from the paper's own tables and figures. They are provided for convenience; the notebook in `../notebooks/` is the code that produces the underlying numbers. The pooled mean gap and top-three check scripts are not yet in the notebook (see the top-level README).

## Tables (`tables/`)

| File | Content |
|---|---|
| `table1_flagged_customer_overlap.csv` | Table 1 (Sec. 3.4): overlap of flagged customers, stage model vs pooled model |
| `table2_top_driver_agreement.csv` | Table 2 (Sec. 3.4): share of customers with the same top driver, pooled vs stage model |
| `tableA_stage_vs_pooled_differences.csv` | Appendix A: stage vs pooled model differences (CV and test) |
| `tableB_stage_by_feature_interactions.csv` | Appendix B: likelihood-ratio tests of stage-by-feature interactions |
| `tableC_composition_matched_rank_comparisons.csv` | Appendix C: composition-matched rank comparisons |
| `tableD_leading_drivers_by_stage.csv` | Appendix D: leading SHAP drivers by stage with top-3 refit stability |
| `tableE_evidence_map.csv` | Appendix E (Table A1): which test backs which claim |
| `tableF_best_model_performance.csv` | Appendix F (Table A2): test-set performance of the best model per variant |
| `tableG_kitagawa_decomposition.csv` | Appendix G (Table A3): share of the stage-to-stage fall in churn due to composition |

## Figures (`figures/`)

| File | Content |
|---|---|
| `fig1_method_vs_algorithm_agreement.png` | Figure 1: agreement between explanation methods on one model vs. agreement between algorithms, against the noise ceiling |
| `fig2_kitagawa_decomposition.png` | Figure 2: Kitagawa decomposition of the fall in churn across stages |
| `fig3_pooled_vs_stage_agreement.png` | Figure 3: observed pooled-vs-stage agreement against refit agreement (flagged customers and top driver) |
