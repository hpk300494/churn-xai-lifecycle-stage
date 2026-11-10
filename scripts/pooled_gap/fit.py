"""Step 1. Fit the pooled and per-stage logistic regressions exactly as the notebook does
(GridSearchCV over C in {0.1, 1, 10}, F1 scoring, stratified 5-fold, seed 42) and save them to out/fitted.pkl."""
from common import *
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
F = {}
for v in VARS:
    Xtr, ytr, Xte, yte = vdata(v)
    pipe, params = GRIDS['LogisticRegression']
    gs = GridSearchCV(pipe(), params, scoring='f1', cv=skf, n_jobs=-1, refit=True).fit(Xtr, ytr)
    F[(v, 'LogisticRegression')] = (gs.best_estimator_, gs.best_params_)
    print(f'{v:>10}  n_train={len(Xtr):5d}  best C={gs.best_params_["clf__C"]}  test ROC-AUC={roc_auc_score(yte, gs.predict_proba(Xte)[:,1]):.3f}')
joblib.dump(F, OUT / 'fitted.pkl')
