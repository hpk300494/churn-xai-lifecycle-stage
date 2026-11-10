"""Step 3a. Pooled mean gap across the four stages, from the refit replicates saved by decision.py (out/decision_out.pkl)."""
from common import OUT
import joblib, numpy as np
R = joblib.load(OUT / 'decision_out.pkl')['R']; ST = list(R)
for name, (a, b) in {'flagged overlap': ('fl_ss', 'fl_ps'), 'top-1 driver': ('dr_ss', 'dr_ps')}.items():
    G = np.mean([np.array(R[s][a]) - np.array(R[s][b]) for s in ST], axis=0)
    print(name, 'pooled mean gap %.1f [%.1f, %.1f] points' % (G.mean()*100, *np.percentile(G, [2.5, 97.5])*100))
