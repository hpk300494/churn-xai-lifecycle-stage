"""Draws Figures 1-3 of the paper from the values printed by the notebook (hard-coded below, with the cell that prints each).
Run: python scripts/make_figures.py   (needs matplotlib). Output PNGs go to the current folder."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Patch
BL, OR, GR = '#2a6f97', '#d97706', '#6b7280'
ST = ['Onboarding', 'Growth', 'Renewal', 'Maturity']

# Figure 1 -- Stage 4c (SHAP vs LIME rho) and Stage 4d-6 Part B (RF vs LR rho and noise ceiling)
sl = [0.912, 0.831, 0.984, 0.899]; rl = [0.681, 0.732, 0.803, 0.807]; ce = [0.998, 0.998, 0.999, 0.999]
x = np.arange(4); w = 0.34
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
b1 = ax.bar(x - w/2, sl, w, color=BL, label='SHAP vs LIME (same model)')
b2 = ax.bar(x + w/2, rl, w, color=OR, label='Random forest vs logistic regression (SHAP)')
ax.hlines(ce, x - w - 0.04, x + w + 0.04, colors='#222', linestyles='--', linewidth=1.2, label='Noise ceiling (RF vs LR)')
for r in list(b1) + list(b2):
    ax.text(r.get_x() + r.get_width()/2, r.get_height() - 0.03, f'{r.get_height():.2f}', ha='center', va='top', color='white', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(ST); ax.set_ylim(0.5, 1.03); ax.set_ylabel('Spearman rho (23 features)')
ax.spines[['top', 'right']].set_visible(False); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig('fig1_agreement.png'); plt.close()

# Figure 2 -- Stage 4d-15 (contract-type cells): composition and rate parts, in churn-rate points
comp = [('Onboarding to Growth', -0.086, -0.129, 21.4), ('Growth to Renewal', -0.084, -0.011, 9.5),
        ('Renewal to Maturity', -0.086, -0.025, 11.1), ('Onboarding to Maturity', -0.275, -0.146, 42.0)]
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200); y = np.arange(4)[::-1]
for yi, (n, c, r, t) in zip(y, comp):
    ax.barh(yi, -c*100, color=BL, height=0.55); ax.barh(yi, -r*100, left=-c*100, color=OR, height=0.55)
    ax.text(-c*100/2, yi, f'{-c*100:.1f}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    if -r*100 > 2.5: ax.text(-c*100 + (-r*100)/2, yi, f'{-r*100:.1f}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.text(-(c + r)*100 + 0.8, yi, f'{t:.1f} points total', va='center', fontsize=7.5, color='#222')
ax.set_yticks(y); ax.set_yticklabels([c[0] for c in comp], fontsize=8.5)
ax.set_xlabel('Drop in churn rate between stages (percentage points)', fontsize=8.5); ax.set_xlim(0, 52); ax.spines[['top', 'right']].set_visible(False)
ax.legend(handles=[Patch(color=BL, label='Composition: different mix of contract types'), Patch(color=OR, label='Rate: churn differs within the same contract type')],
          loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig('fig2_kitagawa.png'); plt.close()

# Figure 3 -- Stage 4d-14 output: (mean, low, high) for flags (A) and per-customer top driver (B)
fl = {'obs': [(.775, .625, .869), (.765, .657, .886), (.765, .647, .882), (.829, .743, .943)],
      'ss':  [(.745, .600, .876), (.664, .529, .794), (.699, .558, .853), (.791, .629, .886)],
      'ps':  [(.675, .500, .825), (.655, .500, .795), (.682, .558, .824), (.789, .685, .886)]}
dr = {'obs': [(.422, .396, .447), (.182, .163, .203), (.487, .462, .512), (.406, .380, .433)],
      'ss':  [(.718, .491, .874), (.433, .013, .959), (.800, .410, .991), (.674, .475, .894)],
      'ps':  [(.373, .164, .598), (.320, .088, .715), (.474, .259, .687), (.348, .164, .519)]}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6), dpi=200, sharey=True)
for ax, d, t in zip(axs, (fl, dr), ('Same customers flagged', 'Same top driver per customer')):
    for k, (col, off, lab) in {'ss': (GR, 0.22, 'Two refits of the stage model'), 'ps': (OR, 0.0, 'Refit pooled vs refit stage model'),
                               'obs': (BL, -0.22, 'Observed pooled vs stage (customer bootstrap)')}.items():
        for i, (m, lo, hi) in enumerate(d[k]):
            yy = len(ST) - 1 - i + off
            ax.plot([lo, hi], [yy, yy], color=col, lw=2.2, solid_capstyle='butt'); ax.plot(m, yy, 'o', color=col, ms=4.5, label=lab if i == 0 else None)
    ax.set_title(t, fontsize=9, loc='left'); ax.set_xlim(0, 1.02); ax.set_xlabel('Share in agreement', fontsize=8.5); ax.spines[['top', 'right']].set_visible(False)
    ax.set_yticks(range(len(ST))); ax.set_yticklabels(ST[::-1], fontsize=8.5)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc='lower center', ncol=1, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0))
plt.tight_layout(rect=(0, 0.15, 1, 1)); plt.savefig('fig3_decision.png')
