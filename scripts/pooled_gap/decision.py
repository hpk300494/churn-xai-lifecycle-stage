"""Step 2. Refit-noise-floor and paired-gap replicates for the decision-level comparison (200 bootstrap refits per stage; slowest step)."""
from common import *
import joblib, time
from sklearn.base import clone
F=joblib.load(OUT / 'fitted.pkl')
pooled_proto=F[('Pooled','LogisticRegression')][0]
stage_proto={s:F[(s,'LogisticRegression')][0] for s in ST}
Xp,yp,_,_=vdata('Pooled')
FLAG=0.10
def raw_of(n):
    rest=n[len('cat__'):] if n.startswith('cat__') else None
    if n.startswith('num__'): return n[len('num__'):]
    return next((c for c in CAT if rest.startswith(c)),rest)
def lin_shap(pipe,X,bg):
    pr,clf=pipe.named_steps['prep'],pipe.named_steps['clf']
    T=pr.transform(X); T=T.toarray() if hasattr(T,'toarray') else T
    B=pr.transform(bg); B=B.toarray() if hasattr(B,'toarray') else B
    phi=np.abs(clf.coef_[0][None,:]*(T-B.mean(0)))
    names=[raw_of(n) for n in pr.get_feature_names_out()]
    out={}
    for n,col in zip(names,phi.T): out[n]=out.get(n,0)+col
    keys=sorted(out); return np.column_stack([out[k] for k in keys]),keys
def top1(pipe,X,bg):
    M,keys=lin_shap(pipe,X,bg); return np.array(keys)[M.argmax(1)]
def flags(p,share=FLAG):
    r=pd.Series(p).rank(pct=True).values; return r>(1-share)
def boot_fit(proto,X,y,rng):
    pos,neg=np.where(y.values==1)[0],np.where(y.values==0)[0]
    ii=np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))])
    return clone(proto).fit(X.iloc[ii],y.iloc[ii])
rng=np.random.default_rng(RANDOM_STATE)
B_REFIT=200; B_CUST=2000
out={}
obs={}; 
data={}
for s in ST:
    Xtr,ytr,Xte,yte=vdata(s)
    t1p=top1(pooled_proto,Xtr,Xtr); t1s=top1(stage_proto[s],Xtr,Xtr)
    sp=pooled_proto.predict_proba(Xte)[:,1]; ss=stage_proto[s].predict_proba(Xte)[:,1]
    hp,hs=flags(sp),flags(ss)
    obs[s]=dict(n_test=len(Xte),n_train=len(Xtr),flagged=int(hs.sum()),same=int((hp&hs).sum()),share=(hp&hs).sum()/hs.sum(),
                caught_stage=int(yte.values[hs].sum()),caught_pooled=int(yte.values[hp].sum()),churners=int(yte.sum()),agree1=(t1p==t1s).mean())
    data[s]=(Xtr,ytr,Xte,yte,t1p,t1s,sp,ss)
# (a) customer-level bootstrap with fixed models
cb={}
for s in ST:
    Xtr,ytr,Xte,yte,t1p,t1s,sp,ss=data[s]
    sh=[];ag=[]
    for _ in range(B_CUST):
        i=rng.integers(0,len(Xte),len(Xte)); a,b=flags(sp[i]),flags(ss[i]); sh.append((a&b).sum()/b.sum())
        j=rng.integers(0,len(Xtr),len(Xtr)); ag.append((t1p[j]==t1s[j]).mean())
    cb[s]=dict(share=np.percentile(sh,[2.5,97.5]),agree=np.percentile(ag,[2.5,97.5]))
# (b) refit distributions
R={s:{k:[] for k in ('fl_ss','fl_pp','fl_ps','dr_ss','dr_pp','dr_ps','lv_ss','lv_ps','rho_ss','rho_ps')} for s in ST}
from scipy.stats import spearmanr
t0=time.time()
for b in range(B_REFIT):
    p1,p2=boot_fit(pooled_proto,Xp,yp,rng),boot_fit(pooled_proto,Xp,yp,rng)
    for s in ST:
        Xtr,ytr,Xte,yte=data[s][:4]
        s1,s2=boot_fit(stage_proto[s],Xtr,ytr,rng),boot_fit(stage_proto[s],Xtr,ytr,rng)
        F_={k:flags(m.predict_proba(Xte)[:,1]) for k,m in (('p1',p1),('p2',p2),('s1',s1),('s2',s2))}
        T_={k:top1(m,Xtr,Xtr) for k,m in (('p1',p1),('p2',p2),('s1',s1),('s2',s2))}
        MS={k:lin_shap(m,Xtr,Xtr) for k,m in (('p1',p1),('p2',p2),('s1',s1),('s2',s2))}
        keys=MS['s1'][1]; MV={k:v[0].mean(0) for k,v in MS.items()}
        R[s]['lv_ss'].append(float(MV['s1'].argmax()==MV['s2'].argmax())); R[s]['lv_ps'].append(float(MV['s1'].argmax()==MV['p1'].argmax()))
        R[s]['rho_ss'].append(spearmanr(MV['s1'],MV['s2'])[0]); R[s]['rho_ps'].append(spearmanr(MV['s1'],MV['p1'])[0])
        ov=lambda a,c:(a&c).sum()/a.sum()
        R[s]['fl_ss'].append(ov(F_['s1'],F_['s2'])); R[s]['fl_pp'].append(ov(F_['p1'],F_['p2'])); R[s]['fl_ps'].append(ov(F_['s1'],F_['p1']))
        R[s]['dr_ss'].append((T_['s1']==T_['s2']).mean()); R[s]['dr_pp'].append((T_['p1']==T_['p2']).mean()); R[s]['dr_ps'].append((T_['s1']==T_['p1']).mean())
    if b%25==0: print('refit',b,round(time.time()-t0),'s',flush=True)
joblib.dump(dict(obs=obs,cb=cb,R=R),OUT / 'decision_out.pkl')
pct=lambda v:(np.mean(v),*np.percentile(v,[2.5,97.5]))
print('\nFLAGGED (LR stage vs LR pooled), top 10% of stage test customers')
for s in ST:
    o=obs[s];c=cb[s]
    gap=np.array(R[s]['fl_ss'])-np.array(R[s]['fl_ps'])
    print(f"{s:10} flagged {o['flagged']:3d} same {o['same']:3d} share {o['share']:.3f} cust-boot CI [{c['share'][0]:.3f},{c['share'][1]:.3f}] | caught {o['caught_stage']}/{o['caught_pooled']} of {o['churners']}")
    print('   refit floors  stage-vs-stage %.3f [%.3f,%.3f]  pooled-vs-pooled %.3f [%.3f,%.3f]  pooled-vs-stage(refit) %.3f [%.3f,%.3f]  gap(ss-ps) %.3f [%.3f,%.3f]'%(*pct(R[s]['fl_ss']),*pct(R[s]['fl_pp']),*pct(R[s]['fl_ps']),*pct(gap)))
print('\nTOP-1 DRIVER')
for s in ST:
    o=obs[s];c=cb[s]
    gap=np.array(R[s]['dr_ss'])-np.array(R[s]['dr_ps'])
    print(f"{s:10} agree {o['agree1']:.3f} cust-boot CI [{c['agree'][0]:.3f},{c['agree'][1]:.3f}] n={o['n_train']}")
    print('   refit floors  stage-vs-stage %.3f [%.3f,%.3f]  pooled-vs-pooled %.3f [%.3f,%.3f]  pooled-vs-stage(refit) %.3f [%.3f,%.3f]  gap(ss-ps) %.3f [%.3f,%.3f]'%(*pct(R[s]['dr_ss']),*pct(R[s]['dr_pp']),*pct(R[s]['dr_ps']),*pct(gap)))

print('\nSTAGE-LEVEL: share of refit replicates with the same leading driver; mean rank correlation of mean-|SHAP| vectors')
for s_ in ST:
    r=R[s_]; print(f"{s_:10} same leading driver: stage-vs-stage {np.mean(r['lv_ss']):.2f}  pooled-vs-stage {np.mean(r['lv_ps']):.2f} | rho stage-vs-stage {pct(r['rho_ss'])[0]:.3f} [{pct(r['rho_ss'])[1]:.3f},{pct(r['rho_ss'])[2]:.3f}]  pooled-vs-stage {pct(r['rho_ps'])[0]:.3f} [{pct(r['rho_ps'])[1]:.3f},{pct(r['rho_ps'])[2]:.3f}]")
